"""Tests for sandbox export functionality."""

import sys
from pathlib import Path

# Add example/ directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, MagicMock
from utils import export_sandbox_files, ExportResult


@pytest.fixture
def mock_sandbox():
    """Create a mock sandbox for testing."""
    sandbox = Mock()
    sandbox.sandbox = Mock()  # Indicate sandbox is initialized
    return sandbox


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    return tmp_path / "test_output"


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_format_bytes(self):
        """Test byte formatting."""
        result = ExportResult(
            success=True,
            output_directory=Path("/tmp/test"),
            timestamp="20250324_120000",
            total_bytes=0,
        )

        assert result._format_bytes(512) == "512 B"
        assert result._format_bytes(1024) == "1.00 KB"
        assert result._format_bytes(1024 * 1024) == "1.00 MB"
        assert result._format_bytes(1024 * 1024 * 1024) == "1.00 GB"

    def test_summary_basic(self):
        """Test basic summary generation."""
        result = ExportResult(
            success=True,
            output_directory=Path("/tmp/test/20250324_120000"),
            timestamp="20250324_120000",
            total_files=5,
            total_bytes=2048,
        )
        result.files_exported = ["file1.txt", "file2.txt"]
        result.directories_processed = ["code", "results"]

        summary = result.summary()
        assert "Export Summary (20250324_120000)" in summary
        assert "Total Files: 5" in summary
        assert "Successfully Exported: 2" in summary
        assert "code" in summary
        assert "results" in summary

    def test_summary_with_failures(self):
        """Test summary with failed files."""
        result = ExportResult(
            success=True,
            output_directory=Path("/tmp/test"),
            timestamp="20250324_120000",
        )
        result.files_failed = [
            {"path": "/home/daytona/results/bad.txt", "error": "File not found"}
        ]

        summary = result.summary()
        assert "Failed Files:" in summary
        assert "bad.txt" in summary
        assert "File not found" in summary


class TestExportSandboxFiles:
    """Tests for export_sandbox_files function."""

    def test_sandbox_not_initialized(self, mock_sandbox, temp_output_dir):
        """Test that uninitialized sandbox raises ValueError."""
        mock_sandbox.sandbox = None

        with pytest.raises(ValueError, match="Sandbox not initialized"):
            export_sandbox_files(mock_sandbox, output_base=str(temp_output_dir))

    def test_output_base_is_file(self, mock_sandbox, tmp_path):
        """Test that output base being a file raises ValueError."""
        # Create a file at the output base location
        output_file = tmp_path / "output.txt"
        output_file.write_text("test")

        with pytest.raises(ValueError, match="is a file"):
            export_sandbox_files(mock_sandbox, output_base=str(output_file))

    def test_basic_export_success(self, mock_sandbox, temp_output_dir):
        """Test successful export of files."""
        # Mock list_directory to return files
        mock_sandbox.list_directory.return_value = [
            {"name": "file1.txt", "type": "file", "path": "/home/daytona/results/file1.txt"},
            {"name": "file2.txt", "type": "file", "path": "/home/daytona/results/file2.txt"},
        ]

        # Mock download methods
        mock_sandbox.download_file_bytes.side_effect = [b"content1", b"content2"]

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["results"],
        )

        assert result.success is True
        assert result.total_files == 2
        assert len(result.files_exported) == 2
        assert len(result.files_failed) == 0
        assert "results" in result.directories_processed

        # Verify files were written
        output_files = list(result.output_directory.rglob("*.txt"))
        assert len(output_files) == 2

    def test_partial_failure_continues(self, mock_sandbox, temp_output_dir):
        """Test that export continues after individual file failures."""
        mock_sandbox.list_directory.return_value = [
            {"name": "good.txt", "type": "file", "path": "/home/daytona/results/good.txt"},
            {"name": "bad.txt", "type": "file", "path": "/home/daytona/results/bad.txt"},
        ]

        # First file succeeds, second fails
        mock_sandbox.download_file_bytes.side_effect = [b"success", None]
        mock_sandbox.read_file.return_value = None  # Fallback also fails

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["results"],
        )

        assert result.success is True  # One file succeeded
        assert result.total_files == 1
        assert len(result.files_exported) == 1
        assert len(result.files_failed) == 1
        assert "bad.txt" in result.files_failed[0]["path"]

    def test_missing_directory_handling(self, mock_sandbox, temp_output_dir):
        """Test graceful handling of non-existent directories."""
        mock_sandbox.list_directory.side_effect = Exception("Directory not found")

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["nonexistent"],
        )

        assert result.success is False
        assert result.total_files == 0
        assert len(result.files_failed) > 0

    def test_empty_directory_handling(self, mock_sandbox, temp_output_dir):
        """Test handling of empty directories."""
        mock_sandbox.list_directory.return_value = []

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["results"],
        )

        assert result.success is False
        assert result.total_files == 0
        assert "results" in result.directories_processed

    def test_recursive_structure(self, mock_sandbox, temp_output_dir):
        """Test export of nested directories."""
        # First call: root directory with subdirectory
        # Second call: subdirectory with file
        mock_sandbox.list_directory.side_effect = [
            [
                {"name": "subdir", "type": "directory", "path": "/home/daytona/results/subdir"},
            ],
            [
                {"name": "file.txt", "type": "file", "path": "/home/daytona/results/subdir/file.txt"},
            ],
        ]

        mock_sandbox.download_file_bytes.return_value = b"nested content"

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["results"],
        )

        assert result.success is True
        assert result.total_files == 1

        # Verify nested structure
        nested_file = result.output_directory / "results" / "subdir" / "file.txt"
        assert nested_file.exists()
        assert nested_file.read_text() == "nested content"

    def test_custom_timestamp_format(self, mock_sandbox, temp_output_dir):
        """Test custom timestamp format."""
        mock_sandbox.list_directory.return_value = []

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            timestamp_format="%Y-%m-%d",
        )

        # Timestamp should be in YYYY-MM-DD format
        assert len(result.timestamp) == 10
        assert result.timestamp.count("-") == 2

    def test_default_directories(self, mock_sandbox, temp_output_dir):
        """Test that default directories are code, data, results."""
        mock_sandbox.list_directory.return_value = []

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
        )

        # Should have tried to process default directories
        assert mock_sandbox.list_directory.call_count == 3

    def test_selective_directories(self, mock_sandbox, temp_output_dir):
        """Test exporting only specified directories."""
        mock_sandbox.list_directory.return_value = [
            {"name": "file.txt", "type": "file", "path": "/home/daytona/results/file.txt"},
        ]
        mock_sandbox.download_file_bytes.return_value = b"content"

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["results"],
        )

        assert result.success is True
        assert "results" in result.directories_processed
        assert len(result.directories_processed) == 1

    def test_fallback_to_read_file(self, mock_sandbox, temp_output_dir):
        """Test fallback from download_file_bytes to read_file."""
        mock_sandbox.list_directory.return_value = [
            {"name": "text.txt", "type": "file", "path": "/home/daytona/results/text.txt"},
        ]

        # download_file_bytes returns None, fallback to read_file
        mock_sandbox.download_file_bytes.return_value = None
        mock_sandbox.read_file.return_value = "text content"

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["results"],
        )

        assert result.success is True
        assert result.total_files == 1

        # Verify file was written with encoded text
        output_file = result.output_directory / "results" / "text.txt"
        assert output_file.exists()
        assert output_file.read_text() == "text content"

    def test_permission_error_handling(self, mock_sandbox, temp_output_dir):
        """Test handling of permission errors."""
        mock_sandbox.list_directory.return_value = [
            {"name": "protected.txt", "type": "file", "path": "/home/daytona/results/protected.txt"},
        ]

        mock_sandbox.download_file_bytes.side_effect = PermissionError("Access denied")

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["results"],
        )

        assert result.success is False
        assert result.total_files == 0
        assert len(result.files_failed) == 1
        assert "Permission denied" in result.files_failed[0]["error"]

    def test_multiple_directories(self, mock_sandbox, temp_output_dir):
        """Test exporting from multiple directories."""
        # list_directory is called twice per directory:
        # 1. Initial check to verify directory exists
        # 2. Recursive discovery within discover_files_recursive
        mock_sandbox.list_directory.side_effect = [
            # First directory: code
            [{"name": "code1.py", "type": "file", "path": "/home/daytona/code/code1.py"}],  # Initial check
            [{"name": "code1.py", "type": "file", "path": "/home/daytona/code/code1.py"}],  # Recursive discovery
            # Second directory: data
            [{"name": "data1.csv", "type": "file", "path": "/home/daytona/data/data1.csv"}],  # Initial check
            [{"name": "data1.csv", "type": "file", "path": "/home/daytona/data/data1.csv"}],  # Recursive discovery
        ]

        mock_sandbox.download_file_bytes.side_effect = [b"code content", b"data content"]

        result = export_sandbox_files(
            mock_sandbox,
            output_base=str(temp_output_dir),
            directories=["code", "data"],
        )

        assert result.success is True
        assert result.total_files == 2
        assert "code" in result.directories_processed
        assert "data" in result.directories_processed

        # Verify both directories were created
        assert (result.output_directory / "code" / "code1.py").exists()
        assert (result.output_directory / "data" / "data1.csv").exists()
