"""Tests for DaytonaBackend."""

from unittest.mock import Mock

import pytest

from ptc_agent.agent.backends.daytona import DaytonaBackend


@pytest.fixture
def mock_sandbox():
    """Create a mock sandbox for testing."""
    sandbox = Mock()
    sandbox.config = Mock()
    sandbox.config.filesystem = Mock()
    sandbox.config.filesystem.enable_path_validation = False
    return sandbox


class TestDaytonaBackendGrepRaw:
    """Tests for DaytonaBackend.grep_raw() method."""

    def test_grep_raw_with_string_list_result(self, mock_sandbox):
        """Test grep_raw when sandbox returns list of strings (ripgrep output).

        This is the bug case - grep_content() returns list of strings like:
        ["/path/file.py:10:matching line text", "/path/file2.py:20:another match"]

        Previously this would fail with: 'str' object has no attribute 'get'
        """
        mock_sandbox.grep_content = Mock(return_value=[
            "/home/daytona/file1.py:10:def hello():",
            "/home/daytona/file2.py:25:def world():"
        ])

        backend = DaytonaBackend(mock_sandbox)
        result = backend.grep_raw("def ", "/")

        # Should parse strings into GrepMatch dicts
        assert len(result) == 2
        assert result[0]["path"] == "/home/daytona/file1.py"
        assert result[0]["line"] == 10
        assert result[0]["text"] == "def hello():"
        assert result[1]["path"] == "/home/daytona/file2.py"
        assert result[1]["line"] == 25
        assert result[1]["text"] == "def world():"

    def test_grep_raw_with_string_result(self, mock_sandbox):
        """Test grep_raw when sandbox returns a single string."""
        mock_sandbox.grep_content = Mock(return_value="/home/daytona/file.py:5:match line")

        backend = DaytonaBackend(mock_sandbox)
        result = backend.grep_raw("match", "/")

        assert len(result) == 1
        assert result[0]["path"] == "/home/daytona/file.py"
        assert result[0]["line"] == 5
        assert result[0]["text"] == "match line"

    def test_grep_raw_with_dict_list_result(self, mock_sandbox):
        """Test grep_raw when sandbox returns list of dicts."""
        mock_sandbox.grep_content = Mock(return_value=[
            {"path": "/home/daytona/file.py", "line": 10, "text": "match"}
        ])

        backend = DaytonaBackend(mock_sandbox)
        result = backend.grep_raw("match", "/")

        assert len(result) == 1
        assert result[0]["path"] == "/home/daytona/file.py"
        assert result[0]["line"] == 10
        assert result[0]["text"] == "match"

    def test_grep_raw_with_empty_result(self, mock_sandbox):
        """Test grep_raw when no matches found."""
        mock_sandbox.grep_content = Mock(return_value=[])

        backend = DaytonaBackend(mock_sandbox)
        result = backend.grep_raw("nomatch", "/")

        assert result == []

    def test_grep_raw_with_invalid_line_number(self, mock_sandbox):
        """Test grep_raw handles invalid line numbers gracefully."""
        mock_sandbox.grep_content = Mock(return_value=[
            "/home/daytona/file.py:notanumber:some text"
        ])

        backend = DaytonaBackend(mock_sandbox)
        result = backend.grep_raw("text", "/")

        # Should handle ValueError and set line to 0
        assert len(result) == 1
        assert result[0]["path"] == "/home/daytona/file.py"
        assert result[0]["line"] == 0
        assert result[0]["text"] == "notanumber:some text"

    def test_grep_raw_with_colons_in_content(self, mock_sandbox):
        """Test grep_raw handles content with colons correctly."""
        mock_sandbox.grep_content = Mock(return_value=[
            "/home/daytona/file.py:15:url = 'http://example.com:8080'"
        ])

        backend = DaytonaBackend(mock_sandbox)
        result = backend.grep_raw("url", "/")

        assert len(result) == 1
        assert result[0]["path"] == "/home/daytona/file.py"
        assert result[0]["line"] == 15
        assert result[0]["text"] == "url = 'http://example.com:8080'"

    def test_grep_raw_with_empty_string_in_list(self, mock_sandbox):
        """Test grep_raw handles empty strings in list."""
        mock_sandbox.grep_content = Mock(return_value=[
            "/home/daytona/file.py:10:match",
            "",  # Empty string should be skipped
            "/home/daytona/file2.py:20:another"
        ])

        backend = DaytonaBackend(mock_sandbox)
        result = backend.grep_raw("match", "/")

        # Should skip empty strings
        assert len(result) == 2
