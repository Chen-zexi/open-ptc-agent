"""Daytona Backend - Implements deepagent SandboxBackendProtocol for Daytona sandbox.

This backend delegates all filesystem and execution operations to PTCSandbox,
enabling deepagent's built-in tools to work with Daytona sandboxes.
"""


import structlog
from deepagents.backends.protocol import EditResult, WriteResult

from ptc_agent.core.sandbox import PTCSandbox

logger = structlog.get_logger(__name__)


class DaytonaBackend:
    """Backend that implements deepagent's SandboxBackendProtocol using Daytona.

    Provides a unified interface for deepagent's FilesystemMiddleware to interact
    with Daytona sandboxes. All operations are delegated to PTCSandbox.
    """

    def __init__(self, sandbox: PTCSandbox, root_dir: str = "/home/daytona", *, virtual_mode: bool = True) -> None:
        """Initialize Daytona backend.

        Args:
            sandbox: PTCSandbox instance for all operations
            root_dir: Root directory for virtual filesystem (default: /workspace)
            virtual_mode: If True, normalize paths relative to root_dir
        """
        self.sandbox = sandbox
        self.root_dir = root_dir.rstrip("/")
        self.virtual_mode = virtual_mode
        logger.info("Initialized DaytonaBackend", root_dir=self.root_dir, virtual_mode=self.virtual_mode)

    def _normalize_path(self, path: str) -> str:
        """Normalize path relative to root_dir when virtual_mode is enabled.

        Converts virtual paths to absolute sandbox paths:
            "/" -> "/home/daytona"
            "/research_request.md" -> "/home/daytona/research_request.md"
            "." -> "/home/daytona"
            "data/file.txt" -> "/home/daytona/data/file.txt"
            "/home/daytona/file.txt" -> "/home/daytona/file.txt" (unchanged)

        Args:
            path: Path to normalize

        Returns:
            Normalized absolute path
        """
        if not self.virtual_mode:
            return path

        if path in (None, "", ".", "/"):
            return self.root_dir

        path = path.strip()

        # Already absolute and in allowed directories - keep as is
        if path.startswith(("/home/daytona", "/tmp")):
            return path

        # Virtual absolute path: /foo -> /home/daytona/foo
        if path.startswith("/"):
            return f"{self.root_dir}{path}"

        # Relative path: foo -> /home/daytona/foo
        return f"{self.root_dir}/{path}"

    def ls_info(self, path: str = ".") -> list[dict]:
        """List directory contents with file information.

        Args:
            path: Directory path to list

        Returns:
            List of FileInfo dicts with path (required), is_dir, size, modified_at
        """
        try:
            normalized_path = self._normalize_path(path)
            entries = self.sandbox.list_directory(normalized_path)
            result = []
            for entry in entries:
                if isinstance(entry, dict):
                    file_info = {
                        "path": entry.get("path") or entry.get("name", ""),
                        "is_dir": entry.get("is_dir", entry.get("type") == "directory"),
                    }
                    if "size" in entry:
                        file_info["size"] = entry["size"]
                    if "modified_at" in entry:
                        file_info["modified_at"] = entry["modified_at"]
                    result.append(file_info)
                else:
                    # If entry is a string, treat as path
                    result.append({"path": str(entry)})
            return result
        except (OSError, ValueError):
            logger.exception(f"Failed to list directory {path}")
            return []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with optional range.

        Args:
            file_path: Path to file
            offset: Line number to start from (0-indexed)
            limit: Number of lines to read

        Returns:
            File content with line numbers, or error string if file not found
        """
        try:
            normalized_path = self._normalize_path(file_path)
            if offset > 0 or limit != 2000:
                content = self.sandbox.read_file_range(normalized_path, offset, limit)
            else:
                content = self.sandbox.read_file(normalized_path)

            if content is None:
                return f"Error: File '{file_path}' not found"
            return content
        except (OSError, FileNotFoundError, PermissionError):
            logger.exception(f"Failed to read file {file_path}")
            return f"Error: File '{file_path}' not found"

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content to file.

        Args:
            file_path: Path to file
            content: Content to write

        Returns:
            WriteResult with path on success, error on failure
        """
        try:
            normalized_path = self._normalize_path(file_path)
            success = self.sandbox.write_file(normalized_path, content)
            if success:
                # files_update=None for external backends (not state-based)
                return WriteResult(path=normalized_path, files_update=None)
            return WriteResult(error=f"Failed to write to '{normalized_path}'")
        except (OSError, PermissionError) as e:
            logger.exception(f"Failed to write file {file_path}")
            return WriteResult(error=str(e))

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False
    ) -> EditResult:
        """Edit file using exact string replacement.

        Args:
            file_path: Path to file
            old_string: Text to replace
            new_string: Replacement text
            replace_all: Replace all occurrences

        Returns:
            EditResult with path and occurrences on success, error on failure
        """
        try:
            normalized_path = self._normalize_path(file_path)
            result = self.sandbox.edit_file(normalized_path, old_string, new_string, replace_all=replace_all)
            if isinstance(result, dict):
                if result.get("success"):
                    return EditResult(
                        path=normalized_path,
                        files_update=None,  # External backend, not state-based
                        occurrences=result.get("occurrences", 1)
                    )
                return EditResult(error=result.get("error", "Edit failed"))
            # If result is not a dict, assume success
            return EditResult(path=normalized_path, files_update=None, occurrences=1)
        except (OSError, FileNotFoundError, PermissionError, ValueError) as e:
            logger.exception(f"Failed to edit file {file_path}")
            return EditResult(error=str(e))

    def grep_raw(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[dict]:
        """Search file contents with regex pattern.

        Args:
            pattern: Regex pattern to search
            path: Path to search in (default: working directory)
            glob: File pattern filter (e.g., "*.py")

        Returns:
            List of GrepMatch dicts with path, line, text, or error string
        """
        try:
            search_path = self._normalize_path(path) if path else self.root_dir
            result = self.sandbox.grep_content(
                pattern=pattern,
                path=search_path,
                output_mode="content",  # Get full content to extract matches
                glob=glob,
                show_line_numbers=True,
            )

            # Convert to GrepMatch format: list of {path, line, text}
            if isinstance(result, str):
                # Parse string output into GrepMatch dicts
                matches = []
                for line in result.strip().split("\n"):
                    if line and ":" in line:
                        # Format: "path:line:text"
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            try:
                                matches.append({
                                    "path": parts[0],
                                    "line": int(parts[1]),
                                    "text": parts[2]
                                })
                            except ValueError:
                                # If line number parsing fails, include as text
                                matches.append({
                                    "path": parts[0],
                                    "line": 0,
                                    "text": ":".join(parts[1:])
                                })
                return matches
            if isinstance(result, list):
                # Already in list format - could be strings or dicts
                matches = []
                for m in result:
                    if isinstance(m, str):
                        # Parse string format "path:line:text"
                        if ":" in m:
                            parts = m.split(":", 2)
                            if len(parts) >= 3:
                                try:
                                    matches.append({
                                        "path": parts[0],
                                        "line": int(parts[1]),
                                        "text": parts[2]
                                    })
                                except ValueError:
                                    matches.append({
                                        "path": parts[0],
                                        "line": 0,
                                        "text": ":".join(parts[1:])
                                    })
                    elif isinstance(m, dict):
                        # Already GrepMatch dict format
                        matches.append({
                            "path": m.get("path", ""),
                            "line": m.get("line", 0),
                            "text": m.get("text", "")
                        })
                return matches
            return []
        except (OSError, ValueError):
            logger.exception("Failed to grep content")
            return []  # Return empty list on error for type consistency

    def glob_info(self, pattern: str, path: str = "/") -> list[dict]:
        """Find files matching glob pattern.

        Args:
            pattern: Glob pattern (e.g., "**/*.py")
            path: Directory to search in

        Returns:
            List of FileInfo dicts with path (required), and optionally is_dir, size, modified_at
        """
        try:
            normalized_path = self._normalize_path(path)
            file_paths = self.sandbox.glob_files(pattern, normalized_path)
            # Convert to FileInfo format
            return [{"path": fp} for fp in file_paths]
        except (OSError, ValueError):
            logger.exception("Failed to glob files")
            return []

    async def execute(self, code: str, timeout: int | None = None) -> dict:
        """Execute Python code in sandbox.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dict with success, stdout, stderr
        """
        try:
            result = await self.sandbox.execute(code, timeout)
            return {
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_id": result.execution_id,
                "files_created": result.files_created,
            }
        except Exception as e:
            logger.exception("Failed to execute code")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
            }

    async def execute_bash(
        self,
        command: str,
        working_dir: str = "/workspace",
        timeout: int = 60
    ) -> dict:
        """Execute bash command in sandbox.

        Args:
            command: Bash command to execute
            working_dir: Working directory
            timeout: Command timeout in seconds

        Returns:
            Dict with success, stdout, stderr, exit_code
        """
        try:
            return await self.sandbox.execute_bash_command(
                command=command,
                working_dir=working_dir,
                timeout=timeout
            )
        except Exception as e:
            logger.exception("Failed to execute bash command")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
            }

    def create_directory(self, dirpath: str) -> bool:
        """Create a directory.

        Args:
            dirpath: Directory path to create

        Returns:
            True if successful
        """
        try:
            normalized_path = self._normalize_path(dirpath)
            return self.sandbox.create_directory(normalized_path)
        except (OSError, PermissionError):
            logger.exception(f"Failed to create directory {dirpath}")
            return False

    def get_work_dir(self) -> str:
        """Get the sandbox working directory.

        Returns:
            Working directory path
        """
        if self.sandbox.sandbox:
            return self.sandbox.sandbox.get_work_dir()
        return "/workspace"
