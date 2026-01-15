"""File operation tools: read, write, edit."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.tools import tool

logger = structlog.get_logger(__name__)


def create_filesystem_tools(sandbox: Any) -> tuple:
    """Factory function to create filesystem tools (Read, Write, Edit)."""

    def _format_cat_n(lines: list[str], *, start_line_number: int) -> str:
        return "\n".join(f"{i:6}\t{line}" for i, line in enumerate(lines, start=start_line_number))

    @tool
    async def read_file(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Read a file with line numbers (cat -n format).

        Args:
            file_path: Path to file (relative or absolute).
            offset: Line offset (0-indexed). Default: 0.
            limit: Maximum number of lines. Default: 2000.

        Returns:
            File contents with line numbers, or ERROR.
        """
        try:
            normalized_path = sandbox.normalize_path(file_path)
            logger.info("Reading file", file_path=file_path, normalized_path=normalized_path, offset=offset, limit=limit)

            if sandbox.config.filesystem.enable_path_validation and not sandbox.validate_path(normalized_path):
                error_msg = f"Access denied: {file_path} is not in allowed directories"
                logger.error(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            start_offset = offset or 0
            max_lines = limit or 2000

            if offset is not None or limit is not None:
                content = await sandbox.aread_file_range(normalized_path, start_offset, max_lines)
            else:
                content = await sandbox.aread_file_text(normalized_path)

            if content is None:
                error_msg = f"File not found: {file_path}"
                logger.warning(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            lines = content.splitlines()
            return _format_cat_n(lines, start_line_number=start_offset + 1)

        except Exception as e:
            error_msg = f"Failed to read file: {e!s}"
            logger.exception(error_msg, file_path=file_path)
            return f"ERROR: {error_msg}"

    @tool
    async def write_file(file_path: str, content: str) -> str:
        """Write content to a file. Overwrites existing."""
        try:
            normalized_path = sandbox.normalize_path(file_path)
            logger.info("Writing file", file_path=file_path, normalized_path=normalized_path, size=len(content))

            if sandbox.config.filesystem.enable_path_validation and not sandbox.validate_path(normalized_path):
                error_msg = f"Access denied: {file_path} is not in allowed directories"
                logger.error(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            success = await sandbox.awrite_file_text(normalized_path, content)
            if not success:
                return "ERROR: Write operation failed"

            bytes_written = len(content.encode("utf-8"))
            virtual_path = sandbox.virtualize_path(normalized_path)
            return f"Wrote {bytes_written} bytes to {virtual_path}"

        except Exception as e:
            error_msg = f"Failed to write file: {e!s}"
            logger.error(error_msg, file_path=file_path, error=str(e), exc_info=True)
            return f"ERROR: {error_msg}"

    @tool
    async def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Replace exact string in a file. Must Read file first."""
        try:
            normalized_path = sandbox.normalize_path(file_path)
            logger.info(
                "Editing file",
                file_path=file_path,
                normalized_path=normalized_path,
                old_string_preview=old_string[:50],
                replace_all=replace_all,
            )

            if sandbox.config.filesystem.enable_path_validation and not sandbox.validate_path(normalized_path):
                error_msg = f"Access denied: {file_path} is not in allowed directories"
                logger.error(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            result = await sandbox.aedit_file_text(normalized_path, old_string, new_string, replace_all=replace_all)
            if not result.get("success", False):
                error_msg = result.get("error", "Edit operation failed")
                return f"ERROR: {error_msg}"

            return str(result.get("message", "File edited successfully"))

        except Exception as e:
            error_msg = f"Failed to edit file: {e!s}"
            logger.error(error_msg, file_path=file_path, error=str(e), exc_info=True)
            return f"ERROR: {error_msg}"

    return read_file, write_file, edit_file
