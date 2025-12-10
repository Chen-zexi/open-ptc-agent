"""File operation tools: read, write, edit."""

import asyncio
from typing import Any

import structlog
from langchain_core.tools import tool

logger = structlog.get_logger(__name__)


def create_filesystem_tools(sandbox: Any) -> tuple:
    """Factory function to create all filesystem tools (Read, Write, Edit).

    Args:
        sandbox: PTCSandbox instance

    Returns:
        Tuple of (read_file, write_file, edit_file) tools
    """

    @tool
    async def read_file(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
        """Read a file with line numbers (cat -n format).

        Args:
            file_path: Path to file (relative or absolute)
            offset: Start line, 1-indexed (optional)
            limit: Number of lines (optional)

        Returns:
            File contents with line numbers, or ERROR
        """
        try:
            # Normalize virtual path to absolute sandbox path
            normalized_path = sandbox.normalize_path(file_path)
            logger.info("Reading file", file_path=file_path, normalized_path=normalized_path, offset=offset, limit=limit)

            # Validate normalized path
            if sandbox.config.filesystem.enable_path_validation and not sandbox.validate_path(normalized_path):
                error_msg = f"Access denied: {file_path} is not in allowed directories"
                logger.error(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            # Read file content with optional offset/limit using normalized path
            if offset is not None or limit is not None:
                content = await asyncio.to_thread(sandbox.read_file_range, normalized_path, offset or 1, limit or 2000)
            else:
                content = await asyncio.to_thread(sandbox.read_file, normalized_path)

            if content is None:
                error_msg = f"File not found: {file_path}"
                logger.warning(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            # Format with line numbers in cat -n format
            lines = content.splitlines()
            start_line = offset or 1
            formatted_lines = []

            for i, line in enumerate(lines):
                line_num = start_line + i
                # Format: "     1→content" (right-aligned line number with arrow)
                formatted_line = f"{line_num:>6}→{line}"
                formatted_lines.append(formatted_line)

            result = "\n".join(formatted_lines)

            logger.info(
                "File read successfully",
                file_path=file_path,
                size=len(content),
                lines=len(lines),
            )

            return result

        except Exception as e:
            error_msg = f"Failed to read file: {e!s}"
            logger.exception(error_msg, file_path=file_path)
            return f"ERROR: {error_msg}"

    @tool
    async def write_file(file_path: str, content: str) -> str:
        """Write content to a file. Overwrites existing.

        Use Read tool first for existing files. Prefer Edit over Write.

        Args:
            file_path: Path to file
            content: Content to write

        Returns:
            Confirmation or ERROR
        """
        try:
            # Normalize virtual path to absolute sandbox path
            normalized_path = sandbox.normalize_path(file_path)
            logger.info("Writing file", file_path=file_path, normalized_path=normalized_path, size=len(content))

            # Validate normalized path
            if sandbox.config.filesystem.enable_path_validation and not sandbox.validate_path(normalized_path):
                error_msg = f"Access denied: {file_path} is not in allowed directories"
                logger.error(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            # Write file using normalized path
            success = await asyncio.to_thread(sandbox.write_file, normalized_path, content)

            if success:
                bytes_written = len(content.encode("utf-8"))
                # Return virtual path in success message
                virtual_path = sandbox.virtualize_path(normalized_path)
                logger.info(
                    "File written successfully",
                    file_path=virtual_path,
                    bytes_written=bytes_written,
                )
                return f"Wrote {bytes_written} bytes to {virtual_path}"
            error_msg = "Write operation failed"
            logger.error(error_msg, file_path=file_path)
            return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = f"Failed to write file: {e!s}"
            logger.error(error_msg, file_path=file_path, error=str(e), exc_info=True)
            return f"ERROR: {error_msg}"

    @tool
    async def edit_file(
        file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace exact string in a file. Must Read file first.

        Args:
            file_path: Path to file
            old_string: Text to find (must be unique unless replace_all)
            new_string: Replacement text
            replace_all: Replace all occurrences (default: False)

        Returns:
            Confirmation or ERROR

        Note: Preserve exact indentation from Read output. Exclude line number prefix.
        """
        try:
            # Normalize virtual path to absolute sandbox path
            normalized_path = sandbox.normalize_path(file_path)
            logger.info(
                "Editing file",
                file_path=file_path,
                normalized_path=normalized_path,
                old_string_preview=old_string[:50],
                replace_all=replace_all,
            )

            # Validate normalized path
            if sandbox.config.filesystem.enable_path_validation and not sandbox.validate_path(normalized_path):
                error_msg = f"Access denied: {file_path} is not in allowed directories"
                logger.error(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            # Edit file using normalized path
            result = await asyncio.to_thread(sandbox.edit_file, normalized_path, old_string, new_string, replace_all)

            if not result.get("success", False):
                error_msg = result.get("error", "Edit operation failed")
                logger.error(error_msg, file_path=file_path)
                return f"ERROR: {error_msg}"

            # Return success message
            message = result.get("message", "File edited successfully")
            logger.info("File edited successfully", file_path=file_path)
            return message

        except Exception as e:
            error_msg = f"Failed to edit file: {e!s}"
            logger.error(error_msg, file_path=file_path, error=str(e), exc_info=True)
            return f"ERROR: {error_msg}"

    return read_file, write_file, edit_file
