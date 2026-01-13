"""Slash command handlers for the CLI."""

from __future__ import annotations

import sys
import termios
import tty
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ptc_cli.core import console, get_syntax_theme
from ptc_cli.display import show_help
from ptc_cli.sandbox.recovery import is_sandbox_error, recover_sandbox

if TYPE_CHECKING:
    from typing import Protocol

    from ptc_cli.core.state import SessionState
    from ptc_cli.display.tokens import TokenTracker

    class _PTCAgent(Protocol):
        """Protocol for PTCAgent type hint."""

        checkpointer: Any

    class _Sandbox(Protocol):
        """Protocol for Sandbox type hint."""

        def glob_files(self, pattern: str, path: str) -> list[str]: ...
        def normalize_path(self, path: str) -> str: ...
        def read_file(self, path: str) -> str | None: ...
        def download_file_bytes(self, path: str) -> bytes | None: ...
        async def execute_bash_command(
            self, command: str, working_dir: str = ..., timeout: int = ..., *, background: bool = ...,  # noqa: ASYNC109
        ) -> dict[str, Any]: ...

    class _SessionManager(Protocol):
        """Protocol for SessionManager type hint."""

        sandbox: _Sandbox | None

        async def get_sandbox(self) -> _Sandbox | None: ...

    class _ModelSwitchContext(Protocol):
        """Protocol for model switch context."""

        agent_config: Any
        ptc_agent: Any
        session: Any
        agent_ref: dict[str, Any]
        checkpointer: Any

        async def recreate_agent(self) -> None:
            """Recreate the agent with new model."""
            ...

# Directories to exclude by default (system/internal dirs)
EXCLUDED_DIRS = {"code", "tools", "mcp_servers"}
HOME_PREFIX = "/home/daytona/"


class SandboxRecoveryError(Exception):
    """Raised when sandbox recovery fails after disconnection."""

    def __init__(self) -> None:
        """Initialize with default message."""
        super().__init__("Failed to recover sandbox")


def _normalize_path(path: str) -> str:
    """Remove /home/daytona/ prefix for cleaner display."""
    if path.startswith(HOME_PREFIX):
        return path[len(HOME_PREFIX):]
    return path


def _render_tree(files: list[str]) -> list[str]:
    """Render file list as a tree structure.

    Args:
        files: List of file paths

    Returns:
        List of formatted tree lines
    """
    # Build directory structure
    tree: dict = {}
    for filepath in sorted(files):
        parts = filepath.split("/")
        current = tree
        for part in parts:
            current = current.setdefault(part, {})

    # Render tree recursively
    lines: list[str] = []

    def render_node(node: dict, prefix: str = "") -> None:
        items = list(node.items())
        for i, (name, children) in enumerate(items):
            is_last_item = i == len(items) - 1
            connector = "└── " if is_last_item else "├── "
            lines.append(f"{prefix}{connector}{name}")

            if children:
                extension = "    " if is_last_item else "│   "
                render_node(children, prefix + extension)

    render_node(tree)
    return lines


async def _execute_with_recovery[T](
    session: _SessionManager,
    operation_name: str,
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Execute a sandbox operation with automatic recovery on failure.

    Args:
        session: The PTC session
        operation_name: Description of operation for error messages
        operation: Async callable that performs the sandbox operation

    Returns:
        Result of the operation (which may be None for operations like read_file)

    Raises:
        SandboxRecoveryError: If sandbox recovery fails
        Exception: If error is not sandbox-related
    """
    try:
        return await operation()
    except Exception as e:
        if is_sandbox_error(str(e)):
            console.print(f"[yellow]Sandbox disconnected while {operation_name}[/yellow]")
            if await recover_sandbox(session, console):
                console.print()
                # Retry once after recovery
                return await operation()
            raise SandboxRecoveryError from e
        raise


async def _handle_files_command(session: _SessionManager | None, *, show_all: bool) -> None:
    """Handle the /files command to list files in sandbox.

    Args:
        session: The PTC session
        show_all: Whether to show all files including system directories
    """
    if not session or not session.sandbox:
        console.print("[yellow]No active sandbox session[/yellow]")
        return

    async def _list_files() -> list[str]:
        sandbox = await session.get_sandbox()
        return sandbox.glob_files("**/*", path=".")  # type: ignore[union-attr]

    try:
        files = await _execute_with_recovery(session, "listing files", _list_files)
    except SandboxRecoveryError:
        return

    # Normalize paths first (remove /home/daytona/ prefix)
    normalized_files = [_normalize_path(f) for f in files]

    if not show_all:
        # Filter out excluded directories (on normalized paths)
        normalized_files = [
            f
            for f in normalized_files
            if not any(f.startswith(d + "/") or f == d for d in EXCLUDED_DIRS)
        ]

    if not normalized_files:
        console.print("[dim]No files found[/dim]")
        if not show_all:
            console.print("[dim]Use /files all to include system directories[/dim]")
    else:
        console.print(f"[bold]Files ({len(normalized_files)}):[/bold]")
        tree_lines = _render_tree(normalized_files)
        for line in tree_lines:
            console.print(f"  {line}")
        if not show_all:
            console.print()
            console.print("[dim]Use /files all to include system directories[/dim]")
    console.print()


async def _handle_view_command(session: _SessionManager | None, path: str) -> None:
    """Handle the /view command to view a file from sandbox.

    Args:
        session: The PTC session
        path: Path to the file to view
    """
    if not path:
        console.print("[yellow]Usage: /view <path>[/yellow]")
        return

    if not session or not session.sandbox:
        console.print("[yellow]No active sandbox session[/yellow]")
        return

    # Check if image file - auto-download instead of terminal rendering
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
    if path.lower().endswith(image_extensions):

        async def _download_image() -> bytes | None:
            sandbox = await session.get_sandbox()
            sandbox_path = sandbox.normalize_path(path)  # type: ignore[union-attr]
            return sandbox.download_file_bytes(sandbox_path)  # type: ignore[union-attr]

        try:
            image_bytes = await _execute_with_recovery(session, "downloading image", _download_image)
        except SandboxRecoveryError:
            return
        if image_bytes:
            local_path = Path.cwd() / Path(path).name
            local_path.write_bytes(image_bytes)
            console.print(f"[green]Image downloaded to: {local_path}[/green]")
        else:
            console.print(f"[red]Failed to download image: {path}[/red]")
    else:
        # Text file - use Rich Syntax highlighting
        async def _read_file() -> str | None:
            sandbox = await session.get_sandbox()
            sandbox_path = sandbox.normalize_path(path)  # type: ignore[union-attr]
            return sandbox.read_file(sandbox_path)  # type: ignore[union-attr]

        try:
            content = await _execute_with_recovery(session, "reading file", _read_file)
        except SandboxRecoveryError:
            return
        if content is None:
            console.print(f"[red]File not found: {path}[/red]")
        else:
            from rich.syntax import Syntax

            ext = Path(path).suffix.lstrip(".") or "text"
            syntax = Syntax(content, ext, theme=get_syntax_theme(), line_numbers=True)
            console.print()
            console.print(syntax)
            console.print()


async def _handle_copy_command(session: _SessionManager | None, path: str) -> None:
    """Handle the /copy command to copy a file to clipboard.

    Args:
        session: The PTC session
        path: Path to the file to copy
    """
    if not path:
        console.print("[yellow]Usage: /copy <path>[/yellow]")
        return

    if not session or not session.sandbox:
        console.print("[yellow]No active sandbox session[/yellow]")
        return

    async def _read_file() -> str | None:
        sandbox = await session.get_sandbox()
        sandbox_path = sandbox.normalize_path(path)  # type: ignore[union-attr]
        return sandbox.read_file(sandbox_path)  # type: ignore[union-attr]

    try:
        content = await _execute_with_recovery(session, "reading file", _read_file)
    except SandboxRecoveryError:
        return
    if content is None:
        console.print(f"[red]File not found: {path}[/red]")
    else:
        try:
            import pyperclip

            pyperclip.copy(content)
            console.print(f"[green]Copied {len(content)} chars to clipboard[/green]")
        except ImportError:
            console.print("[yellow]Clipboard requires pyperclip package[/yellow]")
        except (OSError, RuntimeError) as e:
            # OSError: clipboard access issues, RuntimeError: pyperclip errors
            console.print(f"[red]Clipboard error: {e}[/red]")


def _prompt_model_selection(  # noqa: PLR0911
    models: list[tuple[str, Any]],
    current_model: str,
) -> str | None:
    """Prompt user to select a model with arrow key navigation.

    Args:
        models: List of (name, definition) tuples
        current_model: Name of the currently selected model

    Returns:
        Selected model name, or None if cancelled
    """
    if not models:
        return None

    # Find index of current model
    selected = 0
    for i, (name, _) in enumerate(models):
        if name == current_model:
            selected = i
            break

    try:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            # Hide cursor during menu interaction
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()

            first_render = True
            num_lines = len(models)

            while True:
                if not first_render:
                    # Move cursor back to start of menu
                    sys.stdout.write(f"\033[{num_lines}A\r")

                first_render = False

                # Display options
                for i, (name, definition) in enumerate(models):
                    sys.stdout.write("\r\033[K")  # Clear line

                    is_current = name == current_model
                    provider = getattr(definition, "provider", "")

                    if i == selected:
                        # Selected item - bold
                        if is_current:
                            # Current model highlighted in green
                            sys.stdout.write(f"\033[1;32m[x] {name} ({provider}) *\033[0m\n")
                        else:
                            sys.stdout.write(f"\033[1;36m[x] {name} ({provider})\033[0m\n")
                    elif is_current:
                        # Current but not selected - dim green
                        sys.stdout.write(f"\033[2;32m[ ] {name} ({provider}) *\033[0m\n")
                    else:
                        # Not selected - dim
                        sys.stdout.write(f"\033[2m[ ] {name} ({provider})\033[0m\n")

                sys.stdout.flush()

                # Read key
                char = sys.stdin.read(1)

                if char == "\x1b":  # ESC sequence
                    next1 = sys.stdin.read(1)
                    if next1 == "[":
                        next2 = sys.stdin.read(1)
                        if next2 == "B":  # Down arrow
                            selected = (selected + 1) % len(models)
                        elif next2 == "A":  # Up arrow
                            selected = (selected - 1) % len(models)
                    else:
                        # Plain ESC - cancel
                        sys.stdout.write("\r\n")
                        return None
                elif char in {"\r", "\n"}:  # Enter
                    sys.stdout.write("\r\n")
                    return models[selected][0]
                elif char == "\x03":  # Ctrl+C
                    sys.stdout.write("\r\n")
                    raise KeyboardInterrupt

        finally:
            # Show cursor again
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except (termios.error, AttributeError):
        # Fallback for non-Unix systems
        console.print("\nAvailable models:")
        for i, (name, definition) in enumerate(models, 1):
            provider = getattr(definition, "provider", "")
            marker = " *" if name == current_model else ""
            console.print(f"  {i}. {name} ({provider}){marker}")

        try:
            choice = input("\nEnter number (or press Enter to cancel): ").strip()
            if not choice:
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx][0]
        except (ValueError, EOFError):
            pass
        return None

    return None


async def _handle_model_command(  # noqa: PLR0911
    agent: _PTCAgent,
    session_state: SessionState,
    model_switch_context: _ModelSwitchContext | None,
) -> str | None:
    """Handle the /model command to switch LLM models.

    Args:
        agent: The agent instance (for checkpointer access)
        session_state: Session state for thread_id
        model_switch_context: Context for model switching

    Returns:
        "handled" if command completed, "model_switched" if model was changed
    """
    if model_switch_context is None:
        console.print("[yellow]Model switching not available in this context.[/yellow]")
        return "handled"

    # Check for existing conversation by querying checkpointer
    checkpointer = getattr(agent, "checkpointer", None)
    if checkpointer:
        try:
            config = {"configurable": {"thread_id": session_state.thread_id}}
            state = checkpointer.get_tuple(config)
            if state is not None:
                console.print()
                console.print(
                    "[yellow]Cannot switch models during an active conversation.[/yellow]"
                )
                console.print(
                    "[dim]Use /clear to start a new conversation, then /model to switch.[/dim]"
                )
                console.print()
                return "handled"
        except Exception:  # noqa: BLE001, S110
            # If checkpointer query fails, allow model switch
            pass

    # Load LLM catalog
    try:
        from ptc_agent.config.loaders import load_llm_catalog

        llm_catalog = await load_llm_catalog()
    except FileNotFoundError:
        console.print("[red]llms.json not found. Cannot switch models.[/red]")
        return "handled"
    except ValueError as e:
        console.print(f"[red]Error loading llms.json: {e}[/red]")
        return "handled"

    # Get current model name
    current_model = model_switch_context.agent_config.llm.name

    # Build model list
    models = list(llm_catalog.items())

    if not models:
        console.print("[yellow]No models available in llms.json[/yellow]")
        return "handled"

    console.print()
    console.print("[bold]Select a model (↑/↓ to navigate, Enter to select, Esc to cancel):[/bold]")
    console.print()

    # Prompt for selection
    try:
        selected_name = _prompt_model_selection(models, current_model)
    except KeyboardInterrupt:
        console.print("[dim]Cancelled[/dim]")
        return "handled"

    if selected_name is None:
        console.print("[dim]Cancelled[/dim]")
        return "handled"

    if selected_name == current_model:
        console.print(f"[dim]Already using {selected_name}[/dim]")
        return "handled"

    # Switch the model
    console.print(f"[cyan]Switching to {selected_name}...[/cyan]")

    try:
        # Update config with new LLM definition
        config = model_switch_context.agent_config
        selected_definition = llm_catalog[selected_name]

        config.llm_definition = selected_definition
        config.llm.name = selected_name
        config.llm_client = None  # Clear cached client

        # Recreate the agent
        await model_switch_context.recreate_agent()

        console.print(f"[green]Switched to {selected_name}[/green]")
        console.print()
        return "model_switched"  # noqa: TRY300

    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed to switch model: {e}[/red]")
        return "handled"


async def _handle_download_command(
    session: _SessionManager | None, user_path: str, local_path_str: str
) -> None:
    """Handle the /download command to download a file from sandbox.

    Args:
        session: The PTC session
        user_path: Path in the sandbox
        local_path_str: Local path to save the file
    """
    if not user_path:
        console.print("[yellow]Usage: /download <sandbox_path> [local_path][/yellow]")
        return

    if not session or not session.sandbox:
        console.print("[yellow]No active sandbox session[/yellow]")
        return

    # Expand ~ and make absolute for local path
    local_path = Path(local_path_str).expanduser()
    if not local_path.is_absolute():
        local_path = Path.cwd() / local_path

    # Use bytes for binary files, text for others
    binary_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
    )
    try:
        if user_path.lower().endswith(binary_extensions):

            async def _download_binary() -> bytes | None:
                sandbox = await session.get_sandbox()
                sandbox_path = sandbox.normalize_path(user_path)  # type: ignore[union-attr]
                return sandbox.download_file_bytes(sandbox_path)  # type: ignore[union-attr]

            binary_content = await _execute_with_recovery(session, "downloading file", _download_binary)
            if binary_content:
                local_path.write_bytes(binary_content)
                console.print(f"[green]Downloaded to: {local_path}[/green]")
            else:
                console.print(f"[red]Failed to download: {user_path}[/red]")
        else:

            async def _download_text() -> str | None:
                sandbox = await session.get_sandbox()
                sandbox_path = sandbox.normalize_path(user_path)  # type: ignore[union-attr]
                return sandbox.read_file(sandbox_path)  # type: ignore[union-attr]

            text_content = await _execute_with_recovery(session, "downloading file", _download_text)
            if text_content:
                local_path.write_text(text_content)
                console.print(f"[green]Downloaded to: {local_path}[/green]")
            else:
                console.print(f"[red]Failed to download: {user_path}[/red]")
    except SandboxRecoveryError:
        return
    except (OSError, UnicodeDecodeError) as e:
        # OSError: file I/O errors, UnicodeDecodeError: text encoding issues
        console.print(f"[red]Download error: {e}[/red]")


async def handle_command(
    command: str,
    agent: _PTCAgent,
    token_tracker: TokenTracker,
    session_state: SessionState,
    session: _SessionManager | None = None,
    model_switch_context: _ModelSwitchContext | None = None,
) -> str | None:
    """Handle slash commands.

    Args:
        command: The command string (e.g., "/help")
        agent: The agent instance
        token_tracker: Token tracker for usage display
        session_state: Session state for conversation management
        session: PTC session with sandbox access (optional)
        model_switch_context: Context for model switching (optional)

    Returns:
        "exit" if should exit, "handled" if command was processed, None otherwise
    """
    cmd = command.strip()
    cmd_lower = cmd.lower()

    # Exit command is the only one that returns "exit"
    if cmd_lower in ("/exit", "/q"):
        return "exit"

    # All other commands return "handled", so we can use a single return at the end
    if cmd_lower == "/help":
        show_help()
    elif cmd_lower == "/clear":
        # Reset conversation by generating new thread_id
        session_state.reset_thread()
        console.clear()

        # Clear sandbox directories if session available
        if session and session.sandbox:

            async def _clear_sandbox_dirs() -> bool:
                sandbox = await session.get_sandbox()
                dirs_to_clear = ["data", "results", "code", "large_tool_results"]
                # Use find -delete to avoid glob expansion issues with set -e
                for dir_name in dirs_to_clear:
                    await sandbox.execute_bash_command(  # type: ignore[union-attr]
                        f"find /home/daytona/{dir_name} -mindepth 1 -delete 2>/dev/null || true"
                    )
                return True

            try:
                await _execute_with_recovery(session, "clearing sandbox", _clear_sandbox_dirs)
                console.print("[green]Conversation and sandbox files cleared.[/green]")
            except SandboxRecoveryError:
                console.print("[yellow]Conversation cleared. Sandbox files could not be cleared.[/yellow]")
        else:
            console.print("[green]Conversation cleared.[/green]")
        console.print()
    elif cmd_lower == "/tokens":
        token_tracker.display()
    elif cmd_lower == "/files" or cmd_lower.startswith("/files "):
        show_all = "all" in cmd_lower  # /files all
        await _handle_files_command(session, show_all=show_all)
    elif cmd_lower.startswith("/view "):
        path = cmd[6:].strip()
        await _handle_view_command(session, path)
    elif cmd_lower.startswith("/copy "):
        path = cmd[6:].strip()
        await _handle_copy_command(session, path)
    elif cmd_lower.startswith("/download "):
        parts = cmd[10:].strip().split(maxsplit=1)
        user_path = parts[0] if parts else ""
        local_path_str = parts[1] if len(parts) > 1 else Path(user_path).name
        await _handle_download_command(session, user_path, local_path_str)
    elif cmd_lower == "/model":
        return await _handle_model_command(agent, session_state, model_switch_context)
    else:
        # Unknown command
        console.print(f"[yellow]Unknown command: {command}[/yellow]")
        console.print(
            "[dim]Available: /help, /clear, /tokens, /files, /view, /copy, /download, /model, /exit[/dim]"
        )

    return "handled"
