"""Main entry point and CLI loop for ptc-agent.

This module provides the command-line interface for the PTC Agent, including:
- Command-line argument parsing
- Agent initialization and session management
- Interactive CLI loop with prompt handling
- Dependency checking and logging setup
"""

import argparse
import asyncio
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ptc_agent.agent.agent import PTCAgent
    from ptc_agent.agent.middleware.background.orchestrator import BackgroundSubagentOrchestrator
    from ptc_agent.config.agent import AgentConfig
    from ptc_agent.core.session import Session

    from ptc_cli.core.state import SessionState


@dataclass
class ModelSwitchContext:
    """Context for model switching during a session."""

    agent_config: "AgentConfig"
    ptc_agent: "PTCAgent"
    session: "Session"
    agent_ref: dict[str, Any] = field(default_factory=dict)
    checkpointer: Any = None

    async def recreate_agent(self) -> None:
        """Recreate the agent with current config after model switch."""
        from langgraph.checkpoint.memory import InMemorySaver

        # Create new checkpointer
        new_checkpointer = InMemorySaver()

        # Get the new LLM from updated config
        new_llm = self.agent_config.get_llm_client()

        # Recreate the agent with new LLM
        assert self.session.sandbox is not None, "Session sandbox must be initialized"  # noqa: S101
        assert self.session.mcp_registry is not None, "Session MCP registry must be initialized"  # noqa: S101
        new_agent = self.ptc_agent.create_agent(
            sandbox=self.session.sandbox,
            mcp_registry=self.session.mcp_registry,
            subagent_names=self.agent_config.subagents_enabled,
            checkpointer=new_checkpointer,
            llm=new_llm,
        )

        # Store checkpointer reference
        new_agent.checkpointer = new_checkpointer
        self.checkpointer = new_checkpointer

        # Update agent reference
        self.agent_ref["agent"] = new_agent


def setup_logging() -> None:
    """Redirect logging to file for cleaner CLI experience.

    Logs are written to ~/.ptc-agent/logs/ptc-agent.log with rotation.
    """
    import logging.handlers

    # Create log directory
    log_dir = Path.home() / ".ptc-agent" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ptc-agent.log"

    # Configure file handler with rotation (10MB max, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    # Remove all existing handlers from root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Add file handler only
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)

    # Suppress specific noisy loggers that bypass structlog
    for logger_name in [
        "mcp",
        "mcp.server",
        "mcp.client",
        "httpx",
        "httpcore",
        "urllib3",
        "asyncio",
        "anyio",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Configure structlog to use standard logging (outputs to file)
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    except ImportError:
        pass


def check_cli_dependencies() -> None:
    """Check if CLI dependencies are installed.

    Verifies that all required packages (rich, prompt-toolkit, python-dotenv)
    are available. Exits with an error message if any are missing.

    Raises:
        SystemExit: If any required dependencies are not installed.
    """
    missing = []

    # Check for rich using importlib.util.find_spec
    if importlib.util.find_spec("rich") is None:
        missing.append("rich")

    # Check for prompt_toolkit using importlib.util.find_spec
    if importlib.util.find_spec("prompt_toolkit") is None:
        missing.append("prompt-toolkit")

    # Check for dotenv using importlib.util.find_spec
    if importlib.util.find_spec("dotenv") is None:
        missing.append("python-dotenv")

    if missing:
        print("\nMissing required CLI dependencies!")
        print("\nThe following packages are required to use the ptc-agent CLI:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install them with:")
        print("  pip install ptc-cli")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - command: Subcommand to run (list, help, reset, or None for interactive)
            - agent: Agent identifier for memory storage
            - auto_approve: Whether to auto-approve tool usage
            - sandbox_id: Optional existing sandbox ID to reuse
            - no_splash: Whether to disable startup splash screen
            - new_sandbox: Whether to create new sandbox (disable session persistence)
            - plan_mode: Whether to enable plan mode
    """
    parser = argparse.ArgumentParser(
        description="PTC Agent - Programmatic Tool Calling AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List all available agents")

    # Help command
    subparsers.add_parser("help", help="Show help information")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset an agent")
    reset_parser.add_argument("--agent", required=True, help="Name of agent to reset")
    reset_parser.add_argument("--target", dest="source_agent", help="Copy prompt from another agent")

    # Default interactive mode
    parser.add_argument(
        "--agent",
        default="agent",
        help="Agent identifier for separate memory stores (default: agent).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve tool usage without prompting (disables human-in-the-loop)",
    )
    parser.add_argument(
        "--sandbox-id",
        help="Existing Daytona sandbox ID to reuse (skips creation and cleanup)",
    )
    parser.add_argument(
        "--no-splash",
        action="store_true",
        help="Disable the startup splash screen",
    )
    parser.add_argument(
        "--new-sandbox",
        action="store_true",
        help="Create new sandbox (don't reuse existing session)",
    )
    parser.add_argument(
        "--plan-mode",
        action="store_true",
        help="Enable plan mode: agent must submit a plan for approval before execution",
    )

    return parser.parse_args()


async def simple_cli(
    agent: "BackgroundSubagentOrchestrator",
    session: "Session",
    assistant_id: str | None,
    session_state: "SessionState",
    baseline_tokens: int = 0,
    *,
    no_splash: bool = False,
    ptc_agent: "PTCAgent | None" = None,
    config: "AgentConfig | None" = None,
) -> None:
    """Main CLI loop.

    Provides an interactive command-line interface for interacting with the agent.
    Handles user input, command processing, and agent execution.

    Args:
        agent: The configured agent (BackgroundSubagentOrchestrator)
        session: PTC session with sandbox and MCP registry
        assistant_id: Agent identifier for memory storage
        session_state: Session state with auto-approve and plan mode settings
        baseline_tokens: Base token count for usage tracking
        no_splash: If True, skip displaying the startup splash screen
        ptc_agent: PTCAgent instance for model switching
        config: Agent configuration for model switching

    Note:
        The function runs an infinite loop until the user exits via quit command,
        EOF (Ctrl+D), or KeyboardInterrupt (Ctrl+C).
    """
    from ptc_cli.commands import execute_bash_command, handle_command
    from ptc_cli.core import COLORS, PTC_AGENT_ASCII, console
    from ptc_cli.display import TokenTracker
    from ptc_cli.input import SandboxFileCompleter, create_prompt_session
    from ptc_cli.streaming import execute_task
    from ptc_cli.streaming.errors import get_api_error_message, is_api_error

    if not no_splash:
        console.print(PTC_AGENT_ASCII, style=f"bold {COLORS['primary']}")
        console.print()

    # Display sandbox info
    if session and session.sandbox:
        sandbox_id = getattr(session.sandbox, "sandbox_id", "unknown")
        console.print(f"[yellow]Daytona sandbox: {sandbox_id}[/yellow]")
        console.print()

    console.print(f"[dim]Local directory: {Path.cwd()}[/dim]")
    console.print()

    if session_state.plan_mode:
        console.print("  [cyan]Plan Mode: ON[/cyan] [dim](agent will submit plan for approval before execution)[/dim]")
        console.print()

    if session_state.auto_approve:
        console.print("  [yellow]Auto-approve: ON[/yellow] [dim](tools run without confirmation)[/dim]")
        console.print()

    # Localize modifier names and show key symbols (macOS vs others)
    if sys.platform == "darwin":
        tips = (
            "  Tips: Enter to submit, Option + Enter for newline (or Esc+Enter), "
            "Ctrl+E to open editor, Shift+Tab to toggle plan mode, Ctrl+C to interrupt"
        )
    else:
        tips = (
            "  Tips: Enter to submit, Alt+Enter (or Esc+Enter) for newline, Ctrl+E to open editor, Shift+Tab to toggle plan mode, Ctrl+C to interrupt"
        )
    console.print(tips, style=f"dim {COLORS['dim']}")

    console.print()

    # Create sandbox file completer and populate initial cache
    sandbox_completer = SandboxFileCompleter()
    if session and session.sandbox:
        sandbox = await session.get_sandbox()
        if sandbox is not None:
            files = sandbox.glob_files("**/*", path=".")
            home_prefix = "/home/daytona/"
            normalized = [f.removeprefix(home_prefix) for f in files]
            sandbox_completer.set_files(normalized)

    # Create agent reference (used by toolbar and model switching)
    agent_ref: dict[str, Any] = {"agent": agent}

    # Create prompt session and token tracker
    prompt_session = create_prompt_session(assistant_id, session_state, sandbox_completer, agent_ref)
    token_tracker = TokenTracker()
    token_tracker.set_baseline(baseline_tokens)

    # Create model switch context for /model command
    model_switch_context: ModelSwitchContext | None = None
    if ptc_agent is not None and config is not None:
        model_switch_context = ModelSwitchContext(
            agent_config=config,
            ptc_agent=ptc_agent,
            session=session,
            agent_ref=agent_ref,
            checkpointer=getattr(agent, "checkpointer", None),
        )

    while True:
        try:
            user_input = await prompt_session.prompt_async()
            if session_state.exit_hint_handle:
                session_state.exit_hint_handle.cancel()
                session_state.exit_hint_handle = None
            session_state.exit_hint_until = None
            user_input = user_input.strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("\nGoodbye!", style=COLORS["primary"])
            break

        if not user_input:
            continue

        # Get current agent (may change after /model command)
        current_agent = agent_ref["agent"]

        # Check for slash commands first
        if user_input.startswith("/"):
            result = await handle_command(
                user_input,
                current_agent,
                token_tracker,
                session_state,
                session=session,
                model_switch_context=model_switch_context,
            )
            if result == "exit":
                console.print("\nGoodbye!", style=COLORS["primary"])
                break
            if result:
                # Command was handled, continue to next input
                continue

        # Check for bash commands (!)
        if user_input.startswith("!"):
            sandbox = session.sandbox if session else None
            await execute_bash_command(user_input, sandbox=sandbox)
            continue

        # Handle regular quit keywords
        if user_input.lower() in ["quit", "exit", "q"]:
            console.print("\nGoodbye!", style=COLORS["primary"])
            break

        try:
            # Get background registry from orchestrator for status display
            bg_registry = None
            if hasattr(current_agent, "middleware") and hasattr(current_agent.middleware, "registry"):
                bg_registry = current_agent.middleware.registry

            await execute_task(
                user_input,
                current_agent,
                assistant_id,
                session_state,
                token_tracker,
                session=session,
                sandbox_completer=sandbox_completer,
                background_registry=bg_registry,
            )
        except Exception as e:
            # Safety net for API errors that weren't caught in execute_task
            if is_api_error(e):
                console.print()
                console.print(get_api_error_message(e))
                console.print()
                continue  # Stay in CLI loop
            raise  # Re-raise other exceptions


async def main(
    assistant_id: str,
    session_state: "SessionState",
    sandbox_id: str | None = None,
) -> None:
    """Main entry point with session initialization.

    Initializes the agent, session, and sandbox, then starts the interactive CLI loop.
    Handles errors and cleanup gracefully.

    Args:
        assistant_id: Agent identifier for memory storage
        session_state: Session state with auto-approve and plan mode settings
        sandbox_id: Optional existing sandbox ID to reuse instead of creating new

    Raises:
        SystemExit: On KeyboardInterrupt or unhandled exceptions
    """
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from ptc_cli.agent import create_agent_with_session
    from ptc_cli.core import console

    session = None
    error_occurred = False

    try:
        console.print()

        # Save original stderr file descriptor BEFORE any redirect
        # This is used for the spinner so it stays visible
        original_stderr_fd = os.dup(sys.stderr.fileno())
        original_stderr = os.fdopen(original_stderr_fd, "w")
        progress_console = Console(file=original_stderr)

        # Also save original stdout fd for restoration
        original_stdout_fd = os.dup(sys.stdout.fileno())

        # Redirect stdout/stderr at FD level to capture subprocess output
        log_dir = Path.home() / ".ptc-agent" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        init_log = log_dir / "init.log"

        with init_log.open("a") as log_file:
            # Redirect at file descriptor level (affects subprocesses!)
            os.dup2(log_file.fileno(), sys.stdout.fileno())
            os.dup2(log_file.fileno(), sys.stderr.fileno())

            try:
                # Progress uses original stderr - visible!
                with Progress(
                    SpinnerColumn(spinner_name="dots"),
                    TextColumn("[progress.description]{task.description}"),
                    console=progress_console,
                    transient=True,
                    refresh_per_second=15,
                ) as progress:
                    task = progress.add_task("Loading configuration...", total=None)

                    def update_step(step: str) -> None:
                        progress.update(task, description=step)

                    agent, session, reusing_sandbox, ptc_agent, config = await create_agent_with_session(
                        agent_name=assistant_id,
                        sandbox_id=sandbox_id,
                        persist_session=session_state.persist_session,
                        on_progress=update_step,
                    )
                    session_state.reusing_sandbox = reusing_sandbox
                    progress.update(task, description="Agent ready!")
            finally:
                # Restore original file descriptors
                os.dup2(original_stdout_fd, sys.stdout.fileno())
                os.dup2(original_stderr_fd, sys.stderr.fileno())
                os.close(original_stdout_fd)

        if session_state.reusing_sandbox:
            console.print("[green]✓ Reconnected to existing sandbox[/green]")
        else:
            console.print("[green]✓ Agent initialized[/green]")
        console.print()

        await simple_cli(
            agent,
            session,
            assistant_id,
            session_state,
            no_splash=session_state.no_splash,
            ptc_agent=ptc_agent,
            config=config,
        )

    except KeyboardInterrupt:
        error_occurred = True
        console.print("\n\n[yellow]Interrupted[/yellow]")
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        # Catch all exceptions to provide clean error reporting
        error_occurred = True
        error_message = f"\n[bold red]Error:[/bold red] {e}\n"
        console.print(error_message)
        console.print_exception()
        sys.exit(1)
    finally:
        # Cleanup session with progress spinner
        if session is not None:
            # Skip cleanup if persisting session and no error occurred
            if session_state.persist_session and not error_occurred:
                # Stop sandbox (don't delete) for faster restart next time
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Stopping sandbox...", total=None)
                        await session.stop()
                    console.print("[dim]Sandbox stopped - will resume on next run[/dim]")
                except Exception as e:  # noqa: BLE001
                    # Catch all exceptions during cleanup to avoid masking original error
                    warning_message = f"[yellow]Warning: Could not stop sandbox: {e}[/yellow]"
                    console.print(warning_message)
            else:
                # Full cleanup - delete sandbox
                try:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Cleaning up sandbox...", total=None)
                        await session.cleanup()
                    console.print("[dim]Session cleaned up[/dim]")
                except Exception as e:  # noqa: BLE001
                    # Catch all exceptions during cleanup to avoid masking original error
                    warning_message = f"[yellow]Warning: Cleanup failed: {e}[/yellow]"
                    console.print(warning_message)


def cli_main() -> None:
    """Entry point for console script.

    Main entry function for the ptc-agent CLI. Handles:
    - Platform-specific setup (gRPC fork support on macOS)
    - Dependency checking
    - Logging configuration
    - Command-line argument parsing
    - Command routing (help, list, reset, or interactive mode)

    This function is registered as the console_scripts entry point in pyproject.toml.

    Raises:
        SystemExit: On KeyboardInterrupt or unhandled errors
    """
    # Fix for gRPC fork issue on macOS
    # https://github.com/grpc/grpc/issues/37642
    if sys.platform == "darwin":
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

    # Check dependencies first
    check_cli_dependencies()

    # Redirect logging to file for clean CLI output
    setup_logging()

    # Import after dependency check
    from ptc_cli.agent import list_agents, reset_agent
    from ptc_cli.core import SessionState, console
    from ptc_cli.display import show_help

    try:
        args = parse_args()

        if args.command == "help":
            show_help()
        elif args.command == "list":
            list_agents()
        elif args.command == "reset":
            reset_agent(args.agent, args.source_agent)
        else:
            # Create session state from args
            session_state = SessionState(
                auto_approve=args.auto_approve,
                no_splash=args.no_splash,
                persist_session=not args.new_sandbox,
                plan_mode=args.plan_mode,
            )

            asyncio.run(
                main(
                    args.agent,
                    session_state,
                    args.sandbox_id,
                )
            )
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C - suppress ugly traceback
        console.print("\n\n[yellow]Interrupted[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    cli_main()
