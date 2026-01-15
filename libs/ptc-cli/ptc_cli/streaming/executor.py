# pyright: reportGeneralTypeIssues=false
"""Task execution and streaming logic for the CLI."""

import asyncio
import os
import select
import sys
import threading
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol

import structlog
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from ptc_cli.core import COLORS, console
from ptc_cli.display import (
    TokenTracker,
    format_tool_display,
    format_tool_message_content,
    render_todo_list,
    truncate_error,
)
from ptc_cli.input import parse_file_mentions
from ptc_cli.sandbox.health import EmptyResultTracker, check_sandbox_health
from ptc_cli.sandbox.recovery import is_sandbox_error, recover_sandbox
from ptc_cli.streaming.errors import get_api_error_message, is_api_error
from ptc_cli.streaming.state import StreamingState
from ptc_cli.streaming.tool_buffer import ToolCallChunkBuffer

termios: Any | None
tty: Any | None
try:
    import termios
    import tty
except ImportError:  # pragma: no cover
    termios = None
    tty = None

if TYPE_CHECKING:
    from ptc_cli.core.state import SessionState

logger = structlog.get_logger(__name__)

# Constants
_MAX_FILE_SIZE = 50000  # Maximum file size to include in context
_CHUNK_TUPLE_SIZE = 3  # Expected size of chunk tuple with subgraphs
_MESSAGE_TUPLE_SIZE = 2  # Expected size of message tuple

# HITL (Human-in-the-Loop) support for plan mode
try:
    from langchain.agents.middleware.human_in_the_loop import HITLRequest
    from langgraph.types import Command
    from pydantic import TypeAdapter, ValidationError

    _HITL_REQUEST_ADAPTER: TypeAdapter[HITLRequest] | None = TypeAdapter(HITLRequest)
    HITL_AVAILABLE = True
except ImportError:
    HITL_AVAILABLE = False
    _HITL_REQUEST_ADAPTER = None
    Command = None  # type: ignore[misc, assignment]


class _EscInterruptWatcher:
    """Watches for ESC key presses during streaming.

    prompt-toolkit keybindings only work while PromptSession is active. During streaming
    we need a separate watcher so users can interrupt with ESC.
    """

    def __init__(self, *, loop: asyncio.AbstractEventLoop, on_escape: Callable[[], None]) -> None:
        self._loop = loop
        self._on_escape = on_escape
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if not sys.stdin.isatty() or termios is None or tty is None:
            return

        self._thread = threading.Thread(target=self._run, name="esc_interrupt_watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)

    def _run(self) -> None:
        if termios is None or tty is None:
            return

        fd = sys.stdin.fileno()
        try:
            old_attrs = termios.tcgetattr(fd)
        except OSError:
            return

        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                readable, _w, _x = select.select([fd], [], [], 0.1)
                if not readable:
                    continue

                ch = os.read(fd, 1)
                if ch == b"\x1b":  # ESC
                    with suppress(Exception):
                        self._loop.call_soon_threadsafe(self._on_escape)
                    return
        finally:
            with suppress(Exception):
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


class _BackgroundTask(Protocol):
    completed: bool
    result_seen: bool
    display_id: str
    asyncio_task: asyncio.Task[object] | None
    result: object | None
    error: str | None


class _BackgroundTaskRegistry(Protocol):
    _tasks: dict[str, _BackgroundTask]


def _format_task_status_line(registry: _BackgroundTaskRegistry) -> Text:
    """Format a status line showing background task states.

    Args:
        registry: BackgroundTaskRegistry instance

    Returns:
        Rich Text object with formatted status
    """
    tasks = list(registry._tasks.values())
    if not tasks:
        return Text("")

    # Sync completion status for all tasks
    for task in tasks:
        if not task.completed and task.asyncio_task and task.asyncio_task.done():
            task.completed = True
            try:
                task.result = task.asyncio_task.result()
            except BaseException as e:  # noqa: BLE001
                task.error = str(e)
                task.result = {"success": False, "error": str(e)}

    running = [t for t in tasks if not t.completed]
    completed = [t for t in tasks if t.completed and not t.result_seen]

    text = Text()

    if running:
        text.append("Running: ", style="dim")
        for i, t in enumerate(running):
            if i > 0:
                text.append(", ", style="dim")
            text.append(t.display_id, style="cyan")
        if completed:
            text.append("  |  ", style="dim")

    if completed:
        text.append("Completed: ", style="dim green")
        for i, t in enumerate(completed):
            if i > 0:
                text.append(", ", style="dim")
            text.append(t.display_id, style="green")

    if running:
        text.append("\n", style="dim")
        text.append("Use task_output() to check progress or results", style="dim")

    return text


async def _display_live_task_status(registry: _BackgroundTaskRegistry, console: Console) -> None:
    """Display live-updating status of background tasks.

    Shows running and completed tasks, updating in real-time until
    all tasks complete or max wait time passes.

    Args:
        registry: BackgroundTaskRegistry instance
        console: Rich Console instance
    """
    # Sync completion status for all tasks first
    for task in registry._tasks.values():
        if not task.completed and task.asyncio_task and task.asyncio_task.done():
            task.completed = True
            try:
                task.result = task.asyncio_task.result()
            except BaseException as e:  # noqa: BLE001
                task.error = str(e)
                task.result = {"success": False, "error": str(e)}

    # Initial check
    running = [t for t in registry._tasks.values() if not t.completed]

    if not running:
        # All already completed, just show static status
        status_text = _format_task_status_line(registry)
        if status_text:
            console.print(status_text)
        return

    # Track initial state BEFORE the loop
    prev_running_count = len(running)
    last_change_time = asyncio.get_event_loop().time()
    max_wait = 30.0

    # Use Live display for real-time updates
    with Live(_format_task_status_line(registry), console=console, refresh_per_second=2, transient=False) as live:
        while True:
            await asyncio.sleep(0.5)

            # Update status (this syncs completion internally via _format_task_status_line)
            status_text = _format_task_status_line(registry)
            live.update(status_text)

            # Count currently running tasks
            curr_running_count = len([t for t in registry._tasks.values() if not t.completed])

            # Check if state changed
            if curr_running_count != prev_running_count:
                last_change_time = asyncio.get_event_loop().time()
                prev_running_count = curr_running_count

            # Exit conditions
            if curr_running_count == 0:
                # All tasks completed
                await asyncio.sleep(0.5)  # Brief pause to show final status
                break

            elapsed = asyncio.get_event_loop().time() - last_change_time
            if elapsed > max_wait:
                # Been waiting too long, stop updating
                break


async def _prompt_for_plan_approval(action_request: dict) -> tuple[dict, str | None]:
    """Show plan and prompt user for approval with arrow key navigation.

    Args:
        action_request: The action request from HITL middleware

    Returns:
        Tuple of (decision dict, feedback string or None)
        - decision: Dict with 'type' key ('approve' or 'reject'), no message field
        - feedback: User feedback for rejection, or None for approval/cancel
    """
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from rich.markdown import Markdown

    from ptc_cli.core import console

    description = action_request.get("description", "No description available")

    # Display the plan for review with markdown rendering
    console.print()
    md_content = Markdown(description)
    console.print(
        Panel(
            md_content,
            title="[bold cyan]📋 Plan Review[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    console.print()

    # Arrow key menu
    options = ["Accept", "Reject with feedback"]
    selected = [0]  # Use list to allow modification in nested function

    def get_menu_text() -> str:
        lines = []
        for i, option in enumerate(options):
            if i == selected[0]:
                lines.append(f"  → {option}")
            else:
                lines.append(f"    {option}")
        lines.append("")
        lines.append("  (↑/↓ to navigate, Enter to select)")
        return "\n".join(lines)

    kb = KeyBindings()

    @kb.add("up")
    def _(_event: object) -> None:
        selected[0] = max(0, selected[0] - 1)

    @kb.add("down")
    def _(_event: object) -> None:
        selected[0] = min(len(options) - 1, selected[0] + 1)

    @kb.add("enter")
    def _(event: "Any") -> None:  # noqa: ANN401
        event.app.exit(result=selected[0])

    @kb.add("c-c")
    def _(event: "Any") -> None:  # noqa: ANN401
        event.app.exit(result=-1)  # Cancelled

    layout = Layout(Window(FormattedTextControl(get_menu_text)))
    app: Application[int] = Application(layout=layout, key_bindings=kb, full_screen=False)

    try:
        result = await app.run_async()
    except KeyboardInterrupt:
        result = -1

    if result == -1:  # Cancelled
        console.print()
        return {"type": "reject"}, "User cancelled"
    if result == 0:  # Accept
        console.print()
        console.print("[green]✓ Plan approved. Starting execution...[/green]")
        return {"type": "approve"}, None
    # Reject with feedback
    console.print()
    try:
        feedback_session: PromptSession[str] = PromptSession()
        feedback = await feedback_session.prompt_async("  Feedback: ")
    except KeyboardInterrupt:
        return {"type": "reject"}, "User cancelled"
    else:
        return {"type": "reject"}, (feedback or "No feedback provided")


async def execute_task(  # noqa: PLR0911  # pyright: ignore[reportGeneralTypeIssues]
    user_input: str,
    agent: "Any",  # noqa: ANN401
    assistant_id: str | None,
    session_state: "SessionState",
    token_tracker: TokenTracker | None = None,
    session: "Any" = None,  # noqa: ANN401
    sandbox_completer: "Any" = None,  # noqa: ANN401
    background_registry: "Any" = None,  # noqa: ANN401
    _retry_count: int = 0,
) -> None:
    """Execute any task by passing it directly to the AI agent.

    Args:
        user_input: User's input text
        agent: The agent to execute the task
        assistant_id: Agent identifier
        session_state: Session state with auto-approve settings
        token_tracker: Optional token tracker
        session: Optional session for sandbox recovery
        sandbox_completer: Optional completer to refresh file cache after task
        background_registry: Optional BackgroundTaskRegistry for status display
        _retry_count: Internal retry counter (do not set manually)
    """
    # Check for completed background tasks and inject notification
    background_notification = None
    if hasattr(agent, "check_and_get_notification"):
        background_notification = await agent.check_and_get_notification()
        if background_notification:
            console.print()
            console.print(
                Panel(
                    Markdown(background_notification),
                    title="[bold cyan]Background Tasks Completed[/bold cyan]",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
            )
            console.print()

    # Parse file mentions and inject content from sandbox
    prompt_text, mentioned_paths = parse_file_mentions(user_input)

    if mentioned_paths and session and session.sandbox:
        sandbox = await session.get_sandbox()
        context_parts = [prompt_text, "\n\n## Referenced Files\n"]
        for path in mentioned_paths:
            try:
                sandbox_path = sandbox.normalize_path(path)
                content = await sandbox.aread_file_text(sandbox_path)
                if content is None:
                    console.print(f"[yellow]Warning: File not found in sandbox: {path}[/yellow]")
                    context_parts.append(f"\n### {path}\n[File not found: {path}]")
                else:
                    # Limit file content to reasonable size
                    if len(content) > _MAX_FILE_SIZE:
                        content = content[:_MAX_FILE_SIZE] + "\n... (file truncated)"
                    context_parts.append(f"\n### {path}\nPath: `{sandbox_path}`\n```\n{content}\n```")
            except Exception as e:  # noqa: BLE001
                context_parts.append(f"\n### {path}\n[Error reading file: {e}]")

        final_input = "\n".join(context_parts)
    elif mentioned_paths and (not session or not session.sandbox):
        console.print("[yellow]Warning: @file mentions require an active sandbox session[/yellow]")
        final_input = prompt_text
    else:
        final_input = prompt_text

    config = {
        "configurable": {"thread_id": session_state.thread_id},
        "metadata": {"assistant_id": assistant_id} if assistant_id else {},
    }

    captured_input_tokens = 0
    captured_output_tokens = 0
    current_todos = None  # Track current todo list state

    # Initialize streaming state
    state = StreamingState(console, f"[bold {COLORS['thinking']}]Agent is thinking...", COLORS)

    # Initialize tool buffer
    tool_buffer = ToolCallChunkBuffer()

    # Initialize empty result tracker
    empty_tracker = EmptyResultTracker()

    tool_icons = {
        "read_file": "📖",
        "write_file": "✏️",
        "edit_file": "✂️",
        "ls": "📁",
        "glob": "🔍",
        "grep": "🔎",
        "shell": "⚡",
        "execute": "🔧",
        "execute_code": "🔧",
        "Bash": "⚡",
        "Read": "📖",
        "Write": "✏️",
        "Edit": "✂️",
        "Glob": "🔍",
        "Grep": "🔎",
        "web_search": "🌐",
        "http_request": "🌍",
        "task": "🤖",
        "wait": "⏳",
        "task_output": "📤",
        "write_todos": "📋",
        "submit_plan": "📋",
    }

    # Build messages - inject plan mode reminder if enabled
    messages = []
    if getattr(session_state, "plan_mode", False):
        messages.append(
            {
                "role": "user",
                "content": (
                    "<system-reminder>You are in Plan Mode. Before executing any write operations "
                    '(Write, Edit, Bash, execute_code), you MUST first call submit_plan(description="...") '
                    "with a detailed description of your plan for user review.</system-reminder>"
                ),
            }
        )
    # Inject background task notification if there is one
    if background_notification:
        messages.append({"role": "user", "content": f"[SYSTEM NOTIFICATION]\n{background_notification}"})
    messages.append({"role": "user", "content": final_input})

    # Stream input - may need to loop if there are interrupts (plan mode)
    # Type as Any since it can be either a dict or Command
    stream_input: Any = {"messages": messages}

    # Allow ESC to interrupt the foreground stream (without exiting the CLI).
    # Ctrl+C remains a hard-exit via SIGINT.
    loop = asyncio.get_running_loop()
    this_task = asyncio.current_task()

    def _on_escape() -> None:
        session_state.esc_interrupt_requested = True
        if this_task is not None:
            this_task.cancel()

    esc_watcher = _EscInterruptWatcher(loop=loop, on_escape=_on_escape)
    esc_watcher.start()

    try:
        while True:  # Interrupt loop for plan mode approval
            interrupt_occurred = False
            pending_interrupts: dict = {}  # {interrupt_id: HITLRequest}
            hitl_response: dict = {}

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates"],
                subgraphs=True,
                config=config,
            ):
                # Unpack chunk - with subgraphs=True and dual-mode, it's (namespace, stream_mode, data)
                if not isinstance(chunk, tuple) or len(chunk) != _CHUNK_TUPLE_SIZE:
                    continue

                namespace, current_stream_mode, data = chunk

                # Skip subagent chunks - only display main agent output
                # Main agent has empty namespace (), subagents have non-empty namespace
                if namespace:
                    # Still capture HITL interrupts from subgraphs (safety)
                    if (
                        current_stream_mode == "updates"
                        and HITL_AVAILABLE
                        and _HITL_REQUEST_ADAPTER
                        and isinstance(data, dict)
                        and "__interrupt__" in data
                    ):
                        interrupts = data["__interrupt__"]
                        if interrupts:
                            for interrupt_obj in interrupts:
                                try:
                                    validated = _HITL_REQUEST_ADAPTER.validate_python(interrupt_obj.value)
                                    pending_interrupts[interrupt_obj.id] = validated
                                    interrupt_occurred = True
                                except ValidationError as e:
                                    logger.warning(
                                        "Invalid HITL request data",
                                        error=str(e),
                                    )

                    # Update spinner to show subagent count
                    if background_registry:
                        try:
                            count = background_registry.pending_count
                            if count > 0:
                                label = "subagent" if count == 1 else "subagents"
                                state.update_spinner(f"[bold {COLORS['thinking']}]{count} {label} running...")
                        except (AttributeError, TypeError):
                            pass  # Ignore if registry doesn't have pending_count
                    continue

                # Handle UPDATES stream - for todos and interrupts
                if current_stream_mode == "updates":
                    if not isinstance(data, dict):
                        continue

                    # Check for HITL interrupts (plan mode approval)
                    if HITL_AVAILABLE and _HITL_REQUEST_ADAPTER and "__interrupt__" in data:
                        interrupts = data["__interrupt__"]
                        if interrupts:
                            for interrupt_obj in interrupts:
                                try:
                                    validated = _HITL_REQUEST_ADAPTER.validate_python(interrupt_obj.value)
                                    pending_interrupts[interrupt_obj.id] = validated
                                    interrupt_occurred = True
                                except ValidationError as e:
                                    logger.warning(
                                        "Invalid HITL request data",
                                        error=str(e),
                                    )

                    # Extract chunk_data from updates for todo checking
                    chunk_data = next(iter(data.values())) if data else None
                    if chunk_data and isinstance(chunk_data, dict) and "todos" in chunk_data:
                        new_todos = chunk_data["todos"]
                        if new_todos != current_todos:
                            current_todos = new_todos
                            # Stop spinner before rendering todos
                            if state.spinner_active:
                                state.stop_spinner()
                            console.print()
                            render_todo_list(new_todos)
                            console.print()

                # Handle MESSAGES stream - for content and tool calls
                elif current_stream_mode == "messages":
                    # Messages stream returns (message, metadata) tuples
                    if not isinstance(data, tuple) or len(data) != _MESSAGE_TUPLE_SIZE:
                        continue

                    message, _metadata = data

                    # Check message type
                    msg_type = getattr(message, "type", None)

                    if msg_type == "human":
                        raw_content = getattr(message, "content", "")
                        content = format_tool_message_content(raw_content)
                        if content:
                            state.flush_text(final=True)
                            if state.spinner_active:
                                state.stop_spinner()
                            if not state.has_responded:
                                console.print("●", style=COLORS["agent"], markup=False, end=" ")
                                state.has_responded = True
                            markdown = Markdown(content)
                            console.print(markdown, style=COLORS["agent"])
                            console.print()
                        continue

                    if msg_type == "tool":
                        # Tool results - show errors
                        tool_name = getattr(message, "name", "")
                        tool_status = getattr(message, "status", "success")
                        tool_content = format_tool_message_content(getattr(message, "content", ""))

                        # Reset spinner message after tool completes
                        if state.spinner_active:
                            state.update_spinner(f"[bold {COLORS['thinking']}]Agent is thinking...")

                        if tool_name in ("shell", "Bash") and tool_status != "success":
                            state.flush_text(final=True)
                            if tool_content:
                                if state.spinner_active:
                                    state.stop_spinner()
                                console.print()
                                console.print(truncate_error(tool_content), style="red", markup=False)
                                console.print()
                        elif tool_content and isinstance(tool_content, str):
                            stripped = tool_content.lstrip()
                            if stripped.lower().startswith("error"):
                                # Check if this is a sandbox disconnection error
                                if is_sandbox_error(tool_content) and _retry_count == 0 and session:
                                    state.flush_text(final=True)
                                    if state.spinner_active:
                                        state.stop_spinner()
                                    console.print()
                                    console.print("[yellow]⟳ Sandbox disconnected[/yellow]")

                                    if await recover_sandbox(session, console):
                                        console.print()
                                        # Retry the task once
                                        return await execute_task(
                                            user_input,
                                            agent,
                                            assistant_id,
                                            session_state,
                                            token_tracker,
                                            session,
                                            sandbox_completer,
                                            background_registry,
                                            _retry_count=1,
                                        )
                                    return None  # Recovery failed, stop

                                # Regular error - just display it
                                state.flush_text(final=True)
                                if state.spinner_active:
                                    state.stop_spinner()
                                console.print()
                                console.print(truncate_error(tool_content), style="red", markup=False)
                                console.print()
                            elif tool_name in ("task", "wait", "task_output"):
                                # Background task tools return results via ToolMessage; show them so
                                # users aren't left waiting if the agent doesn't echo/summarize.
                                state.flush_text(final=True)
                                if state.spinner_active:
                                    state.stop_spinner()
                                console.print()
                                icon = tool_icons.get(tool_name, "🔧")
                                title = {
                                    "task": "Subagent result",
                                    "wait": "Subagent results",
                                    "task_output": "Task output",
                                }.get(tool_name, f"{tool_name} result")

                                tool_call_id = getattr(message, "tool_call_id", "")
                                if tool_call_id and background_registry:
                                    try:
                                        task = background_registry.get_by_id(tool_call_id)
                                        display_id = getattr(task, "display_id", "")
                                        if display_id:
                                            title = f"{title} ({display_id})"
                                    except Exception:  # noqa: BLE001, S110
                                        pass

                                console.print(
                                    Panel(
                                        Markdown(tool_content),
                                        title=f"{icon} {title}",
                                        border_style=COLORS["tool"],
                                        box=box.ROUNDED,
                                        padding=(0, 1),
                                    )
                                )
                                console.print()
                                state.update_spinner(f"[bold {COLORS['thinking']}]Agent is thinking...")
                                state.start_spinner()

                        # Track consecutive empty results from sensitive tools
                        if (
                            empty_tracker.record(tool_name, tool_content)
                            and _retry_count == 0
                            and session
                            and not await check_sandbox_health(session)
                        ):
                            # Threshold exceeded - check sandbox health
                            state.flush_text(final=True)
                            if state.spinner_active:
                                state.stop_spinner()
                            console.print()
                            console.print("[yellow]⟳ Sandbox disconnected (detected from empty results)[/yellow]")

                            if await recover_sandbox(session, console):
                                console.print()
                                # Retry the task once
                                return await execute_task(
                                    user_input,
                                    agent,
                                    assistant_id,
                                    session_state,
                                    token_tracker,
                                    session,
                                    sandbox_completer,
                                    background_registry,
                                    _retry_count=1,
                                )
                            return None  # Recovery failed, stop
                        continue

                    # Check if this is an AIMessage with content_blocks
                    if not hasattr(message, "content_blocks"):
                        # Fallback - check for content attribute
                        content = getattr(message, "content", "")
                        if content and isinstance(content, str):
                            state.append_text(content)
                        continue

                    # Extract token usage if available
                    if token_tracker and hasattr(message, "usage_metadata"):
                        usage = message.usage_metadata
                        if usage:
                            input_toks = usage.get("input_tokens", 0)
                            output_toks = usage.get("output_tokens", 0)
                            if input_toks or output_toks:
                                captured_input_tokens = max(captured_input_tokens, input_toks)
                                captured_output_tokens = max(captured_output_tokens, output_toks)

                    # Process content blocks
                    for block in message.content_blocks:
                        block_type = block.get("type")

                        # Handle text blocks
                        if block_type == "text":
                            text = block.get("text", "")
                            if text:
                                state.append_text(text)

                        # Handle tool call chunks
                        elif block_type in ("tool_call_chunk", "tool_call"):
                            complete_tool = tool_buffer.add_chunk(block)
                            if complete_tool is None:
                                continue

                            tool_name = complete_tool["name"]
                            tool_id = complete_tool["id"]
                            tool_args = complete_tool["args"]

                            state.flush_text(final=True)
                            if tool_id is not None:
                                if tool_buffer.was_displayed(tool_id):
                                    continue
                                tool_buffer.mark_displayed(tool_id)

                            icon = tool_icons.get(tool_name, "🔧")

                            if state.spinner_active:
                                state.stop_spinner()

                            if state.has_responded:
                                console.print()

                            display_str = format_tool_display(tool_name, tool_args)
                            console.print(
                                f"  {icon} {display_str}",
                                style=f"dim {COLORS['tool']}",
                                markup=False,
                            )

                            # Restart spinner with context about which tool is executing
                            state.update_spinner(f"[bold {COLORS['thinking']}]Executing {tool_name}...")
                            state.start_spinner()

                    if getattr(message, "chunk_position", None) == "last":
                        state.flush_text(final=True)

            # After streaming loop
            state.flush_text(final=True)

            # Handle HITL interrupt (plan mode approval)
            if interrupt_occurred and pending_interrupts:
                if state.spinner_active:
                    state.stop_spinner()

                any_rejected = False

                for interrupt_id, hitl_request in pending_interrupts.items():
                    # Check if auto-approve is enabled
                    if getattr(session_state, "auto_approve", False):
                        # Auto-approve all actions
                        decisions = [{"type": "approve"} for _ in hitl_request.get("action_requests", [])]
                        console.print()
                        console.print("[dim]⚡ Auto-approved plan[/dim]")
                    else:
                        # Prompt user for approval
                        decisions = []
                        esc_watcher.stop()
                        try:
                            for action_request in hitl_request.get("action_requests", []):
                                decision, feedback = await _prompt_for_plan_approval(action_request)

                                if decision.get("type") == "reject":
                                    any_rejected = True
                                    # Put feedback in decision message for HITL to use in ToolMessage
                                    feedback_text = feedback or "No feedback provided"
                                    decision["message"] = (
                                        f"<system-reminder>Your plan was rejected. User feedback: {feedback_text}. "
                                        "You MUST submit the revised plan for review using submit_plan before proceeding.</system-reminder>"
                                    )

                                decisions.append(decision)
                        finally:
                            esc_watcher.start()

                    hitl_response[interrupt_id] = {"decisions": decisions}

                # Update spinner based on decision
                if any_rejected:
                    console.print()
                    console.print("[yellow]Plan rejected. Agent will revise based on your feedback.[/yellow]")
                    state.update_spinner(f"[bold {COLORS['thinking']}]Revising plan...")
                else:
                    state.update_spinner(f"[bold {COLORS['thinking']}]Executing plan...")

                # Resume with decision (no HumanMessage injection needed -
                # approve: tool returns ToolMessage + HumanMessage
                # reject: HITL creates ToolMessage with decision["message"])
                stream_input = Command(resume=hitl_response)
                state.start_spinner()
                # Continue the while loop to resume streaming
            else:
                # No interrupt, break out of while loop
                break

        # After streaming completes, check for background tasks and display live status
        if hasattr(agent, "middleware") and hasattr(agent.middleware, "registry"):
            registry = agent.middleware.registry
            if registry.task_count > 0:
                console.print()
                await _display_live_task_status(registry, console)

    except asyncio.CancelledError:
        # ESC interrupt cancels only the foreground stream.
        # Ctrl+C exits the entire CLI (SIGINT) and should not be swallowed.
        if getattr(session_state, "esc_interrupt_requested", False):
            session_state.esc_interrupt_requested = False
            if state.spinner_active:
                state.stop_spinner()
            console.print("\n[yellow]Interrupted (Esc)[/yellow]")
            return None
        raise

    except KeyboardInterrupt:
        # Ctrl+C exits the CLI during streaming.
        raise


    except Exception as e:
        # Handle API errors gracefully (rate limits, auth failures, connection errors)
        if is_api_error(e):
            if state.spinner_active:
                state.stop_spinner()
            console.print()
            console.print(get_api_error_message(e))
            console.print()
            return None  # Return to CLI loop

        # Check if this is a sandbox-related error we can recover from
        error_msg = str(e)
        if is_sandbox_error(error_msg) and _retry_count == 0 and session:
            if state.spinner_active:
                state.stop_spinner()
            console.print()
            console.print("[yellow]⟳ Sandbox disconnected[/yellow]")

            if await recover_sandbox(session, console):
                console.print()
                # Retry the task once
                return await execute_task(
                    user_input,
                    agent,
                    assistant_id,
                    session_state,
                    token_tracker,
                    session,
                    sandbox_completer,
                    background_registry,
                    _retry_count=1,
                )
            return None  # Recovery failed, stop
        # Re-raise non-sandbox errors
        raise

    finally:
        esc_watcher.stop()

    if state.spinner_active:
        state.stop_spinner()

    if state.has_responded:
        console.print()
        # Track token usage
        if token_tracker and (captured_input_tokens or captured_output_tokens):
            token_tracker.add(captured_input_tokens, captured_output_tokens)

    # Refresh sandbox file cache in background (non-blocking)
    if session and session.sandbox and sandbox_completer:

        async def _refresh_cache() -> None:
            try:
                sandbox = await session.get_sandbox()
                files = await sandbox.aglob_files("**/*", path=".")
                # Normalize paths (remove /home/daytona/ prefix)
                home_prefix = "/home/daytona/"
                normalized = [f.removeprefix(home_prefix) for f in files]
                sandbox_completer.set_files(normalized)
            except Exception:  # noqa: S110, BLE001
                pass  # Silently ignore cache refresh errors

        # Create background task (intentionally not awaited)
        _ = asyncio.create_task(_refresh_cache())  # noqa: RUF006
    return None
