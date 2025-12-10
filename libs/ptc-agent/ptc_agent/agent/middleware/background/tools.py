"""Background subagent management tools.

This module provides tools for the main agent to interact with background
subagents: waiting for results and checking progress.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.tools import StructuredTool

if TYPE_CHECKING:
    from ptc_agent.agent.middleware.background.middleware import BackgroundSubagentMiddleware
    from ptc_agent.agent.middleware.background.registry import BackgroundTask, BackgroundTaskRegistry

logger = structlog.get_logger(__name__)


def create_wait_tool(middleware: BackgroundSubagentMiddleware) -> StructuredTool:
    """Create the wait tool for entering the waiting room.

    This tool allows the main agent to explicitly wait for background
    subagent(s) to complete and retrieve their results.

    Args:
        middleware: The BackgroundSubagentMiddleware instance

    Returns:
        A StructuredTool for waiting
    """

    async def wait_for_subagents(
        task_number: int | None = None,
        timeout: float = 60.0,
    ) -> str:
        """Wait for background task(s) to complete.

        Args:
            task_number: Task number (1, 2, ...) or None for all
            timeout: Max seconds (default: 60)

        Returns:
            Task result(s)
        """
        registry = middleware.registry

        if task_number is not None:
            # Wait for specific task
            logger.info(
                "Waiting for specific task",
                task_number=task_number,
                timeout=timeout,
            )
            result = await registry.wait_for_specific(task_number, timeout)
            task = await registry.get_by_number(task_number)

            if task:
                return (
                    f"**{task.display_id}** ({task.subagent_type}) completed:\n\n"
                    f"{_format_result(result)}"
                )
            return f"Task-{task_number} not found"
        # Wait for all tasks
        logger.info("Waiting for all background tasks", timeout=timeout)
        results = await registry.wait_for_all(timeout=timeout)
        # Don't store in _pending_results - results are returned directly
        # to the agent via the tool response. Storing them would cause
        # the orchestrator to inject a duplicate HumanMessage later.

        if not results:
            return "No background tasks were pending."

        output = f"All {len(results)} background task(s) completed:\n\n"
        for task_id, result in results.items():
            task = registry.get_by_id(task_id)
            if task:
                output += f"### {task.display_id} ({task.subagent_type})\n"
                output += _format_result(result) + "\n\n"
        return output

    return StructuredTool.from_function(
        name="wait",
        description=(
            "Wait for background subagent(s) to complete and retrieve their results. "
            "Use wait(task_number=1) for a specific task or wait() for all pending tasks. "
            "You can also specify a custom timeout in seconds."
        ),
        coroutine=wait_for_subagents,
    )


def create_task_progress_tool(registry: BackgroundTaskRegistry) -> StructuredTool:
    """Create tool to check background task progress.

    This tool allows the main agent to monitor the status and progress
    of background subagents without waiting for them to complete.

    Args:
        registry: The BackgroundTaskRegistry instance

    Returns:
        A StructuredTool for checking progress
    """

    async def task_progress(task_number: int | None = None) -> str:
        """Check background task progress.

        Args:
            task_number: Task number or None for all

        Returns:
            Status, tool calls, elapsed time
        """
        if task_number is not None:
            task = await registry.get_by_number(task_number)
            if not task:
                return f"Task-{task_number} not found"
            return _format_task_progress(task)

        # Show all tasks
        all_tasks = await registry.get_all_tasks()
        if not all_tasks:
            return "No background tasks have been assigned yet."

        pending_count = sum(1 for t in all_tasks if not t.completed)
        completed_count = len(all_tasks) - pending_count

        output = (
            f"**Background Tasks** ({len(all_tasks)} total: "
            f"{completed_count} completed, {pending_count} running)\n\n"
        )

        for task in sorted(all_tasks, key=lambda t: t.task_number):
            output += _format_task_progress(task) + "\n"

        return output

    return StructuredTool.from_function(
        name="task_progress",
        description=(
            "Check the progress of background subagent tasks. Shows status, "
            "tool calls made, current activity, and elapsed time. "
            "Use task_progress(task_number=1) for a specific task or "
            "task_progress() to see all tasks."
        ),
        coroutine=task_progress,
    )


def extract_result_content(result: dict[str, Any] | Any) -> tuple[bool, str]:
    """Extract content from a task result.

    Handles various result types including raw values, dicts with success/error,
    objects with .content attribute, and Command types with .update.messages.

    Args:
        result: The task result (dict, Command, or raw value)

    Returns:
        Tuple of (success: bool, content: str)
    """
    if not isinstance(result, dict):
        return (True, str(result))

    if result.get("success"):
        inner = result.get("result")
        if inner is None:
            return (True, "Task completed successfully (no output)")
        if hasattr(inner, "content"):
            return (True, str(inner.content))
        # Handle Command type
        if hasattr(inner, "update"):
            update = inner.update
            if isinstance(update, dict) and "messages" in update:
                messages = update["messages"]
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        return (True, str(last_msg.content))
        return (True, str(inner))

    error = result.get("error", "Unknown error")
    status = result.get("status", "error")
    return (False, f"{status.upper()}: {error}")


def _format_result(result: dict[str, Any] | Any) -> str:
    """Format a single task result for display.

    Args:
        result: The task result dict

    Returns:
        Formatted string
    """
    success, content = extract_result_content(result)
    if success:
        return content
    return f"**{content}**"


def _format_task_progress(task: BackgroundTask) -> str:
    """Format progress info for a single task.

    Args:
        task: The BackgroundTask to format

    Returns:
        Formatted progress string
    """
    elapsed = time.time() - task.created_at

    # Status indicator
    status = ("[ERROR]" if task.error else "[DONE]") if task.completed else "[RUNNING]"

    # Tool call summary (always show, even if 0)
    tool_summary = f" | {task.total_tool_calls} tool calls"
    if task.tool_call_counts:
        # Show top 3 tools
        top_tools = sorted(
            task.tool_call_counts.items(),
            key=lambda x: -x[1]
        )[:3]
        tool_details = ", ".join(f"{t}: {c}" for t, c in top_tools)
        tool_summary += f" ({tool_details})"

    # Current activity (only for running tasks)
    activity = ""
    if not task.completed and task.current_tool:
        activity = f"\n  Currently executing: `{task.current_tool}`"

    return (
        f"### {task.display_id}: {task.subagent_type}\n"
        f"  Status: {status} | Elapsed: {elapsed:.1f}s{tool_summary}{activity}\n"
        f"  Task: {task.description[:100]}{'...' if len(task.description) > 100 else ''}"
    )
