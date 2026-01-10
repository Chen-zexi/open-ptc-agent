"""Tests for task tool result rendering in the streaming executor."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from rich.markdown import Markdown
from rich.panel import Panel

import ptc_cli.streaming.executor as executor_module
from ptc_cli.streaming.executor import execute_task


class FakeStreamingAgent:
    """Minimal agent that yields a predefined stream of events."""

    def __init__(self, events: list[tuple[tuple, str, object]]) -> None:
        self._events = events

    async def astream(self, *_args: object, **_kwargs: object):
        for event in self._events:
            yield event


@pytest.mark.asyncio
async def test_execute_task_displays_task_tool_result_and_hides_subgraph_messages(session_state, monkeypatch):
    """ToolMessage(name='task') is shown even when subgraph messages are skipped."""
    mock_console = Mock()
    mock_status = Mock()
    mock_console.status.return_value = mock_status
    mock_console.print = Mock()

    monkeypatch.setattr(executor_module, "console", mock_console)
    monkeypatch.setattr(
        executor_module,
        "COLORS",
        {"agent": "#10b981", "tool": "#fbbf24", "thinking": "#6b7280"},
    )

    events = [
        (("tools:subgraph",), "messages", (AIMessage(content="SUBGRAPH SHOULD NOT DISPLAY"), {})),
        ((), "messages", (ToolMessage(content="FINAL REPORT", tool_call_id="call_task_1", name="task"), {})),
    ]
    agent = FakeStreamingAgent(events)

    await execute_task(
        "hi",
        agent,
        assistant_id=None,
        session_state=session_state,
    )

    panel_calls = [call for call in mock_console.print.call_args_list if call.args and isinstance(call.args[0], Panel)]
    assert any(isinstance(call.args[0].renderable, Markdown) and "FINAL REPORT" in call.args[0].renderable.markup for call in panel_calls)
    assert not any(
        isinstance(call.args[0].renderable, Markdown) and "SUBGRAPH SHOULD NOT DISPLAY" in call.args[0].renderable.markup for call in panel_calls
    )
