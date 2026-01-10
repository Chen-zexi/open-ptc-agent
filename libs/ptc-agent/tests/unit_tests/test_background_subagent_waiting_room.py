"""Tests for background subagent waiting room behavior."""

import asyncio

from ptc_agent.agent.middleware.background.middleware import BackgroundSubagentMiddleware


async def test_aafter_agent_collects_results_for_already_done_tasks():
    """Completed asyncio tasks should still be collected even if pending_count is 0."""
    middleware = BackgroundSubagentMiddleware(timeout=1.0, enabled=True)

    async def _done_result():
        return {"success": True, "result": "ok"}

    asyncio_task = asyncio.create_task(_done_result())
    await asyncio_task  # Ensure done before aafter_agent runs

    await middleware.registry.register(
        task_id="tool_call_id_1",
        description="test task",
        subagent_type="general-purpose",
        asyncio_task=asyncio_task,
    )

    assert middleware.registry.pending_count == 0  # done() tasks aren't "pending"

    update = await middleware.aafter_agent(state={}, runtime={})

    assert update is not None
    assert update.get("_has_pending_results") is True
    assert middleware.get_pending_results().get("tool_call_id_1") == {"success": True, "result": "ok"}
