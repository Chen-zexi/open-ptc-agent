"""Background subagent orchestrator.

This module provides an orchestrator that wraps the agent and handles
re-invocation when background subagent results are collected.
"""

from collections.abc import AsyncIterator
from typing import Any

import structlog
from langchain_core.messages import HumanMessage

from ptc_agent.agent.middleware.background.middleware import BackgroundSubagentMiddleware
from ptc_agent.agent.middleware.background.tools import extract_result_content

logger = structlog.get_logger(__name__)


class BackgroundSubagentOrchestrator:
    """Orchestrator that handles re-invocation after background results.

    Since the middleware's after_agent hook cannot return a Command to
    re-enter the agent loop, this orchestrator wraps the agent invocation
    and implements the multi-invoke pattern:

    1. First invocation: Agent runs normally, spawning background tasks
    2. After agent ends: Middleware collects background results
    3. If results pending: Orchestrator re-invokes agent with results
    4. Second invocation: Agent synthesizes results into final response

    Usage:
        middleware = BackgroundSubagentMiddleware(timeout=60.0)
        agent = create_deep_agent(
            model=...,
            tools=...,
            middleware=[middleware],
        )
        orchestrator = BackgroundSubagentOrchestrator(agent, middleware)

        # Use orchestrator instead of agent directly
        result = await orchestrator.ainvoke(input_state)
    """

    def __init__(
        self,
        agent: Any,
        middleware: BackgroundSubagentMiddleware,
        max_iterations: int = 3,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            agent: The deepagent instance to wrap
            middleware: The BackgroundSubagentMiddleware instance
            max_iterations: Maximum number of re-invocation iterations
        """
        self.agent = agent
        self.middleware = middleware
        self.max_iterations = max_iterations

    async def ainvoke(
        self,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke agent with automatic re-invocation for background results.

        This method:
        1. Invokes the agent with the input state
        2. Checks if background results are pending
        3. If pending, re-invokes the agent with collected results
        4. Returns the final result

        Args:
            input_state: Initial state for the agent
            config: Optional config dict for the agent

        Returns:
            Final agent result
        """
        config = config or {}
        iteration = 0
        current_state = input_state

        while iteration < self.max_iterations:
            iteration += 1

            logger.info(
                "Orchestrator invoking agent",
                iteration=iteration,
                has_messages="messages" in current_state,
            )

            # Invoke the agent
            result = await self.agent.ainvoke(current_state, config)

            # Check for pending background results
            pending_results = self.middleware.get_pending_results()

            if not pending_results:
                logger.debug(
                    "No pending background results, returning",
                    iteration=iteration,
                )
                return result

            logger.info(
                "Background results pending, preparing re-invocation",
                result_count=len(pending_results),
                iteration=iteration,
            )

            # Format results for injection
            results_summary = self._format_results(pending_results)

            # Get messages from result
            messages = result.get("messages", [])

            # Inject results as new human message
            synthesis_message = HumanMessage(
                content=(
                    f"Your background subagent tasks have completed. "
                    f"Here are the results:\n\n{results_summary}\n\n"
                    f"Please synthesize these results into your final response "
                    f"to the user's original request."
                )
            )

            # Create new state for synthesis
            current_state = {"messages": [*messages, synthesis_message]}

            # Clear middleware results for next iteration
            self.middleware.clear_results()

        logger.warning(
            "Orchestrator reached max iterations",
            max_iterations=self.max_iterations,
        )
        return result

    def invoke(
        self,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronous invoke - no background task support.

        For sync execution, background tasks are not supported.
        Falls back to direct agent invocation.

        Args:
            input_state: Initial state for the agent
            config: Optional config dict for the agent

        Returns:
            Agent result
        """
        logger.warning(
            "Sync invoke called - background tasks not supported in sync mode"
        )
        return self.agent.invoke(input_state, config or {})

    async def astream(
        self,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        stream_mode: str | list[str] | None = None,
        subgraphs: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream agent responses with background result handling.

        Streams the agent's responses, handling background results between
        invocations. Fully compatible with LangGraph's streaming API.

        Args:
            input_state: Initial state for the agent
            config: Optional config dict for the agent
            stream_mode: Stream mode(s) - "values", "updates", "messages", or list
            subgraphs: Whether to include subgraph events
            **kwargs: Additional arguments passed to underlying agent.astream()

        Yields:
            Agent events/updates (format depends on stream_mode)
        """
        config = config or {}
        iteration = 0
        current_state = input_state

        # Build kwargs for underlying astream call
        stream_kwargs: dict[str, Any] = {**kwargs}
        if stream_mode is not None:
            stream_kwargs["stream_mode"] = stream_mode
        if subgraphs:
            stream_kwargs["subgraphs"] = subgraphs

        while iteration < self.max_iterations:
            iteration += 1

            logger.info(
                "Orchestrator streaming agent",
                iteration=iteration,
                stream_mode=stream_mode,
                subgraphs=subgraphs,
            )

            # Stream the agent with all parameters
            async for event in self.agent.astream(current_state, config, **stream_kwargs):
                yield event

            # After streaming completes, check for pending results
            pending_results = self.middleware.get_pending_results()

            if not pending_results:
                return

            logger.info(
                "Background results pending after stream, re-invoking",
                result_count=len(pending_results),
            )

            # Get final state from agent (need to invoke to get final state)
            result = await self.agent.ainvoke(current_state, config)
            messages = result.get("messages", [])

            # Format and inject results
            results_summary = self._format_results(pending_results)
            synthesis_message = HumanMessage(
                content=(
                    f"Your background subagent tasks have completed. "
                    f"Here are the results:\n\n{results_summary}\n\n"
                    f"Please synthesize these results into your final response."
                )
            )

            current_state = {"messages": [*messages, synthesis_message]}
            self.middleware.clear_results()

    def _format_results(self, results: dict[str, Any]) -> str:
        """Format background results for injection into the agent.

        Args:
            results: Dict mapping task_id to result

        Returns:
            Formatted string of results
        """
        lines = []

        for task_id, result in results.items():
            # Look up task to get display_id (Task-N format)
            task = self.middleware.registry.get_by_id(task_id)
            display_name = task.display_id if task else f"Task: {task_id}"
            lines.append(f"### {display_name}")

            success, content = extract_result_content(result)
            if success:
                lines.append("Status: Success")
                lines.append(f"Result:\n{content}")
            else:
                lines.append("Status: error")
                lines.append(f"Error: {content}")

            lines.append("")

        return "\n".join(lines)

    def with_config(self, config: dict[str, Any]) -> "BackgroundSubagentOrchestrator":
        """Return orchestrator with config applied to underlying agent.

        Args:
            config: Config to apply

        Returns:
            New orchestrator with configured agent
        """
        configured_agent = self.agent.with_config(config)
        return BackgroundSubagentOrchestrator(
            agent=configured_agent,
            middleware=self.middleware,
            max_iterations=self.max_iterations,
        )

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying agent.

        This allows the orchestrator to be used as a drop-in replacement
        for the agent in most cases.
        """
        return getattr(self.agent, name)
