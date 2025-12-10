"""Agent lifecycle functions for the CLI."""

from collections.abc import Callable

from ptc_cli.agent.management import get_agent_md_content
from ptc_cli.agent.persistence import (
    delete_persisted_session,
    get_session_config_hash,
    load_persisted_session,
    save_persisted_session,
    update_session_last_used,
)


async def create_agent_with_session(
    agent_name: str,
    sandbox_id: str | None = None,
    *,
    persist_session: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> tuple:
    """Create agent with session, sandbox, and MCP registry.

    Args:
        agent_name: Agent identifier for memory storage
        sandbox_id: Optional existing sandbox ID to reuse (overrides persistence)
        persist_session: Whether to persist/reuse sandbox sessions
        on_progress: Optional callback for progress updates

    Returns:
        Tuple of (agent, session, reusing_sandbox, ptc_agent, config)
    """
    def report(step: str) -> None:
        if on_progress:
            on_progress(step)

    # Import PTC Agent modules
    from ptc_agent.agent.agent import PTCAgent
    from ptc_agent.config import ConfigContext, load_from_files
    from ptc_agent.core.session import SessionManager

    report("Loading configuration...")

    # Load config with CLI context (searches ~/.ptc-agent/ first, auto-generates if missing)
    config = await load_from_files(context=ConfigContext.CLI, auto_generate=True)
    config.validate_api_keys()

    # Calculate config hash for invalidation detection
    config_hash = get_session_config_hash(config)

    # Determine sandbox_id to use
    reusing_sandbox = False
    persisted_sandbox_id = None

    if sandbox_id:
        # Explicit sandbox_id provided via CLI - use it directly
        persisted_sandbox_id = sandbox_id
    elif persist_session:
        # Check for persisted session
        persisted = load_persisted_session(agent_name)
        if persisted:
            if persisted["config_hash"] == config_hash:
                # Config matches, try to reuse sandbox
                persisted_sandbox_id = persisted["sandbox_id"]
            else:
                # Config changed, invalidate session
                delete_persisted_session(agent_name)

    report("Creating session...")

    # Create session (manages sandbox + MCP)
    session = SessionManager.get_session(agent_name, config.to_core_config())

    if persisted_sandbox_id:
        report("Reconnecting to sandbox...")
        try:
            await session.initialize(sandbox_id=persisted_sandbox_id)
            reusing_sandbox = True
            # Update last_used timestamp
            if persist_session:
                update_session_last_used(agent_name)
        except Exception:  # noqa: BLE001
            # Reconnection failed, create new sandbox
            report("Creating new sandbox...")
            delete_persisted_session(agent_name)
            # Clear the failed session and create fresh one
            SessionManager._sessions.pop(agent_name, None)
            session = SessionManager.get_session(agent_name, config.to_core_config())
            await session.initialize()
            reusing_sandbox = False
    else:
        # No existing sandbox, create new one
        report("Creating sandbox...")
        await session.initialize()

    if persist_session and not reusing_sandbox and session.sandbox:
        sandbox_id_to_save = getattr(session.sandbox, "sandbox_id", None)
        if sandbox_id_to_save:
            save_persisted_session(agent_name, sandbox_id_to_save, config_hash)

    report("Creating agent...")

    # Create checkpointer for HITL interrupt/resume (submit_plan tool)
    checkpointer = None
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    except ImportError:
        pass

    # Load agent.md content
    # NOTE: Currently not active in practice
    agent_md_content = get_agent_md_content(agent_name)

    # Create agent using existing PTCAgent
    ptc_agent = PTCAgent(config)
    assert session.sandbox is not None, "Session sandbox must be initialized"  # noqa: S101
    assert session.mcp_registry is not None, "Session MCP registry must be initialized"  # noqa: S101
    agent = ptc_agent.create_agent(
        sandbox=session.sandbox,
        mcp_registry=session.mcp_registry,
        subagent_names=config.subagents_enabled,
        checkpointer=checkpointer,
        system_prompt_suffix=agent_md_content,
    )

    # Store checkpointer reference on agent for CLI access
    if checkpointer:
        agent.checkpointer = checkpointer

    return agent, session, reusing_sandbox, ptc_agent, config
