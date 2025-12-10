"""Agent management functions for the CLI."""

import shutil

from ptc_cli.core import console, settings


def get_agent_md_content(agent_name: str) -> str | None:
    """Load user and project agent.md files.

    These files contain custom instructions that are appended to the
    base system prompt from the ptc-agent package.

    NOTE: agent.md is currently NOT ACTIVE because the agent runs in a
    remote sandbox and cannot modify local files.

    Args:
        agent_name: Name of the agent

    Returns:
        Combined agent.md content, or None if no files exist
    """
    parts = []

    # User memory: ~/.ptc-agent/{agent}/agent.md
    user_memory = settings.get_user_agent_md_path(agent_name)
    if user_memory.exists():
        parts.append(f"## User Instructions\n{user_memory.read_text()}")

    # Project memory: .ptc-agent/agent.md
    project_memory = settings.get_project_agent_md_path()
    if project_memory and project_memory.exists():
        parts.append(f"## Project Instructions\n{project_memory.read_text()}")

    return "\n\n".join(parts) if parts else None


def list_agents() -> None:
    """List all available agents."""
    ptc_agent_dir = settings.user_ptc_agent_dir

    if not ptc_agent_dir.exists():
        console.print("[dim]No agents found. Create one by running ptc-agent.[/dim]")
        return

    agents = []
    for path in ptc_agent_dir.iterdir():
        if path.is_dir() and not path.name.startswith("."):
            agent_md = path / "agent.md"
            has_memory = agent_md.exists()
            agents.append((path.name, has_memory))

    if not agents:
        console.print("[dim]No agents found. Create one by running ptc-agent.[/dim]")
        return

    console.print("[bold]Available agents:[/bold]")
    console.print()

    for name, has_memory in sorted(agents):
        memory_indicator = "[green]●[/green]" if has_memory else "[dim]○[/dim]"
        console.print(f"  {memory_indicator} {name}")

    console.print()
    console.print("[dim]● = has agent.md memory file[/dim]")


def reset_agent(agent_name: str, source_agent: str | None = None) -> None:
    """Reset an agent to default or copy from another agent.

    When resetting to default (no source_agent), the agent.md file is deleted
    so the agent uses only the base system prompt from ptc-agent package.

    Args:
        agent_name: Name of agent to reset
        source_agent: Optional agent to copy prompt from
    """
    agent_dir = settings.get_agent_dir(agent_name)
    agent_md = agent_dir / "agent.md"

    if source_agent:
        # Copy from source agent
        source_md = settings.get_user_agent_md_path(source_agent)
        if not source_md.exists():
            console.print(f"[red]Source agent '{source_agent}' not found or has no agent.md[/red]")
            return

        agent_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_md, agent_md)
        console.print(f"[green]Copied agent.md from '{source_agent}' to '{agent_name}'[/green]")
    elif agent_md.exists():
        # Reset to default by removing agent.md
        # The agent will use only the base system prompt from ptc-agent package
        agent_md.unlink()
        console.print(f"[green]Reset '{agent_name}' to default (removed agent.md)[/green]")
    else:
        console.print(f"[dim]Agent '{agent_name}' already using default prompt[/dim]")

    console.print(f"[dim]Agent directory: {agent_dir}[/dim]")
