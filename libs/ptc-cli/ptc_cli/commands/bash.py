"""Bash command execution for the CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ptc_cli.core import console

if TYPE_CHECKING:
    from ptc_agent.core.sandbox import PTCSandbox


async def execute_bash_command(
    command: str,
    sandbox: PTCSandbox | None = None,
) -> None:
    """Execute a bash command in the sandbox.

    Args:
        command: The command string (including the leading !)
        sandbox: The PTCSandbox instance to execute commands in
    """
    # Strip the leading ! and any whitespace
    bash_cmd = command[1:].strip()

    if not bash_cmd:
        console.print("[yellow]No command specified[/yellow]")
        return

    if sandbox is None:
        console.print("[red]Sandbox not initialized. Cannot run bash commands.[/red]")
        return

    console.print(f"[dim]$ {bash_cmd}[/dim]")
    console.print()

    try:
        result = await sandbox.execute_bash_command(
            bash_cmd,
            timeout=60,
        )

        # Print stdout
        if result.get("stdout"):
            console.print(result["stdout"].rstrip())

        # Print stderr in red
        if result.get("stderr"):
            console.print(result["stderr"].rstrip(), style="red")

        # Show return code if non-zero
        exit_code = result.get("exit_code", 0)
        if exit_code != 0:
            console.print(f"[dim]Exit code: {exit_code}[/dim]")

    except TimeoutError:
        console.print("[red]Command timed out (60s limit)[/red]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error executing command: {e}[/red]")

    console.print()
