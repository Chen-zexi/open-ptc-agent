"""Command handlers for the CLI."""

from ptc_cli.commands.bash import execute_bash_command
from ptc_cli.commands.slash import handle_command

__all__ = ["execute_bash_command", "handle_command"]
