"""Utility functions for displaying messages and prompts in Jupyter notebooks."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from IPython.display import Image, Markdown, display
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = structlog.get_logger(__name__)

console = Console()


@dataclass
class ExportResult:
    """Result of sandbox export operation."""

    success: bool
    output_directory: Path
    timestamp: str
    files_exported: List[str] = field(default_factory=list)
    files_failed: List[Dict[str, str]] = field(default_factory=list)
    directories_processed: List[str] = field(default_factory=list)
    total_files: int = 0
    total_bytes: int = 0

    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes as human-readable size (KB, MB, GB)."""
        if bytes_val < 1024:
            return f"{bytes_val} B"
        elif bytes_val < 1024 * 1024:
            return f"{bytes_val / 1024:.2f} KB"
        elif bytes_val < 1024 * 1024 * 1024:
            return f"{bytes_val / (1024 * 1024):.2f} MB"
        else:
            return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"

    def summary(self) -> str:
        """Generate human-readable summary with stats and failures."""
        lines = []
        lines.append(f"\nExport Summary ({self.timestamp})")
        lines.append("=" * 60)
        lines.append(f"Output Directory: {self.output_directory}")
        lines.append(f"Total Files: {self.total_files}")
        lines.append(f"Successfully Exported: {len(self.files_exported)}")
        lines.append(f"Failed: {len(self.files_failed)}")
        lines.append(f"Total Size: {self._format_bytes(self.total_bytes)}")

        if self.directories_processed:
            lines.append("\nDirectories Processed:")
            for directory in self.directories_processed:
                lines.append(f"  - {directory}")

        if self.files_failed:
            lines.append("\nFailed Files:")
            for failure in self.files_failed:
                lines.append(f"  - {failure['path']}: {failure['error']}")

        lines.append("")
        return "\n".join(lines)


def format_message_content(message):
    """Convert message content to displayable string."""
    parts = []
    tool_calls_processed = False

    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "tool_use":
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))

    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)


def format_messages(messages):
    """Format and display a list of messages with Rich formatting."""
    for m in messages:
        msg_type = m.__class__.__name__.replace("Message", "")
        content = format_message_content(m)

        if msg_type == "Human":
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type == "Ai":
            console.print(Panel(content, title="🤖 Assistant", border_style="green"))
        elif msg_type == "Tool":
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))


def format_message(messages):
    """Alias for format_messages for backward compatibility."""
    return format_messages(messages)


def dump_messages(
    messages,
    output_path,
    filename: str = "messages.json"
) -> Path:
    """Dump agent messages to a JSON file for debugging.

    Args:
        messages: List of LangChain message objects
        output_path: Directory to save the file (str or Path)
        filename: Name of the output file (default: messages.json)

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename

    serialized_messages = []
    for message in messages:
        # Try model_dump() for Pydantic-based LangChain messages
        try:
            msg_dict = message.model_dump()
            msg_dict["type"] = message.__class__.__name__
            serialized_messages.append(msg_dict)
        except (AttributeError, TypeError):
            # Fallback to manual extraction
            msg_dict = {
                "type": message.__class__.__name__,
                "content": message.content if hasattr(message, "content") else str(message),
            }
            if hasattr(message, "tool_calls") and message.tool_calls:
                msg_dict["tool_calls"] = message.tool_calls
            if hasattr(message, "tool_call_id"):
                msg_dict["tool_call_id"] = message.tool_call_id
            if hasattr(message, "name"):
                msg_dict["name"] = message.name
            if hasattr(message, "additional_kwargs"):
                msg_dict["additional_kwargs"] = message.additional_kwargs
            serialized_messages.append(msg_dict)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serialized_messages, f, indent=2, default=str)

    return file_path


def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "blue"):
    """Display a prompt with rich formatting and XML tag highlighting.

    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    # Create a formatted display of the prompt
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r"<[^>]+>", style="bold blue")  # Highlight XML tags
    formatted_text.highlight_regex(
        r"##[^#\n]+", style="bold magenta"
    )  # Highlight headers
    formatted_text.highlight_regex(
        r"###[^#\n]+", style="bold cyan"
    )  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(
        Panel(
            formatted_text,
            title=f"[bold green]{title}[/bold green]",
            border_style=border_style,
            padding=(1, 2),
        )
    )


def print_agent_config(config, session, ptc_agent=None):
    """Display agent configuration summary.

    Args:
        config: AgentConfig instance
        session: Session instance with mcp_registry
        ptc_agent: PTCAgent instance (optional, for dynamic subagent display)
    """
    print("=" * 70)
    print("AGENT CONFIGURATION SUMMARY")
    print("=" * 70)

    # 1. LLM Configuration
    print("\n📦 LLM CONFIGURATION")
    print("-" * 40)
    print(f"  Name:      {config.llm.name}")
    print(f"  Model ID:  {config.llm_definition.model_id}")
    print(f"  Provider:  {config.llm_definition.provider}")
    if config.llm_definition.parameters:
        print(f"  Parameters: {config.llm_definition.parameters}")

    # 2. MCP Servers and Tools
    print("\n🔌 MCP SERVERS")
    print("-" * 40)
    tools_by_server = session.mcp_registry.get_all_tools()
    total_tools = 0
    for server_name, tools in tools_by_server.items():
        tool_names = [t.name for t in tools]
        total_tools += len(tools)
        print(f"  {server_name}: {len(tools)} tools")
        print(f"    {tool_names}")
    print(f"\n  Total: {len(tools_by_server)} servers, {total_tools} tools")

    # 3. Native Tools
    print("\n🛠️ NATIVE TOOLS")
    print("-" * 40)
    if ptc_agent and hasattr(ptc_agent, "native_tools") and ptc_agent.native_tools:
        print(f"  {', '.join(ptc_agent.native_tools)}")
    else:
        print("  (tools not available - agent not created yet)")

    # 4. Subagents - Dynamic from ptc_agent
    print("\n🤖 SUBAGENTS")
    print("-" * 40)

    if ptc_agent and hasattr(ptc_agent, "subagents") and ptc_agent.subagents:
        for name, info in ptc_agent.subagents.items():
            tools_str = ", ".join(info.get("tools", []))
            print(f"  {name}: {tools_str}")
    else:
        print("  (no subagents configured or agent not created yet)")

    print("\n" + "=" * 70)


def print_sandbox_tree(sandbox, path=".", indent=""):
    """Print directory tree of sandbox.

    Args:
        sandbox: PTCSandbox instance
        path: Starting path in sandbox (default: ".")
        indent: Current indentation string (used recursively)
    """
    entries = sandbox.list_directory(path)
    dirs = [e for e in entries if e["type"] == "directory"]
    files = [e for e in entries if e["type"] == "file"]

    for i, entry in enumerate(dirs + files):
        is_last = i == len(dirs) + len(files) - 1
        connector = "└── " if is_last else "├── "
        suffix = "/" if entry["type"] == "directory" else ""
        print(f"{indent}{connector}{entry['name']}{suffix}")

        if entry["type"] == "directory":
            new_indent = indent + ("    " if is_last else "│   ")
            print_sandbox_tree(sandbox, entry["path"], new_indent)


def display_sandbox_image(sandbox, filepath):
    """Display a single image from the sandbox.

    Args:
        sandbox: PTCSandbox instance
        filepath: Path to image file in sandbox (e.g., "results/chart.png")
    """
    image_bytes = sandbox.download_file_bytes(filepath)
    if image_bytes:
        print(f"📊 {filepath}")
        display(Image(data=image_bytes))
    else:
        print(f"❌ Could not load image: {filepath}")


def display_sandbox_images(sandbox, directory="results"):
    """Display all images from a sandbox directory.

    Args:
        sandbox: PTCSandbox instance
        directory: Directory path in sandbox (default: "results")
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'}

    try:
        entries = sandbox.list_directory(directory)
    except Exception as e:
        print(f"❌ Could not list directory {directory}: {e}")
        return

    image_count = 0
    for entry in entries:
        if entry["type"] == "file":
            ext = Path(entry["name"]).suffix.lower()
            if ext in image_extensions:
                filepath = entry["path"]
                image_bytes = sandbox.download_file_bytes(filepath)
                if image_bytes:
                    print(f"📊 {filepath}")
                    display(Image(data=image_bytes))
                    image_count += 1

    if image_count == 0:
        print(f"No images found in {directory}/")
    else:
        print(f"\n✅ Displayed {image_count} image(s)")


def display_result(sandbox, filepath: str = "results/result.md"):
    """Display a markdown file from sandbox, rendered in notebook.

    Args:
        sandbox: PTCSandbox instance
        filepath: Path to markdown file in sandbox (default: results/result.md)
    """
    content = sandbox.read_file(filepath)
    if content:
        display(Markdown(content))
    else:
        print(f"Could not load markdown file: {filepath}")


def export_sandbox_files(
    sandbox,
    output_base: str = "output",
    directories: Optional[List[str]] = None,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> ExportResult:
    """Export files from Daytona sandbox to local filesystem.

    Downloads files from specified sandbox directories with partial download
    support - continues downloading available files even if some fail.

    Args:
        sandbox: PTCSandbox instance to export from
        output_base: Base directory for exports (default: "output")
        directories: List of directories to export (default: ["code", "data", "results"])
        timestamp_format: strftime format for timestamp directory (default: sortable)

    Returns:
        ExportResult with detailed successes, failures, and statistics

    Examples:
        >>> # Export default directories (code, data, results)
        >>> result = export_sandbox_files(sandbox)
        >>> print(result.summary())

        >>> # Export only results directory
        >>> result = export_sandbox_files(sandbox, directories=['results'])
        >>> if result.files_failed:
        >>>     for failure in result.files_failed:
        >>>         print(f"{failure['path']}: {failure['error']}")

        >>> # Custom location and timestamp
        >>> result = export_sandbox_files(
        >>>     sandbox,
        >>>     output_base="my_exports",
        >>>     timestamp_format="%Y-%m-%d_%H-%M"
        >>> )
    """
    # Phase 1: Initialization
    if not hasattr(sandbox, "sandbox") or sandbox.sandbox is None:
        raise ValueError("Sandbox not initialized. Call sandbox.setup() first.")

    # Set default directories if not specified
    if directories is None:
        directories = ["code", "data", "results"]

    # Generate timestamp and create output directory
    timestamp = datetime.now().strftime(timestamp_format)
    output_dir = Path(output_base) / timestamp

    # Validate output base is not a file
    if Path(output_base).exists() and Path(output_base).is_file():
        raise ValueError(f"Output base path '{output_base}' is a file, not a directory")

    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Cannot create output directory '{output_dir}': {e}") from e

    # Initialize result tracking
    result = ExportResult(
        success=False,
        output_directory=output_dir.resolve(),
        timestamp=timestamp,
    )

    # Helper functions
    def normalize_sandbox_path(directory: str) -> str:
        """Convert relative directory to absolute sandbox path."""
        if directory.startswith("/"):
            return directory
        return f"/home/daytona/{directory}"

    def get_relative_local_path(sandbox_path: str) -> str:
        """Strip /home/daytona/ prefix for local path construction."""
        prefix = "/home/daytona/"
        if sandbox_path.startswith(prefix):
            return sandbox_path[len(prefix) :]
        return sandbox_path

    def discover_files_recursive(sandbox_path: str) -> List[Dict[str, str]]:
        """Recursively discover all files in directory tree."""
        all_files = []

        try:
            entries = sandbox.list_directory(sandbox_path)
        except Exception as e:
            result.files_failed.append(
                {"path": sandbox_path, "error": f"Cannot list directory: {str(e)}"}
            )
            return all_files

        for entry in entries:
            if entry["type"] == "file":
                all_files.append(
                    {
                        "sandbox_path": entry["path"],
                        "relative_path": get_relative_local_path(entry["path"]),
                    }
                )
            elif entry["type"] == "directory":
                # Recurse into subdirectory
                sub_files = discover_files_recursive(entry["path"])
                all_files.extend(sub_files)

        return all_files

    # Phase 2: Process each directory
    for directory in directories:
        normalized_path = normalize_sandbox_path(directory)

        try:
            # Try to list the directory to verify it exists
            entries = sandbox.list_directory(normalized_path)
            result.directories_processed.append(directory)

            # If empty, continue to next
            if not entries:
                print(f"ℹ️  Directory '{directory}' is empty")
                continue

        except Exception as e:
            print(f"⚠️  Could not access directory '{directory}': {e}")
            result.files_failed.append(
                {"path": normalized_path, "error": f"Cannot access directory: {str(e)}"}
            )
            continue

        # Phase 3: Discover files recursively
        print(f"📂 Discovering files in '{directory}'...")
        files_to_download = discover_files_recursive(normalized_path)
        print(f"   Found {len(files_to_download)} file(s)")

        # Phase 4: Download files with error recovery
        for file_info in files_to_download:
            sandbox_path = file_info["sandbox_path"]
            relative_path = file_info["relative_path"]
            local_path = output_dir / relative_path

            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Primary method: download_file_bytes (works for all file types)
                content = sandbox.download_file_bytes(sandbox_path)

                if content is None:
                    # Fallback: try read_file for text files
                    content_text = sandbox.read_file(sandbox_path)
                    if content_text:
                        content = content_text.encode("utf-8")
                    else:
                        raise ValueError("Both download methods returned None")

                # Write to local filesystem
                local_path.write_bytes(content)

                # Track success
                result.files_exported.append(sandbox_path)
                result.total_bytes += len(content)
                result.total_files += 1

            except PermissionError as e:
                result.files_failed.append(
                    {"path": sandbox_path, "error": f"Permission denied: {str(e)}"}
                )
                print(f"   ❌ Permission denied: {relative_path}")

            except OSError as e:
                result.files_failed.append(
                    {"path": sandbox_path, "error": f"OS error: {str(e)}"}
                )
                print(f"   ❌ OS error: {relative_path}")

            except Exception as e:
                result.files_failed.append(
                    {
                        "path": sandbox_path,
                        "error": f"{type(e).__name__}: {str(e)}",
                    }
                )
                print(f"   ❌ Failed: {relative_path}")

    # Phase 5: Finalization
    result.success = result.total_files > 0

    if result.success:
        print(f"\n✅ Export completed successfully!")
        print(f"   Exported {result.total_files} file(s) to: {output_dir}")
    else:
        print(f"\n⚠️  Export completed with no files exported")

    return result
