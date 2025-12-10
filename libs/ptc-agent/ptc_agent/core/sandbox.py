"""PTC Sandbox - Manages Daytona sandbox for Programmatic Tool Calling execution."""

import asyncio
import base64
import hashlib
import json
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import Any

import aiofiles
import structlog
from daytona_sdk import Daytona, DaytonaConfig
from daytona_sdk.common.daytona import (
    CreateSandboxFromSnapshotParams,
    Image,
)
from daytona_sdk.common.snapshot import CreateSnapshotParams

from ptc_agent.config.core import CoreConfig

from .mcp_registry import MCPRegistry
from .tool_generator import ToolFunctionGenerator

logger = structlog.get_logger(__name__)


@dataclass
class ChartData:
    """Captured chart from matplotlib execution."""

    type: str
    title: str
    png_base64: str | None = None
    elements: list[Any] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""

    success: bool
    stdout: str
    stderr: str
    duration: float
    files_created: list[str]
    files_modified: list[str]
    execution_id: str
    code_hash: str
    charts: list[ChartData] = field(default_factory=list)


class PTCSandbox:
    """Manages Daytona sandbox for Programmatic Tool Calling (PTC) execution."""

    # Default Python dependencies installed in sandbox
    DEFAULT_DEPENDENCIES = [
        # Core
        "mcp", "fastmcp", "pandas", "requests", "aiohttp", "httpx",
        # Data science
        "numpy", "scipy", "scikit-learn", "statsmodels",
        # Financial data
        "yfinance",
        # Visualization
        "matplotlib", "seaborn", "plotly", "mplfinance==0.12.10b0",
        # Image analysis
        "pillow", "opencv-python-headless", "scikit-image",
        # File formats
        "openpyxl", "xlrd", "python-docx", "pypdf",
        "beautifulsoup4", "lxml", "pyyaml",
        # Utilities
        "tqdm", "tabulate",
    ]

    def __init__(self, config: CoreConfig, mcp_registry: MCPRegistry | None = None) -> None:
        """Initialize PTC sandbox.

        Args:
            config: Configuration object
            mcp_registry: MCP registry with connected servers (can be None for reconnect)
        """
        self.config = config
        self.mcp_registry = mcp_registry

        # Initialize Daytona with proper config
        daytona_config = DaytonaConfig(
            api_key=config.daytona.api_key,
            api_url=config.daytona.base_url
        )
        self.daytona_client = Daytona(daytona_config)
        # External Daytona SDK sandbox object - Any type is required since it's from external SDK
        self.sandbox: Any | None = None
        self.sandbox_id: str | None = None
        self.tool_generator = ToolFunctionGenerator()
        self.execution_count = 0
        self.bash_execution_count = 0

        logger.info("Initialized PTCSandbox")

    async def _run_sync(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous function in a thread pool to avoid blocking.

        This wrapper is used for Daytona SDK calls which are synchronous.

        Args:
            func: The synchronous function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call
        """
        loop = asyncio.get_event_loop()
        if kwargs:
            func_with_kwargs = partial(func, *args, **kwargs)
            return await loop.run_in_executor(None, func_with_kwargs)
        if args:
            return await loop.run_in_executor(None, func, *args)
        return await loop.run_in_executor(None, func)

    def _get_mcp_packages(self) -> list[str]:
        """Extract MCP package names from enabled stdio servers.

        Returns:
            List of MCP package names to install globally
        """
        mcp_packages = []
        for server in self.config.mcp.servers:
            if not server.enabled:
                continue
            if server.transport == "stdio" and server.command == "npx":
                # Extract package name from npx arguments
                # Format: ["npx", "-y", "package-name", ...]
                if len(server.args) >= 2 and server.args[0] == "-y":
                    mcp_packages.append(server.args[1])
        return mcp_packages

    def _normalize_search_path(self, path: str) -> str:
        """Normalize search path to absolute sandbox path.

        Converts relative/virtual paths to absolute paths for search operations.

        Args:
            path: Path to normalize (".", relative, or absolute)

        Returns:
            Absolute sandbox path
        """
        if path == ".":
            return self.config.filesystem.working_directory
        if not path.startswith("/"):
            return f"{self.config.filesystem.working_directory}/{path}"
        return path

    def _create_snapshot_image(self) -> Image:
        """Create image definition for snapshot with Node.js and MCP servers.

        Returns:
            Image definition with base dependencies and configuration
        """
        # Use class-level default dependencies
        dependencies = self.DEFAULT_DEPENDENCIES

        # Get MCP server npm packages from config (only enabled servers)
        mcp_packages = self._get_mcp_packages()

        # Build image using declarative builder
        # Note: Directories are created in _setup_workspace(), not in snapshot
        image = (
            Image.debian_slim(self.config.daytona.python_version)
            .run_commands(
                # Install system dependencies including ripgrep for fast search
                "apt-get update",
                "apt-get install -y curl ripgrep jq git unzip",
                # Install uv for fast Python package management
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                "mv /root/.local/bin/uv /usr/local/bin/uv",
                # Install Node.js 20.x LTS
                "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
                "apt-get install -y nodejs",
                # Install MCP server packages globally
                *[f"npm install -g {pkg}" for pkg in mcp_packages],
                # Clean up apt cache to reduce image size
                "apt-get clean",
                "rm -rf /var/lib/apt/lists/*"
            )
            .pip_install(*dependencies)  # Unpack list as individual arguments
            .workdir("/home/daytona")
        )

        logger.info(
            "Created snapshot image definition",
            python_version=self.config.daytona.python_version,
            dependencies=dependencies,
            mcp_packages=mcp_packages,
        )

        return image

    def _get_snapshot_hash(self) -> str:
        """Generate hash for snapshot versioning based on configuration.

        Returns:
            8-character hash of snapshot configuration
        """
        # Get MCP server npm packages from config (only enabled servers)
        mcp_packages = self._get_mcp_packages()

        # Include configuration that affects the snapshot in the hash
        config_data = {
            "python_version": self.config.daytona.python_version,
            "dependencies": self.DEFAULT_DEPENDENCIES,
            "mcp_packages": sorted(mcp_packages),  # Include MCP packages in hash
            "apt_packages": ["curl", "nodejs", "ripgrep", "uv", "jq", "git", "unzip"],  # Include apt/curl-installed packages in hash
        }

        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]

    async def _ensure_snapshot(self) -> str | None:
        """Ensure snapshot exists, create if needed.

        Returns:
            Snapshot name if available, None otherwise
        """
        if not self.config.daytona.snapshot_enabled:
            logger.debug("Snapshot feature disabled in config")
            return None

        # Generate versioned snapshot name with config hash
        config_hash = self._get_snapshot_hash()
        base_name = self.config.daytona.snapshot_name or "ptc-base"
        snapshot_name = f"{base_name}-{config_hash}"

        logger.info("Checking for snapshot", snapshot_name=snapshot_name)

        # Check if snapshot exists and is usable
        try:
            snapshots_result = await self._run_sync(self.daytona_client.snapshot.list)
            snapshots = snapshots_result.items if hasattr(snapshots_result, "items") else snapshots_result

            # Only consider active or building snapshots as existing
            # Failed snapshots should be recreated
            snapshot_obj = None
            for s in snapshots:
                if hasattr(s, "name") and s.name == snapshot_name:
                    snapshot_obj = s
                    break

            if snapshot_obj:
                state = snapshot_obj.state.value if hasattr(snapshot_obj.state, "value") else str(snapshot_obj.state)
                if state == "build_failed":
                    logger.warning(
                        "Found failed snapshot, will recreate",
                        snapshot_name=snapshot_name,
                        error=snapshot_obj.error_reason
                    )
                    # Delete failed snapshot
                    try:
                        await self._run_sync(self.daytona_client.snapshot.delete, snapshot_obj)
                        logger.info("Deleted failed snapshot", snapshot_name=snapshot_name)
                        # Give the deletion a moment to complete
                        await asyncio.sleep(2)
                    except OSError as del_err:
                        logger.warning("Could not delete failed snapshot", error=str(del_err))
                    snapshot_exists = False
                elif state in ["active", "building"]:
                    snapshot_exists = True
                else:
                    logger.warning(f"Snapshot in unexpected state: {state}")
                    snapshot_exists = False
            else:
                snapshot_exists = False

        except OSError as e:
            logger.warning("Error listing snapshots", error=str(e))
            snapshot_exists = False

        # Create snapshot if it doesn't exist
        if not snapshot_exists and self.config.daytona.snapshot_auto_create:
            logger.info("Creating snapshot", snapshot_name=snapshot_name)
            image = self._create_snapshot_image()

            try:
                await self._run_sync(
                    self.daytona_client.snapshot.create,
                    CreateSnapshotParams(
                        name=snapshot_name,
                        image=image
                    ),
                    on_logs=lambda log: logger.debug("Snapshot build", log=log)
                )
                logger.info("Snapshot created successfully", snapshot_name=snapshot_name)
                return snapshot_name
            except OSError as e:
                error_str = str(e)
                # Check if snapshot already exists (race condition or list failed)
                if "already exists" in error_str.lower():
                    logger.info(
                        "Snapshot already exists, will use it",
                        snapshot_name=snapshot_name
                    )
                    return snapshot_name
                logger.error("Failed to create snapshot", error=error_str)
                return None

        if snapshot_exists:
            logger.info("Using existing snapshot", snapshot_name=snapshot_name)
            return snapshot_name

        logger.warning("Snapshot not found and auto_create disabled")
        return None

    async def setup_sandbox_workspace(self) -> str | None:
        """Create sandbox and setup workspace directories.

        Can run concurrently with MCP registry connection since it doesn't
        require the registry.

        Returns:
            snapshot_name if used, None otherwise
        """
        logger.info("Setting up sandbox workspace")

        # Try to use snapshot if enabled
        snapshot_name = await self._ensure_snapshot()

        if snapshot_name:
            # Create sandbox from snapshot (FAST!)
            logger.info("Creating sandbox from snapshot", snapshot_name=snapshot_name)
            try:
                self.sandbox = await self._run_sync(
                    self.daytona_client.create,
                    CreateSandboxFromSnapshotParams(snapshot=snapshot_name)
                )
                logger.info("Sandbox created from snapshot", snapshot_name=snapshot_name)
            except OSError as e:
                logger.warning(
                    "Failed to create from snapshot, falling back to default",
                    error=str(e)
                )
                snapshot_name = None

        if not snapshot_name:
            # Fallback to default creation
            logger.info("Creating sandbox from default image")
            self.sandbox = await self._run_sync(self.daytona_client.create)

            self.sandbox_id = self.sandbox.id if hasattr(self.sandbox, "id") else str(id(self.sandbox))
            logger.info("Daytona sandbox created", sandbox_id=self.sandbox_id)

            # Set up workspace structure
            await self._setup_workspace()

            # Install dependencies
            await self._install_dependencies()
        else:
            # Snapshot-based creation
            assert self.sandbox is not None
            self.sandbox_id = self.sandbox.id if hasattr(self.sandbox, "id") else str(id(self.sandbox))
            logger.info(
                "Sandbox ready from snapshot",
                sandbox_id=self.sandbox_id,
                snapshot=snapshot_name
            )
            # Ensure workspace directories exist (results, data, etc.)
            await self._setup_workspace()

        logger.info("Sandbox workspace ready", sandbox_id=self.sandbox_id)
        return snapshot_name

    async def setup_tools_and_mcp(self, snapshot_name: str | None) -> None:
        """Install tool modules and start MCP servers.

        Requires MCP registry to be connected first.

        Args:
            snapshot_name: Snapshot name from setup_sandbox_workspace(), or None
        """
        logger.info("Setting up tools and MCP servers")

        # Upload custom Python MCP server files to sandbox
        await self._upload_mcp_server_files()

        # Always generate and install tool modules (dynamic content)
        await self._install_tool_modules()

        # Start internal MCP servers (when using snapshot with Node.js)
        if snapshot_name:
            # Node.js and MCP packages are available in snapshot
            await self._start_internal_mcp_servers()
        else:
            logger.warning(
                "Skipping internal MCP servers - not using snapshot. "
                "MCP tools will not work without snapshot."
            )

        logger.info("Tools and MCP servers ready", sandbox_id=self.sandbox_id)

    async def setup(self) -> None:
        """Set up the sandbox environment.

        For async initialization, use setup_sandbox_workspace() and
        setup_tools_and_mcp() separately via Session.initialize().
        """
        snapshot_name = await self.setup_sandbox_workspace()
        await self.setup_tools_and_mcp(snapshot_name)
        logger.info("Sandbox setup complete", sandbox_id=self.sandbox_id)

    async def reconnect(self, sandbox_id: str) -> None:
        """Reconnect to a stopped sandbox.

        This is a fast path for session persistence - it starts a stopped
        sandbox and skips all setup work (file uploads, tool modules, etc.)
        since they're already present from the first session.

        Args:
            sandbox_id: The ID of an existing Daytona sandbox

        Raises:
            RuntimeError: If sandbox cannot be found or is in invalid state
        """
        logger.info("Reconnecting to stopped sandbox", sandbox_id=sandbox_id)

        # Get the existing sandbox from Daytona with error handling
        try:
            self.sandbox = await self._run_sync(self.daytona_client.get, sandbox_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to find sandbox {sandbox_id}. It may have been deleted. "
                f"Original error: {e}"
            ) from e

        self.sandbox_id = sandbox_id

        # Check sandbox state before attempting to start
        state = getattr(self.sandbox, "state", None)
        if state:
            state_value = state.value if hasattr(state, "value") else str(state)
            if state_value == "started":
                logger.info("Sandbox already started, skipping start", sandbox_id=sandbox_id)
            elif state_value in ("stopped", "starting"):
                logger.info("Starting stopped sandbox", sandbox_id=sandbox_id, state=state_value)
                await self._run_sync(self.sandbox.start, timeout=60)
            else:
                raise RuntimeError(
                    f"Cannot reconnect to sandbox in state: {state_value}. "
                    f"Expected 'stopped' or 'started'."
                )
        else:
            # No state attribute, assume we need to start
            logger.info("Starting sandbox (state unknown)", sandbox_id=sandbox_id)
            await self._run_sync(self.sandbox.start, timeout=60)

        # Get work directory reference
        self._work_dir = await self._run_sync(self.sandbox.get_work_dir)
        logger.info(f"Sandbox working directory: {self._work_dir}")

        # SKIP: _setup_workspace() - directories already exist
        # SKIP: _upload_mcp_server_files() - files already uploaded
        # SKIP: _install_tool_modules() - tool modules already installed

        # Initialize MCP server sessions (needed for tool execution)
        self.mcp_server_sessions: dict[str, Any] = {}
        await self._start_internal_mcp_servers()

        logger.info(
            "Sandbox started from stopped state",
            sandbox_id=self.sandbox_id,
        )

    async def stop_sandbox(self) -> None:
        """Stop the sandbox without deleting it.

        Used for session persistence - stops the sandbox so it can be
        restarted quickly on the next session, rather than deleting it.
        """
        if self.sandbox:
            logger.info("Stopping sandbox", sandbox_id=self.sandbox_id)
            await self._run_sync(self.sandbox.stop, timeout=60)
            logger.info("Sandbox stopped", sandbox_id=self.sandbox_id)

    async def _setup_workspace(self) -> None:
        """Create workspace directory structure."""
        logger.info("Setting up workspace structure")

        # Get the working directory
        assert self.sandbox is not None
        work_dir = await self._run_sync(self.sandbox.get_work_dir)
        logger.info(f"Sandbox working directory: {work_dir}")

        # Store work_dir for use by other methods
        self._work_dir = work_dir

        # Use absolute paths to ensure directories are created correctly
        directories = [
            f"{work_dir}/tools",
            f"{work_dir}/tools/docs",
            f"{work_dir}/results",
            f"{work_dir}/data",
            f"{work_dir}/code",
        ]

        # Create all directories in parallel for faster setup
        async def create_directory(directory: str) -> None:
            try:
                assert self.sandbox is not None
                await self._run_sync(self.sandbox.process.exec, f"mkdir -p {directory}")
                logger.info(f"Created directory: {directory}")
            except OSError as e:
                logger.warning(f"Error creating directory {directory}: {e}")

        await asyncio.gather(*[create_directory(d) for d in directories])

    async def _upload_mcp_server_files(self) -> None:
        """Upload custom Python MCP server files to sandbox.

        For Python MCP servers configured with 'uv run python mcp_servers/xxx.py',
        this method uploads the Python files to the sandbox so they can be executed
        as subprocesses inside the sandbox environment.
        """
        work_dir = getattr(self, "_work_dir", "/home/daytona")
        mcp_servers_dir = f"{work_dir}/mcp_servers"

        # Collect files to upload
        files_to_upload = []

        # Get config file directory
        config_dir = getattr(self.config, "config_file_dir", None)

        for server in self.config.mcp.servers:
            if not server.enabled:
                continue
            # Only handle Python MCP servers (uv run python ...)
            if server.transport == "stdio" and server.command == "uv":
                if len(server.args) >= 3 and server.args[0] == "run" and server.args[1] == "python":
                    local_path = server.args[2]  # e.g., "mcp_servers/yfinance_mcp_server.py"

                    # Resolve relative paths against config file directory first
                    path_obj = Path(local_path)
                    resolved_path = None

                    if not path_obj.is_absolute() and config_dir:
                        # Try resolving against config file directory
                        config_relative_path = (config_dir / local_path).resolve()
                        if config_relative_path.exists():
                            resolved_path = str(config_relative_path)
                            logger.debug(
                                "Resolved MCP server path relative to config",
                                server=server.name,
                                original=local_path,
                                resolved=resolved_path,
                            )

                    # Fall back to CWD-relative path
                    if resolved_path is None and path_obj.exists():
                        resolved_path = local_path

                    if resolved_path:
                        filename = Path(resolved_path).name
                        sandbox_path = f"{mcp_servers_dir}/{filename}"
                        files_to_upload.append((server.name, resolved_path, sandbox_path))
                    else:
                        searched_paths = [local_path]
                        if config_dir:
                            searched_paths.append(str(config_dir / local_path))
                        logger.warning(
                            f"MCP server file not found: {local_path}",
                            server=server.name,
                            searched_paths=searched_paths,
                        )

        # If we have files to upload, create directory and upload in parallel
        if files_to_upload:
            assert self.sandbox is not None
            await self._run_sync(self.sandbox.process.exec, f"mkdir -p {mcp_servers_dir}")

            async def upload_file(server_name: str, local_path: str, sandbox_path: str) -> None:
                # Read file from host using aiofiles to avoid blocking
                async with aiofiles.open(local_path) as f:
                    content = await f.read()

                # Upload to sandbox
                assert self.sandbox is not None
                await self._run_sync(
                    self.sandbox.fs.upload_file,
                    content.encode("utf-8"),
                    sandbox_path
                )

                logger.info(
                    "Uploaded MCP server file",
                    server=server_name,
                    local_path=local_path,
                    sandbox_path=sandbox_path
                )

            # Upload all files in parallel
            await asyncio.gather(*[
                upload_file(server_name, local_path, sandbox_path)
                for server_name, local_path, sandbox_path in files_to_upload
            ])

    async def _install_dependencies(self) -> None:
        """Install required Python packages in sandbox."""
        logger.info("Installing dependencies")

        dependencies = [
            "mcp",
            "pandas",
            "requests",
            "aiohttp",
        ]

        install_cmd = f"uv pip install -q {' '.join(dependencies)}"

        try:
            assert self.sandbox is not None
            _result = await self._run_sync(self.sandbox.process.exec, install_cmd)
            logger.info("Dependencies installed")
        except OSError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise

    async def _install_tool_modules(self) -> None:
        """Generate and install tool modules from MCP servers."""
        logger.info("Installing tool modules")

        # Get work directory (set by _setup_workspace)
        work_dir = getattr(self, "_work_dir", "/home/daytona")

        # Collect all files to upload (content generation is CPU-bound, fast)
        uploads: list[tuple[bytes, str, tuple[str, dict[str, str]] | None]] = []

        # 1. MCP client module
        mcp_client_code = self.tool_generator.generate_mcp_client_code(
            self.config.mcp.servers
        )
        mcp_client_path = f"{work_dir}/tools/mcp_client.py"
        uploads.append((
            mcp_client_code.encode("utf-8"),
            mcp_client_path,
            ("MCP client module installed", {"path": mcp_client_path})
        ))

        # 2. Tool modules and documentation
        assert self.mcp_registry is not None
        tools_by_server = self.mcp_registry.get_all_tools()

        # Create per-server doc directories
        assert self.sandbox is not None
        for server_name in tools_by_server:
            doc_dir = f"{work_dir}/tools/docs/{server_name}"
            await self._run_sync(self.sandbox.process.exec, f"mkdir -p {doc_dir}")

        for server_name, tools in tools_by_server.items():
            # Generate Python module
            module_code = self.tool_generator.generate_tool_module(
                server_name, tools
            )
            module_path = f"{work_dir}/tools/{server_name}.py"
            uploads.append((
                module_code.encode("utf-8"),
                module_path,
                ("Tool module installed", {"server": server_name, "path": module_path, "tool_count": str(len(tools))})
            ))

            # Generate documentation for each tool
            for tool in tools:
                doc = self.tool_generator.generate_tool_documentation(tool)
                doc_path = f"{work_dir}/tools/docs/{server_name}/{tool.name}.md"
                upload_item: tuple[bytes, str, tuple[str, dict[str, str]] | None] = (doc.encode("utf-8"), doc_path, None)
                uploads.append(upload_item)

        # 3. __init__.py for tools package
        init_content = '"""Auto-generated tool modules from MCP servers."""\n'
        init_path = f"{work_dir}/tools/__init__.py"
        init_item: tuple[bytes, str, tuple[str, dict[str, str]] | None] = (init_content.encode("utf-8"), init_path, None)
        uploads.append(init_item)

        # Upload all files in parallel
        async def upload_file(content_bytes: bytes, path: str, log_info: tuple[str, dict[str, str]] | None) -> None:
            assert self.sandbox is not None
            await self._run_sync(
                self.sandbox.fs.upload_file,
                content_bytes,
                path
            )
            if log_info:
                msg, kwargs = log_info
                logger.info(msg, **kwargs)

        await asyncio.gather(*[
            upload_file(content, path, log_info)
            for content, path, log_info in uploads
        ])

        logger.info("Tool modules installation complete")

    async def _start_internal_mcp_servers(self) -> None:
        """Start MCP servers as background processes inside sandbox."""
        logger.info("Starting internal MCP servers")

        # Track server sessions for lifecycle management
        self.mcp_server_sessions = {}

        for server in self.config.mcp.servers:
            if not server.enabled:
                continue
            if server.transport != "stdio":
                logger.warning(
                    f"Skipping non-stdio server {server.name}",
                    transport=server.transport
                )
                continue

            try:
                # Build the command to start the MCP server
                if server.command == "npx":
                    # npx -y package-name [args...]
                    cmd_parts = [server.command, *server.args]
                    cmd = " ".join(cmd_parts)
                else:
                    # Custom command
                    cmd = f"{server.command} {' '.join(server.args)}"

                # Add environment variables if specified
                env_vars = []
                if hasattr(server, "env") and server.env:
                    for key, value in server.env.items():
                        # Environment variables might have ${VAR} syntax, resolve them
                        # For now, we'll pass them as-is and they'll need to be set in sandbox
                        env_vars.append(f"{key}={value}")

                # Create PTY session for the MCP server
                session_name = f"mcp-{server.name}"

                logger.info(
                    "Creating MCP server session",
                    server=server.name,
                    session=session_name,
                    command=cmd
                )

                # Create session (but don't start the server yet, we'll do that when needed)
                # For now, just track that this server should be available
                self.mcp_server_sessions[server.name] = {
                    "session_name": session_name,
                    "command": cmd,
                    "env": env_vars,
                    "started": False
                }

                logger.info(
                    "MCP server session configured",
                    server=server.name,
                    session=session_name
                )

            except OSError as e:
                logger.error(
                    "Failed to configure MCP server session",
                    server=server.name,
                    error=str(e)
                )

        logger.info(
            "Internal MCP server configuration complete",
            servers=list(self.mcp_server_sessions.keys())
        )

    def _detect_missing_imports(self, stderr: str) -> list[str]:
        """Extract missing module names from ImportError/ModuleNotFoundError.

        Args:
            stderr: Standard error output from code execution

        Returns:
            List of missing package names (base package only, e.g., 'foo' from 'foo.bar')
        """
        import re
        patterns = [
            r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]",
            r"ImportError: No module named ['\"]([^'\"]+)['\"]",
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, stderr))

        # Handle submodule imports (e.g., "foo.bar" -> "foo")
        # Also deduplicate
        base_packages = list({m.split(".")[0] for m in matches})

        if base_packages:
            logger.info(
                "Detected missing imports",
                packages=base_packages,
            )

        return base_packages

    async def _install_package(self, package: str) -> bool:
        """Install a Python package in the sandbox.

        Args:
            package: Package name to install

        Returns:
            True if installation succeeded, False otherwise
        """
        try:
            logger.info(f"Auto-installing missing package: {package}")
            assert self.sandbox is not None
            result = await self._run_sync(
                self.sandbox.process.exec,
                f"uv pip install -q {package}"
            )
            exit_code = getattr(result, "exit_code", 1)
            if exit_code == 0:
                logger.info(f"Successfully installed package: {package}")
                return True
            logger.warning(f"Failed to install package: {package}, exit_code={exit_code}")
            return False
        except OSError as e:
            logger.warning(f"Failed to install {package}: {e}")
            return False

    async def execute(
        self, code: str, timeout: int | None = None, *, auto_install: bool = True, max_retries: int = 2
    ) -> ExecutionResult:
        """Execute Python code in the sandbox with optional auto-install for missing dependencies.

        Args:
            code: Python code to execute
            timeout: Optional timeout in seconds
            auto_install: Whether to automatically install missing packages on ImportError (default: True)
            max_retries: Maximum number of retries after auto-installing packages (default: 2)

        Returns:
            ExecutionResult with execution details
        """
        if not self.sandbox:
            raise RuntimeError("Sandbox not initialized. Call setup() first.")

        self.execution_count += 1
        execution_id = f"exec_{self.execution_count:04d}"
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        logger.info(
            "Executing code",
            execution_id=execution_id,
            code_hash=code_hash,
            code_length=len(code),
            auto_install=auto_install,
        )

        start_time = time.time()

        try:
            # Write code to file
            code_path = f"code/{execution_id}.py"
            await self._run_sync(
                self.sandbox.fs.upload_file,
                code.encode("utf-8"),
                code_path
            )

            # Get list of files before execution
            files_before = await self._list_result_files()

            # Execute code
            timeout_val = timeout or self.config.security.max_execution_time

            # Set PYTHONPATH to working directory so code can import from tools/
            # Also pass MCP server environment variables
            work_dir = await self._run_sync(self.sandbox.get_work_dir)

            exec_env = {"PYTHONPATH": work_dir}

            # Add environment variables from MCP server configs (only enabled servers)
            import os
            for server in self.config.mcp.servers:
                if not server.enabled:
                    continue
                if hasattr(server, "env") and server.env:
                    for key, value in server.env.items():
                        # Resolve ${VAR} placeholders from host environment
                        if value.startswith("${") and value.endswith("}"):
                            var_name = value[2:-1]
                            resolved_value = os.getenv(var_name)
                            if resolved_value:
                                exec_env[key] = resolved_value
                        else:
                            exec_env[key] = value

            # Use code_run() for native artifact support (captures matplotlib charts)
            from daytona_sdk.common.process import CodeRunParams

            result = await self._run_sync(
                self.sandbox.process.code_run,
                code,
                params=CodeRunParams(env=exec_env),
                timeout=timeout_val
            )

            # Get stdout/stderr and exit code from Daytona ExecuteResponse
            # The result object has: exit_code, result (stdout), artifacts
            if hasattr(result, "result"):
                # Daytona SDK ExecuteResponse.result contains the stdout
                stdout = result.result or ""
            elif hasattr(result, "stdout"):
                stdout = result.stdout or ""
            else:
                stdout = ""

            # Get stderr - check multiple possible locations
            if hasattr(result, "stderr"):
                stderr = result.stderr or ""
            elif hasattr(result, "artifacts") and hasattr(result.artifacts, "stderr"):
                stderr = result.artifacts.stderr or ""
            else:
                stderr = ""

            exit_code = getattr(result, "exit_code", 1)

            # Determine success based on exit code
            success = (exit_code == 0)

            # Extract charts from artifacts (matplotlib captures)
            charts = []
            if hasattr(result, "artifacts") and result.artifacts and hasattr(result.artifacts, "charts") and result.artifacts.charts:
                for chart in result.artifacts.charts:
                    chart_type = chart.type.value if hasattr(chart.type, "value") else str(chart.type)
                    charts.append(ChartData(
                        type=chart_type,
                        title=chart.title if hasattr(chart, "title") else "",
                        png_base64=chart.png if hasattr(chart, "png") else None,
                        elements=chart.elements if hasattr(chart, "elements") else []
                    ))
                logger.info(f"Captured {len(charts)} chart(s) from artifacts")

            # Get files after execution
            files_after = await self._list_result_files()

            # Determine file changes
            files_created = [f for f in files_after if f not in files_before]
            files_modified: list[str] = []  # TODO: Implement modification tracking

            duration = time.time() - start_time

            execution_result = ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                duration=duration,
                files_created=files_created,
                files_modified=files_modified,
                execution_id=execution_id,
                code_hash=code_hash,
                charts=charts,
            )

            # Auto-install missing packages and retry if enabled
            if not success and auto_install and max_retries > 0:
                missing_packages = self._detect_missing_imports(stderr)
                if missing_packages:
                    logger.info(
                        "Attempting auto-install and retry",
                        execution_id=execution_id,
                        missing_packages=missing_packages,
                        retries_remaining=max_retries,
                    )

                    # Install missing packages
                    for package in missing_packages:
                        await self._install_package(package)

                    # Retry execution with decremented retry count
                    return await self.execute(
                        code=code,
                        timeout=timeout,
                        auto_install=auto_install,
                        max_retries=max_retries - 1
                    )

            logger.info(
                "Code execution completed",
                execution_id=execution_id,
                success=success,
                duration=duration,
                files_created=len(files_created),
                charts_captured=len(charts),
            )

            return execution_result

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                "Code execution failed",
                execution_id=execution_id,
                error=str(e),
                duration=duration,
            )

            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                duration=duration,
                files_created=[],
                files_modified=[],
                execution_id=execution_id,
                code_hash=code_hash,
                charts=[],
            )

    async def execute_bash_command(
        self, command: str, working_dir: str = "/home/daytona", timeout: int = 60, *, background: bool = False
    ) -> dict[str, Any]:
        """Execute a bash command in the sandbox.

        Args:
            command: Bash command to execute
            working_dir: Working directory for command execution (default: /home/daytona)
            timeout: Maximum execution time in seconds (default: 60)
            background: Run command in background (not fully implemented yet)

        Returns:
            Dictionary with success, stdout, stderr, exit_code, bash_id, command_hash
        """
        try:
            # Generate bash execution ID for tracking
            self.bash_execution_count += 1
            bash_id = f"bash_{self.bash_execution_count:04d}"
            command_hash = hashlib.sha256(command.encode()).hexdigest()[:16]
            from datetime import UTC, datetime
            timestamp = datetime.now(tz=UTC).isoformat()

            logger.info(
                "Executing bash command",
                bash_id=bash_id,
                command_hash=command_hash,
                command=command[:100],
                working_dir=working_dir,
            )

            # Build the full bash command with working directory
            # Use cd to change directory, then execute command
            full_command = f"cd {working_dir} && {command}"

            # Create a shell script with metadata header for logging
            script_content = textwrap.dedent(f"""\
                #!/bin/bash
                # Bash Execution Log
                # ID: {bash_id}
                # Working Directory: {working_dir}
                # Timestamp: {timestamp}
                # Command Hash: {command_hash}

                set -e  # Exit on error (optional, can be removed for more lenient execution)
                {full_command}
            """)

            # Write script to code/ directory for persistent logging
            # Use relative path for upload (Daytona SDK handles it relative to work_dir)
            script_relative_path = f"code/{bash_id}.sh"
            assert self.sandbox is not None
            await self._run_sync(
                self.sandbox.fs.upload_file,
                script_content.encode("utf-8"),
                script_relative_path
            )

            # Get work directory for absolute path in bash execution
            work_dir_path = getattr(self, "_work_dir", "/home/daytona")
            script_absolute_path = f"{work_dir_path}/{script_relative_path}"

            # Execute the script using the sandbox's execution method
            # Since Daytona SDK uses process.execute, we'll use Python to run bash
            python_wrapper = textwrap.dedent(f"""\
                import subprocess
                import sys

                try:
                    result = subprocess.run(
                        ['bash', '{script_absolute_path}'],
                        capture_output=True,
                        text=True,
                        timeout={timeout}
                    )
                    print(result.stdout, end='')  # noqa: T201
                    sys.stderr.write(result.stderr)
                    sys.exit(result.returncode)
                except subprocess.TimeoutExpired:
                    sys.stderr.write(f"Command timed out after {timeout} seconds")
                    sys.exit(124)
                except (OSError, subprocess.SubprocessError) as e:
                    sys.stderr.write(f"Error executing command: {{e}}")
                    sys.exit(1)
            """)

            # Execute via Python wrapper
            result = await self.execute(python_wrapper)

            # Parse the result
            if result.success:
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": 0,
                    "bash_id": bash_id,
                    "command_hash": command_hash,
                }
            # Extract exit code from stderr if possible
            exit_code = 1
            stderr = result.stderr if result.stderr else result.stdout

            return {
                "success": False,
                "stdout": result.stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "bash_id": bash_id,
                "command_hash": command_hash,
            }

        except Exception as e:
            logger.error(f"Failed to execute bash command: {e}", exc_info=True)
            # Note: bash_id may not be defined if error occurs early
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Exception during bash execution: {e!s}",
                "exit_code": -1,
                "bash_id": getattr(self, "_last_bash_id", None),
                "command_hash": None,
            }

    async def _list_result_files(self) -> list[str]:
        """List files in the results directory.

        Returns:
            List of file paths relative to workspace (e.g., "results/file.csv")
        """
        try:
            assert self.sandbox is not None
            file_infos = await self._run_sync(self.sandbox.fs.list_files, "results")
            if not file_infos:
                return []
            # Return paths relative to workspace, not just filenames
            return [f"results/{str(f.name) if hasattr(f, 'name') else str(f)}" for f in file_infos]
        except (OSError, AttributeError) as e:
            logger.warning(f"Error listing result files: {e}")
            return []

    def read_file(self, filepath: str) -> str | None:
        """Read a file from the sandbox.

        Args:
            filepath: Path to file in sandbox

        Returns:
            File contents or None if error
        """
        try:
            # download_file returns bytes
            assert self.sandbox is not None
            content_bytes = self.sandbox.fs.download_file(filepath)
            return content_bytes.decode("utf-8") if content_bytes else None
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            return None

    def download_file_bytes(self, filepath: str) -> bytes | None:
        """Download raw bytes from sandbox - works for any file type including images.

        Args:
            filepath: Path to file in sandbox

        Returns:
            Raw bytes or None if error
        """
        try:
            assert self.sandbox is not None
            return self.sandbox.fs.download_file(filepath)
        except OSError as e:
            logger.error(f"Failed to download file bytes {filepath}: {e}")
            return None

    def download_file(self, sandbox_path: str, local_path: str) -> bool:
        """Download a file from sandbox to local filesystem.

        Args:
            sandbox_path: Path in sandbox
            local_path: Local path to save to

        Returns:
            True if successful
        """
        try:
            content = self.read_file(sandbox_path)
            if content:
                Path(local_path).write_text(content)
                logger.info(f"Downloaded {sandbox_path} to {local_path}")
                return True
            return False
        except OSError as e:
            logger.error(f"Failed to download file: {e}")
            return False

    def get_file_info(self, filepath: str) -> dict[str, Any]:
        """Get information about a file.

        Args:
            filepath: Path to file

        Returns:
            Dictionary with file information
        """
        try:
            content = self.read_file(filepath)
            if content:
                return {
                    "path": filepath,
                    "size": len(content),
                    "lines": len(content.splitlines()),
                    "exists": True,
                }
            return {"path": filepath, "exists": False}
        except (OSError, AttributeError) as e:
            logger.error(f"Error getting file info: {e}")
            return {"path": filepath, "exists": False, "error": str(e)}

    def normalize_path(self, path: str) -> str:
        """Normalize virtual path to absolute sandbox path (input normalization).

        Converts agent's virtual paths to real sandbox paths:
            "/" or "." or "" -> {working_directory}
            "/results/file.txt" -> {working_directory}/results/file.txt
            "data/file.txt" -> {working_directory}/data/file.txt
            "{working_directory}/file.txt" -> unchanged
            "/tmp/file.txt" -> unchanged

        Args:
            path: Virtual or relative path from agent

        Returns:
            Absolute sandbox path
        """
        # Use configured working_directory as the prefix for path normalization
        work_dir = self.config.filesystem.working_directory

        if path in (None, "", ".", "/"):
            return work_dir

        path = path.strip()

        # Already in allowed directories - keep as is (just normalize . and ..)
        for allowed_dir in self.config.filesystem.allowed_directories:
            if path.startswith(allowed_dir):
                return str(Path(path))

        # Virtual absolute path: /foo -> /home/daytona/foo
        if path.startswith("/"):
            return str(Path(f"{work_dir}{path}"))

        # Relative path: foo -> /home/daytona/foo
        return str(Path(f"{work_dir}/{path}"))

    def virtualize_path(self, path: str) -> str:
        """Convert real sandbox path to virtual path (output normalization).

        Strips working_directory prefix from paths returned to agent:
            {working_directory}/results/file.txt -> /results/file.txt
            {working_directory}/tools/docs/foo.md -> /tools/docs/foo.md
            /tmp/file.txt -> /tmp/file.txt (unchanged)

        Args:
            path: Absolute sandbox path

        Returns:
            Virtual path for agent consumption
        """
        # Use configured working_directory as the prefix to strip
        work_dir = self.config.filesystem.working_directory

        if path.startswith(work_dir + "/"):
            return path[len(work_dir):]  # Strip prefix, keep leading /
        if path == work_dir:
            return "/"

        return path  # /tmp or other paths unchanged

    def validate_path(self, filepath: str) -> bool:
        """Validate if a path is within allowed directories.

        Args:
            filepath: Path to validate (virtual or absolute)

        Returns:
            True if path is allowed, False otherwise
        """
        if not self.config.filesystem.enable_path_validation:
            return True

        # Normalize the path first (handles virtual paths like /results/...)
        normalized_path = self.normalize_path(filepath)

        # Check against allowed directories
        for allowed_dir in self.config.filesystem.allowed_directories:
            # Exact match or path within allowed directory
            if normalized_path == allowed_dir or normalized_path.startswith(allowed_dir + "/"):
                return True

        logger.warning(
            "Path validation failed",
            path=filepath,
            normalized_path=normalized_path,
            allowed_dirs=self.config.filesystem.allowed_directories,
        )
        return False

    def validate_and_normalize_path(self, path: str) -> tuple[str, str | None]:
        """Normalize path and validate access.

        Combines path normalization and validation into a single operation.

        Args:
            path: Virtual or relative path from agent

        Returns:
            Tuple of (normalized_path, error_message_or_none)
        """
        normalized = self.normalize_path(path)
        if self.config.filesystem.enable_path_validation and not self.validate_path(normalized):
            return normalized, f"Access denied: {path} is not in allowed directories"
        return normalized, None

    def write_file(self, filepath: str, content: str) -> bool:
        """Write content to a file in the sandbox.

        Args:
            filepath: Path to file in sandbox
            content: Content to write

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.config.filesystem.enable_path_validation and not self.validate_path(filepath):
                logger.error(f"Access denied: {filepath} is not in allowed directories")
                return False

            # Upload file via Daytona SDK
            assert self.sandbox is not None
            self.sandbox.fs.upload_file(content.encode("utf-8"), filepath)
            logger.info(f"Wrote {len(content)} bytes to {filepath}")
            return True
        except (OSError, UnicodeEncodeError) as e:
            logger.error(f"Failed to write file {filepath}: {e}")
            return False

    def list_directory(self, directory: str = ".") -> list[dict[str, Any]]:
        """List contents of a directory with type indicators.

        Args:
            directory: Directory path (default: current directory)

        Returns:
            List of dictionaries with name and type (file/directory)
        """
        try:
            if self.config.filesystem.enable_path_validation and not self.validate_path(directory):
                logger.error(f"Access denied: {directory} is not in allowed directories")
                return []

            assert self.sandbox is not None
            file_infos = self.sandbox.fs.list_files(directory)
            if not file_infos:
                return []

            results = []
            for f in file_infos:
                name = str(f.name) if hasattr(f, "name") else str(f)
                is_dir = hasattr(f, "is_dir") and f.is_dir
                results.append({
                    "name": name,
                    "type": "directory" if is_dir else "file",
                    "path": f"{directory}/{name}" if directory != "." else name,
                })

            return results
        except (OSError, AttributeError) as e:
            logger.debug(f"Error listing directory {directory}: {e}")
            return []

    def create_directory(self, dirpath: str) -> bool:
        """Create a directory in the sandbox.

        Args:
            dirpath: Directory path to create

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.config.filesystem.enable_path_validation and not self.validate_path(dirpath):
                logger.error(f"Access denied: {dirpath} is not in allowed directories")
                return False

            # Create directory by uploading an empty file, then deleting it
            # This ensures parent directories are created
            temp_file = f"{dirpath}/.gitkeep"
            assert self.sandbox is not None
            self.sandbox.fs.upload_file(b"", temp_file)
            logger.info(f"Created directory {dirpath}")
            return True
        except OSError as e:
            logger.error(f"Failed to create directory {dirpath}: {e}")
            return False

    def edit_file(self, filepath: str, old_string: str, new_string: str, *, replace_all: bool = False) -> dict[str, Any]:
        """Edit a file using exact string replacement (Claude Code standard).

        Performs exact string matching (whitespace-sensitive). old_string must be
        unique in the file unless replace_all=True.

        Args:
            filepath: Path to file (absolute)
            old_string: The exact text to replace (must exist in file)
            new_string: The text to replace it with (must be different from old_string)
            replace_all: Replace all occurrences (useful for renaming)

        Returns:
            Dictionary with edit results (success, changes, message)
        """
        try:
            if self.config.filesystem.enable_path_validation and not self.validate_path(filepath):
                return {
                    "success": False,
                    "error": f"Access denied: {filepath} is not in allowed directories",
                }

            # Read current content
            content = self.read_file(filepath)
            if content is None:
                return {"success": False, "error": "File not found"}

            # Check if old_string and new_string are different
            if old_string == new_string:
                return {
                    "success": False,
                    "error": "old_string and new_string must be different",
                }

            # Check if old_string exists
            if old_string not in content:
                return {
                    "success": False,
                    "error": f"old_string not found in file: {filepath}",
                }

            # Check uniqueness if not replace_all
            if not replace_all:
                occurrences = content.count(old_string)
                if occurrences > 1:
                    return {
                        "success": False,
                        "error": f"old_string appears {occurrences} times in file. Use replace_all=True to replace all occurrences, or make old_string more specific to be unique.",
                    }

            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
                occurrences = content.count(old_string)
                message = f"Replaced {occurrences} occurrence(s) in {filepath}"
            else:
                new_content = content.replace(old_string, new_string, 1)
                message = f"Successfully edited {filepath}"

            # Write the edited content
            if self.write_file(filepath, new_content):
                return {
                    "success": True,
                    "changed": True,
                    "message": message,
                }
            return {"success": False, "error": "Failed to write edited content"}

        except (OSError, ValueError) as e:
            logger.error(f"Failed to edit file {filepath}: {e}")
            return {"success": False, "error": str(e)}

    def search_files(self, pattern: str, directory: str = ".", exclude: list[str] | None = None) -> list[str]:
        """Search for files matching a pattern (glob-style).

        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.txt")
            directory: Directory to search in
            exclude: List of patterns to exclude

        Returns:
            List of matching file paths
        """
        try:
            if self.config.filesystem.enable_path_validation and not self.validate_path(directory):
                logger.error(f"Access denied: {directory} is not in allowed directories")
                return []

            # Simple recursive file search
            # This is a basic implementation - production would use more sophisticated methods
            import fnmatch

            matches = []

            def search_recursive(current_dir: str) -> None:
                try:
                    entries = self.list_directory(current_dir)
                    for entry in entries:
                        full_path = entry["path"]

                        # Check exclusions
                        if exclude:
                            excluded = False
                            for excl_pattern in exclude:
                                if fnmatch.fnmatch(entry["name"], excl_pattern):
                                    excluded = True
                                    break
                            if excluded:
                                continue

                        # Match files
                        if entry["type"] == "file":
                            if fnmatch.fnmatch(entry["name"], pattern):
                                matches.append(full_path)

                        # Recurse into directories
                        elif entry["type"] == "directory":
                            search_recursive(full_path)

                except (OSError, KeyError) as e:
                    logger.debug(f"Error searching in {current_dir}: {e}")

            search_recursive(directory)
            logger.info(f"Found {len(matches)} files matching {pattern}")
            return matches

        except (OSError, ValueError) as e:
            logger.error(f"Failed to search files: {e}")
            return []

    def glob_files(self, pattern: str, path: str = ".") -> list[str]:
        """Find files matching a glob pattern, sorted by modification time.

        Uses Python's glob.glob in the sandbox for proper recursive pattern support.

        Args:
            pattern: Glob pattern (e.g., "**/*.py", "*.{js,ts}")
            path: Directory to search in (default: ".")

        Returns:
            List of matching file paths sorted by modification time (newest first)
        """
        try:
            if self.config.filesystem.enable_path_validation and not self.validate_path(path):
                logger.error(f"Access denied: {path} is not in allowed directories")
                return []

            # Normalize path for sandbox
            search_path = self._normalize_search_path(path)

            # Make glob recursive by default only for simple patterns (like "*.py")
            # Don't add ** if pattern already contains a path (has "/")
            if "**" not in pattern and "/" not in pattern:
                pattern = f"**/{pattern}"

            # Build Python code to execute glob in sandbox
            # This properly supports ** recursive patterns and mtime sorting
            glob_code = textwrap.dedent(f"""\
                import glob
                import os

                pattern = "{pattern}"
                search_path = "{search_path}"

                full_pattern = os.path.join(search_path, pattern)
                matches = glob.glob(full_pattern, recursive=True)
                files = [f for f in matches if os.path.isfile(f)]

                try:
                    files_with_mtime = [(f, os.path.getmtime(f)) for f in files]
                    sorted_files = sorted(files_with_mtime, key=lambda x: x[1], reverse=True)
                    for f, _ in sorted_files:
                        print(f)  # noqa: T201
                except OSError as e:
                    for f in files:
                        print(f)  # noqa: T201
            """)

            # Encode as base64 to safely pass multi-line code to shell
            encoded_code = base64.b64encode(glob_code.encode()).decode()
            cmd = f'python3 -c "import base64; exec(base64.b64decode(\'{encoded_code}\').decode())"'

            # Execute in sandbox
            assert self.sandbox is not None
            result = self.sandbox.process.exec(cmd, timeout=30)

            # Parse output
            output = result.result.strip() if result.result else ""

            if not output:
                logger.info(f"Found 0 files matching {pattern}")
                return []

            matches = output.split("\n")
            logger.info(f"Found {len(matches)} files matching {pattern}")
            return matches

        except (OSError, ValueError, AttributeError) as e:
            logger.error(f"Failed to glob files: {e}")
            # Fallback to search_files if glob execution fails
            return self.search_files(pattern, path)

    def grep_content(
        self,
        pattern: str,
        path: str = ".",
        output_mode: str = "files_with_matches",
        glob: str | None = None,
        type: str | None = None,  # noqa: A002 - matches ripgrep's --type flag
        *,
        case_insensitive: bool = False,
        show_line_numbers: bool = True,
        lines_after: int | None = None,
        lines_before: int | None = None,
        lines_context: int | None = None,
        multiline: bool = False,
        head_limit: int | None = None,
        offset: int = 0,
    ) -> Any:
        """Search file contents using ripgrep for high performance.

        Args:
            pattern: Regular expression pattern to search
            path: File or directory to search in
            output_mode: "files_with_matches", "content", or "count"
            glob: Filter files by glob pattern
            type: Filter by file type (e.g., "py", "js", "ts") - matches rg --type
            case_insensitive: Case insensitive search
            show_line_numbers: Show line numbers in content mode
            lines_after: Lines to show after match
            lines_before: Lines to show before match
            lines_context: Lines to show before and after match
            multiline: Enable multiline mode
            head_limit: Limit output to first N results
            offset: Skip first N results

        Returns:
            Search results based on output_mode
        """
        try:
            if self.config.filesystem.enable_path_validation and not self.validate_path(path):
                logger.error(f"Access denied: {path} is not in allowed directories")
                return []

            # Build ripgrep command
            cmd = ["rg"]

            # Output mode flags
            if output_mode == "files_with_matches":
                cmd.append("-l")
            elif output_mode == "count":
                cmd.append("-c")
            # "content" is default mode for rg

            # Case sensitivity
            if case_insensitive:
                cmd.append("-i")

            # Line numbers (only for content mode)
            if output_mode == "content" and show_line_numbers:
                cmd.append("-n")

            # Context lines
            if lines_before:
                cmd.extend(["-B", str(lines_before)])
            if lines_after:
                cmd.extend(["-A", str(lines_after)])
            if lines_context:
                cmd.extend(["-C", str(lines_context)])

            # Multiline mode
            if multiline:
                cmd.extend(["-U", "--multiline-dotall"])

            # File filtering
            if glob:
                cmd.extend(["--glob", glob])

            if type:
                cmd.extend(["--type", type])

            # Add pattern and path
            cmd.append(pattern)

            # Normalize path for sandbox
            search_path = self._normalize_search_path(path)
            cmd.append(search_path)

            # Execute ripgrep command
            cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
            logger.debug(f"Executing ripgrep: {cmd_str}")

            assert self.sandbox is not None
            result = self.sandbox.process.exec(cmd_str, timeout=60)

            # Parse output
            output = result.result.strip() if result.result else ""

            if not output:
                logger.info(f"Grep found 0 results for pattern {pattern}")
                return []

            # Process results based on output_mode
            if output_mode == "files_with_matches":
                results: list[str] | list[tuple[str, int]] = output.split("\n")
            elif output_mode == "count":
                # ripgrep count format: filename:count
                count_results: list[tuple[str, int]] = []
                for line in output.split("\n"):
                    if ":" in line:
                        parts = line.rsplit(":", 1)
                        if len(parts) == 2:
                            try:
                                count_results.append((parts[0], int(parts[1])))
                            except ValueError:
                                count_results.append((line, 0))
                    else:
                        count_results.append((line, 0))
                results = count_results
            else:  # content mode
                results = output.split("\n")

            # Apply offset and head_limit
            if offset > 0:
                results = results[offset:]
            if head_limit:
                results = results[:head_limit]

            logger.info(f"Grep found {len(results)} results for pattern {pattern}")
            return results

        except (OSError, ValueError) as e:
            logger.error(f"Failed to grep content: {e}")
            # Fallback to Python-based grep if ripgrep fails
            return self._grep_content_fallback(
                pattern=pattern,
                path=path,
                output_mode=output_mode,
                glob=glob,
                type=type,
                case_insensitive=case_insensitive,
                show_line_numbers=show_line_numbers,
                lines_after=lines_after,
                lines_before=lines_before,
                lines_context=lines_context,
                multiline=multiline,
                head_limit=head_limit,
                offset=offset,
            )

    def _grep_content_fallback(
        self,
        pattern: str,
        path: str = ".",
        output_mode: str = "files_with_matches",
        glob: str | None = None,
        type: str | None = None,  # noqa: A002 - matches ripgrep's --type flag
        *,
        case_insensitive: bool = False,
        show_line_numbers: bool = True,
        lines_after: int | None = None,
        lines_before: int | None = None,
        lines_context: int | None = None,
        multiline: bool = False,
        head_limit: int | None = None,
        offset: int = 0,
    ) -> Any:
        """Fallback Python-based grep if ripgrep is unavailable."""
        try:
            import re

            # Compile regex pattern
            flags = re.IGNORECASE if case_insensitive else 0
            if multiline:
                flags |= re.MULTILINE | re.DOTALL
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                logger.error(f"Invalid regex pattern in grep_content fallback: {e}")
                return []

            # Find files to search
            files = self.search_files(glob or "*", path)

            # Filter by type if specified
            if type:
                type_extensions: dict[str, str | list[str]] = {
                    "py": ".py",
                    "js": ".js",
                    "ts": ".ts",
                    "rust": ".rs",
                    "go": ".go",
                    "java": ".java",
                    "cpp": [".cpp", ".cc", ".cxx", ".h", ".hpp"],
                    "json": ".json",
                    "yaml": [".yaml", ".yml"],
                    "md": ".md",
                }
                if type in type_extensions:
                    ext = type_extensions[type]
                    if isinstance(ext, list):
                        ext_tuple = tuple(ext)
                        files = [f for f in files if f.endswith(ext_tuple)]
                    else:
                        files = [f for f in files if f.endswith(ext)]

            results: list[str] | list[tuple[str, int]] = []

            # Search each file
            for file_path in files:
                try:
                    content = self.read_file(file_path)
                    if content is None:
                        continue

                    if output_mode == "files_with_matches":
                        if regex.search(content):
                            results.append(file_path)  # type: ignore[arg-type]

                    elif output_mode == "count":
                        matches = regex.findall(content)
                        if matches:
                            results.append((file_path, len(matches)))  # type: ignore[arg-type]

                    elif output_mode == "content":
                        lines = content.splitlines()
                        for i, line in enumerate(lines):
                            if regex.search(line):
                                # Build context
                                context_before = lines_before or lines_context or 0
                                context_after = lines_after or lines_context or 0

                                start = max(0, i - context_before)
                                end = min(len(lines), i + context_after + 1)

                                context_lines = []
                                for j in range(start, end):
                                    prefix = f"{file_path}:" if show_line_numbers else ""
                                    line_num = f"{j+1}:" if show_line_numbers else ""
                                    marker = ">" if j == i else " "
                                    context_lines.append(f"{prefix}{line_num}{marker}{lines[j]}")

                                results.append("\n".join(context_lines))  # type: ignore[arg-type]

                except (OSError, UnicodeDecodeError) as e:
                    logger.debug(f"Error searching file {file_path}: {e}")
                    continue

            # Apply offset and head_limit
            if offset > 0:
                results = results[offset:]
            if head_limit:
                results = results[:head_limit]

            logger.info(f"Grep fallback found {len(results)} results for pattern {pattern}")
            return results

        except (OSError, ValueError, ImportError) as e:
            logger.error(f"Failed to grep content (fallback): {e}")
            return []

    def read_file_range(self, file_path: str, offset: int = 1, limit: int = 2000) -> str | None:
        """Read a specific range of lines from a file.

        Args:
            file_path: Path to the file
            offset: Line number to start from (1-indexed, default: 1 = first line)
            limit: Number of lines to read

        Returns:
            File content for the specified range, or None if file not found
        """
        try:
            if self.config.filesystem.enable_path_validation and not self.validate_path(file_path):
                logger.error(f"Access denied: {file_path} is not in allowed directories")
                return None

            # Read full file content
            content = self.read_file(file_path)
            if content is None:
                return None

            # Split into lines and extract range
            # Convert 1-indexed offset to 0-indexed for slicing
            lines = content.splitlines()
            start = max(0, offset - 1)
            end = start + limit

            selected_lines = lines[start:end]
            result = "\n".join(selected_lines)

            logger.info(
                "Read file range",
                file_path=file_path,
                offset=offset,
                limit=limit,
                lines_read=len(selected_lines),
            )

            return result

        except (OSError, ValueError) as e:
            logger.error(f"Failed to read file range: {e}")
            return None

    async def cleanup(self) -> None:
        """Clean up and destroy the sandbox."""
        logger.info("Cleaning up sandbox", sandbox_id=self.sandbox_id)

        if self.sandbox:
            try:
                await self._run_sync(self.sandbox.delete)
                logger.info("Sandbox deleted", sandbox_id=self.sandbox_id)
            except OSError as e:
                logger.error(f"Error deleting sandbox: {e}")

        self.sandbox = None
        self.sandbox_id = None

    async def __aenter__(self) -> "PTCSandbox":
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()
