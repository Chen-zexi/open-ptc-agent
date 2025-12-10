# Changelog

All notable changes to Open PTC Agent are documented here.

Repository: https://github.com/Chen-zexi/open-ptc-agent

## [Dec 8-9, 2025]

### Breaking Changes
- Restructured codebase from `src/` to `libs/` with separate `ptc-agent/` and `ptc-cli/` packages ([f59e890](https://github.com/Chen-zexi/open-ptc-agent/commit/f59e890))

### Added
- Interactive CLI (`ptc-agent` command) with session persistence, plan mode, themes, and rich UI ([57ebb4d](https://github.com/Chen-zexi/open-ptc-agent/commit/57ebb4d))
- Context modes, LLM override, and path resolution in config ([fe78fd9](https://github.com/Chen-zexi/open-ptc-agent/commit/fe78fd9))
- Unit and integration tests for ptc-cli ([2f8726d](https://github.com/Chen-zexi/open-ptc-agent/commit/2f8726d))

### Changed
- Reorganized root level test suite into `libs/ptc-agent` structure ([e2ff1c8](https://github.com/Chen-zexi/open-ptc-agent/commit/e2ff1c8))
- Aligned code formatting and typing with ruff and mypy ([a890d9a](https://github.com/Chen-zexi/open-ptc-agent/commit/a890d9a))

### Fixed
- Message ordering for plan-mode and submit_plan approval/rejection flow ([657e83e](https://github.com/Chen-zexi/open-ptc-agent/commit/657e83e))

## [Dec 1-7, 2025]

### Added
- Sandbox session persistence and async sandbox initialization for improved lifecycle management ([c3ce2e5](https://github.com/Chen-zexi/open-ptc-agent/commit/c3ce2e5))
- Extended `astream` with full LangGraph streaming API compatibility in background orchestrator ([b777cfb](https://github.com/Chen-zexi/open-ptc-agent/commit/b777cfb))

### Changed
- Consolidated configuration files into single `config/` directory with enhanced config loading methods ([e1aad19](https://github.com/Chen-zexi/open-ptc-agent/commit/e1aad19))
- Extracted deepagent built-in middleware to separate module ([1d5138e](https://github.com/Chen-zexi/open-ptc-agent/commit/1d5138e))
- Streamlined tool docstrings to reduce prompt token usage ([967ea7d](https://github.com/Chen-zexi/open-ptc-agent/commit/967ea7d))

## [Nov 27 - Dec 1, 2025]

### Added
- Background subagent execution with fire and collect pattern, giving agents proactive control over task dispatch. Free the main agent from waiting for subagents to finish ([46fcf49](https://github.com/Chen-zexi/open-ptc-agent/commit/46fcf49))
- ViewImageMiddleware for multimodal image input support during agent runtime ([f62a4e6](https://github.com/Chen-zexi/open-ptc-agent/commit/f62a4e6))
- Configurable subagents section in config.yaml with `general-purpose` and `research` agents ([ee7cef1](https://github.com/Chen-zexi/open-ptc-agent/commit/ee7cef1))
- `wait()` and `task_progress()` tools for monitoring background tasks ([46fcf49](https://github.com/Chen-zexi/open-ptc-agent/commit/46fcf49))
- Gemini-3-pro-image model configuration in llms.json ([a95ec60](https://github.com/Chen-zexi/open-ptc-agent/commit/a95ec60))

### Changed
- Renamed `check_task_progress()` tool to `task_progress()` for brevity
- Reorganized prompt templates with new subagent coordination, tool discovery, and data processing components ([895b01b](https://github.com/Chen-zexi/open-ptc-agent/commit/895b01b))
- Extracted config factory functions to `config_loader.py` for DRY configuration loading ([ee7cef1](https://github.com/Chen-zexi/open-ptc-agent/commit/ee7cef1))
- Improved sandbox string formatting using `textwrap.dedent` for cleaner multi-line code ([a95ec60](https://github.com/Chen-zexi/open-ptc-agent/commit/a95ec60))

### Fixed
- Grep result parsing now handles string format `path:line:text` in addition to dict format ([a95ec60](https://github.com/Chen-zexi/open-ptc-agent/commit/a95ec60))
- Glob pattern matching no longer adds `**/` prefix to patterns containing paths ([a95ec60](https://github.com/Chen-zexi/open-ptc-agent/commit/a95ec60))
- Hardcoded test paths replaced with dynamic paths ([6448ec3](https://github.com/Chen-zexi/open-ptc-agent/commit/6448ec3))

## [~ - Nov 27 2025]
- Initial release of Open PTC Agent ([14e1d8f](https://github.com/Chen-zexi/open-ptc-agent/commit/14e1d8f))