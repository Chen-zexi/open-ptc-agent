---
description: Run linting (ruff, mypy) and tests for the project
---

# Lint & Test Command

Run code quality checks and tests for the ptc-agent monorepo.

## IMPORTANT: Package-Specific Configuration

This monorepo has two packages with separate linting configurations:
- `libs/ptc-agent/` - Core agent library
- `libs/ptc-cli/` - CLI application

**You MUST cd into each package directory before running linters** to ensure the package-specific `pyproject.toml` configuration takes effect.

## Your Task

Run linting and type checking on both packages. Follow this workflow:

1. **Run ruff check on both packages:**
   ```bash
   cd libs/ptc-agent && uv run ruff check .
   cd libs/ptc-cli && uv run ruff check .
   ```

2. **Run mypy on both packages:**
   ```bash
   cd libs/ptc-agent && uv run mypy ptc_agent
   cd libs/ptc-cli && uv run mypy ptc_cli
   ```

3. **If there are auto-fixable ruff issues**, fix them:
   ```bash
   cd libs/ptc-agent && uv run ruff check . --fix
   cd libs/ptc-cli && uv run ruff check . --fix
   ```

4. **Report results** - summarize any remaining issues that need manual fixes.

## Additional Commands (if requested)

### Format check
```bash
cd libs/ptc-agent && uv run ruff format --check .
cd libs/ptc-cli && uv run ruff format --check .
```

### Run tests
```bash
uv run pytest
uv run pytest libs/ptc-agent/tests/
uv run pytest libs/ptc-cli/tests/
```
