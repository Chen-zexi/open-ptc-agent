---
description: Run pre-commit checks identical to GitHub CI (ruff, mypy, tests)
---

# Pre-Commit Check

Run all CI checks locally before committing. This matches the GitHub CI workflow exactly.

## Your Task

Run these checks in order, stopping if any fail:

### 1. Lint (ruff check)

```bash
uv run ruff check libs/ptc-agent/ptc_agent/ libs/ptc-agent/tests/
uv run ruff check libs/ptc-cli/ptc_cli/ libs/ptc-cli/tests/
```

If there are auto-fixable issues, fix them:
```bash
uv run ruff check libs/ptc-agent/ptc_agent/ libs/ptc-agent/tests/ --fix
uv run ruff check libs/ptc-cli/ptc_cli/ libs/ptc-cli/tests/ --fix
```

### 2. Type Check (mypy)

```bash
cd libs/ptc-agent && uv run mypy ptc_agent/
cd libs/ptc-cli && uv run mypy ptc_cli/
```

### 3. Unit Tests

```bash
uv run pytest libs/ptc-agent/tests/unit_tests/ -v
uv run pytest libs/ptc-cli/tests/unit_tests/ -v
```

### 4. Integration Tests

```bash
uv run pytest libs/ptc-agent/tests/integration_tests/ -v
uv run pytest libs/ptc-cli/tests/integration_tests/ -v
```

Note: Integration tests require `DAYTONA_API_KEY` and `ANTHROPIC_API_KEY` environment variables.

## Summary

Report results for each step:
- Lint: PASS/FAIL
- Type check: PASS/FAIL
- Unit tests: PASS/FAIL
- Integration tests: PASS/FAIL/SKIPPED (if env vars missing)
