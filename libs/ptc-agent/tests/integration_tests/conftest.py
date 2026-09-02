"""Pytest configuration for ptc-agent integration tests.

These tests spin up a real Daytona sandbox (and, for the LLM-dependent cases,
call the Anthropic API), so they require ``DAYTONA_API_KEY`` and
``ANTHROPIC_API_KEY`` to be present in the environment.

In contributor forks where those secrets are unavailable the suite cannot run,
so we skip the entire integration directory. This keeps CI green for forks
without provisioning infra, while the upstream CI (which provides the secrets)
still exercises the tests for real.
"""
from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_daytona = bool(os.getenv("DAYTONA_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    if has_daytona and has_anthropic:
        return
    missing = []
    if not has_daytona:
        missing.append("DAYTONA_API_KEY")
    if not has_anthropic:
        missing.append("ANTHROPIC_API_KEY")
    skip_marker = pytest.mark.skip(
        reason="Integration tests require live credentials; missing: " + ", ".join(missing)
    )
    for item in items:
        item.add_marker(skip_marker)
