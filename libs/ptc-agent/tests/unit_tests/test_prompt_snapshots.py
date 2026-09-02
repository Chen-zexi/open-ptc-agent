"""Snapshot tests for prompt rendering (ptc_agent.agent.prompts).

Prompt templates (system prompt, sub-agent prompts, shared components) are the
primary interface the agent presents to the LLM. A regression that silently
rewrites a prompt can change agent behaviour in ways that are invisible to unit
tests of Python logic. Snapshot tests make those changes explicit: every render
is compared byte-for-byte against a committed baseline, so a prompt edit shows up
as a diff in the PR.

Rendering is made deterministic by pinning `session_start_time`, which drives the
`{{ date }}` / `{{ datetime }}` variables injected into every template.

Run with `PTC_UPDATE_SNAPSHOTS=1` to (re)generate the baseline files under
`snapshots/prompts/`.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from ptc_agent.agent.prompts.loader import (
    PromptLoader,
    get_loader,
    init_loader,
    reset_loader,
)

FIXED_TIME = datetime(2026, 1, 15, 10, 30, 0, tzinfo=UTC)
SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "prompts"


def _assert_snapshot(name: str, content: str) -> None:
    """Compare rendered output against a committed baseline.

    Set PTC_UPDATE_SNAPSHOTS=1 to (re)write the baselines instead of asserting.
    """
    path = SNAPSHOT_DIR / f"{name}.md"
    if os.getenv("PTC_UPDATE_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
        return
    assert path.exists(), f"snapshot baseline missing: {path} (run with PTC_UPDATE_SNAPSHOTS=1)"
    # Normalize line endings so the baseline is comparable regardless of whether
    # git checked it out with CRLF (autocrlf on Windows) or LF (CI / Linux).
    expected = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    assert content == expected, f"prompt snapshot mismatch for {name!r}"


@pytest.fixture
def loader() -> PromptLoader:
    return PromptLoader(session_start_time=FIXED_TIME)


# ---------------------------------------------------------------------------
# System / sub-agent prompt snapshots
# ---------------------------------------------------------------------------


class TestSystemPromptSnapshots:
    def test_system_prompt_default(self, loader: PromptLoader) -> None:
        _assert_snapshot("system_prompt_default", loader.get_system_prompt())

    def test_system_prompt_with_storage(self, loader: PromptLoader) -> None:
        out = loader.get_system_prompt(storage_enabled=True)
        _assert_snapshot("system_prompt_with_storage", out)
        # The storage branch must actually inject the image_upload component.
        assert "Image Auto-Upload" in out


class TestSubagentPromptSnapshots:
    def test_general_subagent_default(self, loader: PromptLoader) -> None:
        _assert_snapshot("general_subagent_default", loader.get_subagent_prompt("general"))

    def test_general_subagent_with_storage(self, loader: PromptLoader) -> None:
        out = loader.get_subagent_prompt("general", storage_enabled=True)
        _assert_snapshot("general_subagent_with_storage", out)
        assert "Image Auto-Upload" in out

    def test_researcher_subagent_default(self, loader: PromptLoader) -> None:
        _assert_snapshot("researcher_subagent_default", loader.get_subagent_prompt("researcher"))


# ---------------------------------------------------------------------------
# Shared component snapshots (conditional branches exercised)
# ---------------------------------------------------------------------------


class TestComponentSnapshots:
    def test_tool_discovery_default(self, loader: PromptLoader) -> None:
        out = loader.get_component(
            "tool_discovery",
            tool_summary="tavily: web search",
            server_name="tavily",
            tool_name="tavily_search",
        )
        _assert_snapshot("component_tool_discovery_default", out)

    def test_tool_discovery_no_examples(self, loader: PromptLoader) -> None:
        out = loader.get_component(
            "tool_discovery",
            tool_summary="",
            include_examples=False,
            include_anti_patterns=False,
        )
        _assert_snapshot("component_tool_discovery_minimal", out)
        assert "Quick Reference" not in out
        assert "Anti-Patterns" not in out

    def test_image_upload_brief(self, loader: PromptLoader) -> None:
        _assert_snapshot("component_image_upload_brief", loader.get_component("image_upload"))

    def test_image_upload_detailed(self, loader: PromptLoader) -> None:
        out = loader.get_component("image_upload", detailed=True)
        _assert_snapshot("component_image_upload_detailed", out)
        assert "Creating Visualizations" in out

    def test_output_guidelines_default(self, loader: PromptLoader) -> None:
        _assert_snapshot(
            "component_output_guidelines_default", loader.get_component("output_guidelines")
        )

    def test_output_guidelines_for_task_workflow(self, loader: PromptLoader) -> None:
        out = loader.get_component("output_guidelines", for_task_workflow=True)
        _assert_snapshot("component_output_guidelines_task", out)
        assert "Data Persistence" not in out

    def test_subagent_coordination(self, loader: PromptLoader) -> None:
        out = loader.get_component(
            "subagent_coordination",
            subagent_summary="- **general-purpose**: general tasks",
        )
        _assert_snapshot("component_subagent_coordination", out)

    def test_workspace_paths(self, loader: PromptLoader) -> None:
        _assert_snapshot("component_workspace_paths", loader.get_component("workspace_paths"))

    def test_citation_rules(self, loader: PromptLoader) -> None:
        _assert_snapshot("component_citation_rules", loader.get_component("citation_rules"))

    def test_data_processing(self, loader: PromptLoader) -> None:
        _assert_snapshot("component_data_processing", loader.get_component("data_processing"))


# ---------------------------------------------------------------------------
# Determinism + metadata consistency (catches template/config drift)
# ---------------------------------------------------------------------------


class TestRenderDeterminism:
    def test_fixed_time_is_deterministic(self) -> None:
        a = PromptLoader(session_start_time=FIXED_TIME)
        b = PromptLoader(session_start_time=FIXED_TIME)
        assert a.get_system_prompt() == b.get_system_prompt()

    def test_different_time_changes_output(self) -> None:
        a = PromptLoader(session_start_time=FIXED_TIME)
        b = PromptLoader(session_start_time=datetime(2024, 5, 5, 0, 0, 0, tzinfo=UTC))
        assert a.get_system_prompt() != b.get_system_prompt()
        assert "2024-05-05" in b.get_system_prompt()

    def test_date_override_via_kwargs(self, loader: PromptLoader) -> None:
        out = loader.get_system_prompt(date="2024-01-01")
        assert "2024-01-01" in out


class TestPromptMetadataConsistency:
    def test_system_components_match_disk(self, loader: PromptLoader) -> None:
        config_path = loader.templates_dir.parent / "config" / "prompts.yaml"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        documented = set(cfg["system"]["components"])
        on_disk = {
            p.name.removesuffix(".md.j2")
            for p in (loader.templates_dir / "components").glob("*.j2")
        }
        assert documented == on_disk

    def test_component_metadata_complete(self, loader: PromptLoader) -> None:
        config_path = loader.templates_dir.parent / "config" / "prompts.yaml"
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        documented = set(cfg.get("components", {}).keys())
        on_disk = {
            p.name.removesuffix(".md.j2")
            for p in (loader.templates_dir / "components").glob("*.j2")
        }
        assert documented == on_disk

    def test_system_template_includes_exist(self, loader: PromptLoader) -> None:
        system_path = loader.templates_dir / "system.md.j2"
        includes = re.findall(
            r"{%\s*include\s+['\"]([^'\"]+)['\"]", system_path.read_text(encoding="utf-8")
        )
        for inc in includes:
            assert (loader.templates_dir / inc).exists(), f"missing include target: {inc}"


# ---------------------------------------------------------------------------
# Singleton loader behaviour
# ---------------------------------------------------------------------------


class TestLoaderSingleton:
    def test_get_loader_returns_same_instance(self) -> None:
        reset_loader()
        first = get_loader(session_start_time=FIXED_TIME)
        second = get_loader(session_start_time=datetime(1999, 1, 1, tzinfo=UTC))
        assert first is second

    def test_init_loader_creates_new_instance(self) -> None:
        reset_loader()
        first = init_loader(session_start_time=FIXED_TIME)
        second = init_loader(session_start_time=FIXED_TIME)
        assert first is not second

    def test_reset_loader_clears(self) -> None:
        init_loader(session_start_time=FIXED_TIME)
        reset_loader()
        # A fresh get_loader should build a new instance with the supplied time.
        new = get_loader(session_start_time=FIXED_TIME)
        assert isinstance(new, PromptLoader)
