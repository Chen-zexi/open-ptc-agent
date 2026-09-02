"""Unit tests for ptc_agent.config.loaders and ptc_agent.config.utils.

These tests cover the file-based configuration loading pipeline:

- Path utilities (get_default_config_dir, find_project_root, get_config_search_paths,
  find_config_file, ensure_config_dir)
- load_from_dict: section validation and LLM resolution (string name / inline definition
  / catalog lookup) and the error branches
- generate_config_template: FileExistsError / overwrite behaviour
- _load_llm_catalog: missing-file / bad-JSON / missing-'llms'-key branches
- The shared factory functions in config.utils (create_*_config) and YAML loading

The tests are fully isolated: they never rely on the developer's real ~/.ptc-agent or the
repo-root config.yaml. Search paths, env overrides and Path.home are exercised explicitly.

NOTE: these tests also lock in two real bugs that were fixed while writing them:
1. generate_config_template() produced a config.yaml whose `daytona` section was missing
   `auto_archive_interval` / `auto_delete_interval`, so the generated file failed the
   project's own load_from_dict() validation (config init -> immediate crash on boot).
2. create_filesystem_config() silently dropped the user-supplied `working_directory`,
   so filesystem.working_directory configured in config.yaml was ignored at runtime.
"""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest
import yaml

from ptc_agent.config.agent import AgentConfig, LLMDefinition
from ptc_agent.config.core import (
    DaytonaConfig,
    FilesystemConfig,
    LoggingConfig,
    MCPConfig,
    MCPServerConfig,
    SecurityConfig,
)
from ptc_agent.config.loaders import (
    CONFIG_TEMPLATE,
    ConfigContext,
    _load_llm_catalog,
    find_config_file,
    find_project_root,
    generate_config_template,
    get_config_search_paths,
    get_default_config_dir,
    load_from_dict,
)
from ptc_agent.config.utils import (
    DAYTONA_REQUIRED_FIELDS,
    FILESYSTEM_REQUIRED_FIELDS,
    LOGGING_REQUIRED_FIELDS,
    MCP_REQUIRED_FIELDS,
    SECURITY_REQUIRED_FIELDS,
    create_daytona_config,
    create_filesystem_config,
    create_logging_config,
    create_mcp_config,
    create_security_config,
    load_yaml_file,
    validate_required_sections,
    validate_section_fields,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_home(tmp_path: Path) -> Path:
    """A throwaway directory standing in for Path.home() / ~/.ptc-agent."""
    home = tmp_path / "fake_home"
    home.mkdir()
    return home


@pytest.fixture
def patch_home(fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect Path.home() to the fake home for the duration of a test."""
    monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)


def make_catalog() -> dict[str, LLMDefinition]:
    """Build a minimal but valid LLM catalog mirroring llms.json."""
    return {
        "claude-sonnet-4-5": LLMDefinition(
            model_id="claude-sonnet-4-5-20250929",
            provider="anthropic",
            sdk="langchain_anthropic.ChatAnthropic",
            api_key_env="ANTHROPIC_API_KEY",
        )
    }


def valid_config() -> dict:
    """A config.yaml structure that passes load_from_dict() validation."""
    return {
        "llm": {"name": "claude-sonnet-4-5"},
        "daytona": {
            "base_url": "https://app.daytona.io/api",
            "auto_stop_interval": 3600,
            "auto_archive_interval": 86400,
            "auto_delete_interval": 604800,
            "python_version": "3.12",
        },
        "security": {
            "max_execution_time": 300,
            "max_code_length": 10000,
            "max_file_size": 10485760,
            "enable_code_validation": True,
            "allowed_imports": ["os", "sys"],
            "blocked_patterns": ["eval("],
        },
        "mcp": {"servers": [], "tool_discovery_enabled": True},
        "logging": {"level": "INFO", "file": "logs/ptc.log"},
        "filesystem": {
            "allowed_directories": ["/home/daytona"],
            "working_directory": "/home/daytona",
        },
    }


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


class TestPathUtilities:
    def test_get_default_config_dir(self, patch_home: None, fake_home: Path) -> None:
        assert get_default_config_dir() == fake_home / ".ptc-agent"

    def test_find_project_root_finds_git(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        nested = repo / "a" / "b" / "c"
        nested.mkdir(parents=True)
        assert find_project_root(nested) == repo

    def test_find_project_root_walks_up_to_repo(self) -> None:
        # Anchor on the real checkout (via __file__) so the test does not depend on
        # pytest's tmp_path location, which may live outside the repo under the
        # system temp root (e.g. when --basetemp is not used).
        here = Path(__file__)
        repo_root = find_project_root(here)
        assert repo_root is not None
        assert (repo_root / ".git").exists()
        # A deep, real path inside the repo must still resolve to the same root.
        nested = repo_root / "libs" / "ptc-agent" / "tests" / "unit_tests"
        assert find_project_root(nested) == repo_root

    def test_search_paths_sdk_default(
        self, patch_home: None, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        paths = get_config_search_paths(context=ConfigContext.SDK)
        assert paths[0] == tmp_path
        assert paths[-1] == fake_home / ".ptc-agent"

    def test_search_paths_sdk_includes_project_root(
        self, patch_home: None, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        sub = repo / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        paths = get_config_search_paths(context=ConfigContext.SDK)
        assert paths[0] == sub
        assert repo in paths
        assert paths[-1] == fake_home / ".ptc-agent"

    def test_search_paths_cli_home_first(
        self, patch_home: None, fake_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        paths = get_config_search_paths(context=ConfigContext.CLI)
        assert paths[0] == fake_home / ".ptc-agent"
        assert paths[1] == tmp_path

    def test_find_config_file_env_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env_file = tmp_path / "my_config.yaml"
        env_file.write_text("x: 1")
        monkeypatch.setenv("PTC_CONFIG_FILE", str(env_file))
        result = find_config_file(
            "config.yaml", search_paths=[tmp_path / "nope"], env_var="PTC_CONFIG_FILE"
        )
        assert result == env_file

    def test_find_config_file_env_override_missing_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PTC_CONFIG_FILE", str(tmp_path / "does_not_exist.yaml"))
        found = tmp_path / "config.yaml"
        found.write_text("x: 1")
        result = find_config_file(
            "config.yaml", search_paths=[tmp_path], env_var="PTC_CONFIG_FILE"
        )
        assert result == found

    def test_find_config_file_search_order(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("PTC_CONFIG_FILE", raising=False)
        found = tmp_path / "config.yaml"
        found.write_text("x: 1")
        result = find_config_file("config.yaml", search_paths=[tmp_path])
        assert result == found

    def test_find_config_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("PTC_CONFIG_FILE", raising=False)
        assert find_config_file("config.yaml", search_paths=[tmp_path]) is None

    def test_ensure_config_dir_creates(
        self, patch_home: None, fake_home: Path
    ) -> None:
        result = get_default_config_dir()
        result.mkdir(parents=True, exist_ok=True)
        assert result.exists()
        assert result == fake_home / ".ptc-agent"


# ---------------------------------------------------------------------------
# load_from_dict: validation + LLM resolution
# ---------------------------------------------------------------------------


class TestLoadFromDict:
    def test_valid_with_catalog(self) -> None:
        cfg = load_from_dict(valid_config(), make_catalog())
        assert isinstance(cfg, AgentConfig)
        assert cfg.llm.name == "claude-sonnet-4-5"
        assert cfg.llm_definition is not None
        assert cfg.llm_definition.model_id == "claude-sonnet-4-5-20250929"
        assert cfg.daytona.base_url == "https://app.daytona.io/api"
        assert cfg.security.max_execution_time == 300
        assert cfg.filesystem.working_directory == "/home/daytona"
        assert cfg.mcp.tool_discovery_enabled is True
        assert cfg.config_file_dir is None

    def test_string_llm_resolves_from_catalog(self) -> None:
        data = valid_config()
        data["llm"] = "claude-sonnet-4-5"
        cfg = load_from_dict(data, make_catalog())
        assert cfg.llm.name == "claude-sonnet-4-5"
        assert cfg.llm_definition is not None

    def test_inline_llm_without_catalog(self) -> None:
        data = valid_config()
        data["llm"] = {
            "name": "custom",
            "model_id": "m1",
            "provider": "anthropic",
            "sdk": "s",
            "api_key_env": "K",
        }
        cfg = load_from_dict(data)
        assert cfg.llm.name == "custom"
        assert cfg.llm_definition is not None
        assert cfg.llm_definition.model_id == "m1"

    def test_inline_definition_overrides_catalog(self) -> None:
        data = valid_config()
        data["llm"] = {
            "name": "custom",
            "model_id": "inline-model",
            "provider": "anthropic",
            "sdk": "s",
            "api_key_env": "K",
        }
        cfg = load_from_dict(data, make_catalog())
        assert cfg.llm_definition.model_id == "inline-model"

    def test_unknown_llm_name_raises(self) -> None:
        data = valid_config()
        data["llm"] = "nonexistent-model"
        with pytest.raises(ValueError, match="not found in llm_catalog"):
            load_from_dict(data, make_catalog())

    def test_no_catalog_no_inline_raises(self) -> None:
        data = valid_config()
        data["llm"] = "claude-sonnet-4-5"
        with pytest.raises(ValueError, match="cannot be resolved"):
            load_from_dict(data)

    def test_missing_required_section_raises(self) -> None:
        data = valid_config()
        del data["filesystem"]
        with pytest.raises(ValueError, match="Missing required sections"):
            load_from_dict(data, make_catalog())

    def test_invalid_llm_type_raises(self) -> None:
        data = valid_config()
        data["llm"] = 123
        with pytest.raises(ValueError, match="must be either a string"):
            load_from_dict(data, make_catalog())


# ---------------------------------------------------------------------------
# generate_config_template (regression for the daytona-fields bug)
# ---------------------------------------------------------------------------


class TestGenerateConfigTemplate:
    def test_creates_config_and_llms(self, tmp_path: Path) -> None:
        created = generate_config_template(tmp_path)
        assert (tmp_path / "config.yaml").exists()
        assert (tmp_path / "llms.json").exists()
        assert created["config.yaml"] == tmp_path / "config.yaml"
        assert created["llms.json"] == tmp_path / "llms.json"

    def test_file_exists_raises(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("existing")
        with pytest.raises(FileExistsError):
            generate_config_template(tmp_path)

    def test_overwrite_replaces(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("existing")
        generate_config_template(tmp_path, overwrite=True)
        assert (tmp_path / "config.yaml").read_text() == CONFIG_TEMPLATE

    def test_no_llms_skips_llms_json(self, tmp_path: Path) -> None:
        created = generate_config_template(tmp_path, include_llms=False)
        assert "llms.json" not in created
        assert not (tmp_path / "llms.json").exists()

    def test_generated_template_passes_own_validation(self) -> None:
        """Regression: the generated config.yaml must load through load_from_dict."""
        data = yaml.safe_load(CONFIG_TEMPLATE)
        cfg = load_from_dict(data, make_catalog())
        assert isinstance(cfg, AgentConfig)
        assert cfg.daytona.auto_archive_interval == 86400
        assert cfg.daytona.auto_delete_interval == 604800


# ---------------------------------------------------------------------------
# _load_llm_catalog
# ---------------------------------------------------------------------------


class TestLoadLLMCatalog:
    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await _load_llm_catalog(tmp_path / "nope.json")

    async def test_bad_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{ not valid json ,,")
        with pytest.raises(ValueError, match="Failed to parse llms"):
            await _load_llm_catalog(bad)

    async def test_missing_llms_key_raises(self, tmp_path: Path) -> None:
        nokey = tmp_path / "nokey.json"
        nokey.write_text('{"foo": {}}')
        with pytest.raises(ValueError, match="must have 'llms' key"):
            await _load_llm_catalog(nokey)

    async def test_valid_catalog(self, tmp_path: Path) -> None:
        ok = tmp_path / "ok.json"
        ok.write_text(
            '{"llms": {"claude-sonnet-4-5": {"model_id": "x", "provider": '
            '"anthropic", "sdk": "s", "api_key_env": "K"}}}'
        )
        catalog = await _load_llm_catalog(ok)
        assert "claude-sonnet-4-5" in catalog
        assert isinstance(catalog["claude-sonnet-4-5"], LLMDefinition)


# ---------------------------------------------------------------------------
# config.utils: YAML loading + validation + factories
# ---------------------------------------------------------------------------


class TestLoadYamlFile:
    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await load_yaml_file(tmp_path / "nope.yaml")

    async def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("a: [unclosed\n")
        with pytest.raises(ValueError, match="Failed to parse config"):
            await load_yaml_file(f)

    async def test_empty_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.yaml"
        f.write_text("\n   \n")
        with pytest.raises(ValueError, match="is empty"):
            await load_yaml_file(f)

    async def test_valid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "ok.yaml"
        f.write_text("daytona:\n  base_url: x\n")
        data = await load_yaml_file(f)
        assert data["daytona"]["base_url"] == "x"


class TestValidationHelpers:
    def test_validate_required_sections_ok(self) -> None:
        validate_required_sections({"a": 1, "b": 2}, ["a", "b"])

    def test_validate_required_sections_missing(self) -> None:
        with pytest.raises(ValueError, match="Missing required sections"):
            validate_required_sections({"a": 1}, ["a", "b", "c"])

    def test_validate_section_fields_missing(self) -> None:
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_section_fields({"x": 1}, ["x", "y"], "sec")


class TestFactoryFunctions:
    def test_create_daytona_config(self) -> None:
        cfg = create_daytona_config(
            {
                "base_url": "https://app.daytona.io/api",
                "auto_stop_interval": 3600,
                "auto_archive_interval": 86400,
                "auto_delete_interval": 604800,
                "python_version": "3.12",
            }
        )
        assert isinstance(cfg, DaytonaConfig)
        assert cfg.auto_stop_interval == 3600
        assert cfg.python_version == "3.12"

    def test_create_daytona_config_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing required fields"):
            create_daytona_config({"base_url": "x"})

    def test_create_security_config(self) -> None:
        cfg = create_security_config(
            {
                "max_execution_time": 300,
                "max_code_length": 10000,
                "max_file_size": 1,
                "enable_code_validation": True,
                "allowed_imports": ["os"],
                "blocked_patterns": ["eval("],
            }
        )
        assert isinstance(cfg, SecurityConfig)
        assert cfg.max_execution_time == 300

    def test_create_mcp_config_builds_servers(self) -> None:
        cfg = create_mcp_config(
            {
                "servers": [
                    {"name": "tavily", "command": "npx", "args": ["-y", "tavily-mcp"]}
                ],
                "tool_discovery_enabled": True,
            }
        )
        assert isinstance(cfg, MCPConfig)
        assert isinstance(cfg.servers[0], MCPServerConfig)
        assert cfg.servers[0].name == "tavily"
        assert cfg.tool_discovery_enabled is True

    def test_create_logging_config(self) -> None:
        cfg = create_logging_config({"level": "DEBUG", "file": "logs/x.log"})
        assert isinstance(cfg, LoggingConfig)
        assert cfg.level == "DEBUG"

    def test_required_field_constants_complete(self) -> None:
        # Guard against the daytona-fields regression: both intervals must be required.
        assert "auto_archive_interval" in DAYTONA_REQUIRED_FIELDS
        assert "auto_delete_interval" in DAYTONA_REQUIRED_FIELDS
        assert "allowed_directories" in FILESYSTEM_REQUIRED_FIELDS
        assert "level" in LOGGING_REQUIRED_FIELDS
        assert "servers" in MCP_REQUIRED_FIELDS
        assert "max_execution_time" in SECURITY_REQUIRED_FIELDS

    def test_filesystem_working_directory_read_from_config(self) -> None:
        """Regression: user-supplied working_directory must be honoured."""
        cfg = create_filesystem_config(
            {"allowed_directories": ["/home/daytona"], "working_directory": "/srv/app"}
        )
        assert isinstance(cfg, FilesystemConfig)
        assert cfg.working_directory == "/srv/app"

    def test_filesystem_working_directory_default(self) -> None:
        cfg = create_filesystem_config({"allowed_directories": ["/home/daytona"]})
        assert cfg.working_directory == "/home/daytona"
