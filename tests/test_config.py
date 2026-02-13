"""Tests for configuration loading and environment variable handling."""

from pathlib import Path

import pytest

from src.utils.config import Settings, _resolve_env_vars, load_config


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, sample_config_yaml: Path) -> None:
        """Config loading returns a dict with expected keys."""
        config = load_config(str(sample_config_yaml))
        assert isinstance(config, dict)
        assert "app" in config
        assert config["app"]["name"] == "test-engine"

    def test_load_missing_config_raises(self, tmp_dir: Path) -> None:
        """Loading a non-existent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_dir / "nonexistent.yaml"))

    def test_load_invalid_yaml(self, tmp_dir: Path) -> None:
        """Loading invalid YAML content raises an error."""
        bad_config = tmp_dir / "bad.yaml"
        bad_config.write_text(": invalid: yaml: [")
        with pytest.raises(Exception):
            load_config(str(bad_config))

    def test_nested_values_preserved(self, sample_config_yaml: Path) -> None:
        """Nested config values are accessible."""
        config = load_config(str(sample_config_yaml))
        assert config["redis"]["port"] == 6379
        assert config["api"]["host"] == "0.0.0.0"


class TestResolveEnvVars:
    """Tests for environment variable resolution in config values."""

    def test_resolve_env_var_with_default(self) -> None:
        """Env var syntax with default returns default when var not set."""
        result = _resolve_env_vars("${NONEXISTENT_VAR:fallback}")
        assert result == "fallback"

    def test_resolve_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env var syntax returns actual value when var is set."""
        monkeypatch.setenv("TEST_CONFIG_VAR", "custom_value")
        result = _resolve_env_vars("${TEST_CONFIG_VAR:default}")
        assert result == "custom_value"

    def test_plain_string_unchanged(self) -> None:
        """Strings without env var syntax are returned unchanged."""
        assert _resolve_env_vars("plain_string") == "plain_string"

    def test_non_string_passthrough(self) -> None:
        """Non-string values are returned as-is."""
        assert _resolve_env_vars(42) == 42


class TestSettings:
    """Tests for the Settings pydantic model."""

    def test_default_values(self) -> None:
        """Settings have sensible defaults."""
        s = Settings()
        assert s.redis_host == "localhost"
        assert s.redis_port == 6379

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Environment variables override default settings."""
        monkeypatch.setenv("REDIS_HOST", "redis-server")
        monkeypatch.setenv("REDIS_PORT", "6380")
        s = Settings()
        assert s.redis_host == "redis-server"
        assert s.redis_port == 6380
