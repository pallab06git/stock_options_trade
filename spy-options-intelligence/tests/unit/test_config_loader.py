# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Unit tests for ConfigLoader."""

import os
import pytest
import yaml

from src.utils.config_loader import ConfigLoader, ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path, data):
    """Write a dict as YAML to the given path."""
    with open(path, "w") as f:
        yaml.dump(data, f)


def _create_config_dir(tmp_path, settings=None, sources=None, sinks=None, retry=None):
    """
    Create a minimal config directory with all 4 YAML files.
    Returns the config dir path.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    _write_yaml(config_dir / "settings.yaml", settings or {
        "streaming": {"market_hours": {"timezone": "America/New_York"}},
        "monitoring": {"performance": {"commit_latency_seconds": 300}},
    })
    _write_yaml(config_dir / "sources.yaml", sources or {
        "polygon": {
            "api_key": "${POLYGON_API_KEY}",
            "base_url": "https://api.polygon.io",
            "spy": {"ticker": "SPY"},
        }
    })
    _write_yaml(config_dir / "sinks.yaml", sinks or {
        "sinks": {"parquet": {"enabled": True, "compression": "snappy"}},
    })
    _write_yaml(config_dir / "retry_policy.yaml", retry or {
        "retry": {"default": {"max_attempts": 3}},
    })

    return config_dir


def _create_env_file(tmp_path, content="POLYGON_API_KEY=test_key_abc123\n"):
    """Write a .env file and return its path."""
    env_path = tmp_path / ".env"
    env_path.write_text(content)
    return env_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadValidConfig:
    """Happy-path loading with all 4 files + .env."""

    def test_load_valid_config(self, tmp_path, monkeypatch):
        config_dir = _create_config_dir(tmp_path)
        env_path = _create_env_file(tmp_path)

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(env_path))
        config = loader.load()

        # Keys from all 4 files present
        assert "polygon" in config
        assert "sinks" in config
        assert "retry" in config
        assert "streaming" in config
        assert "monitoring" in config

        # API key resolved
        assert config["polygon"]["api_key"] == "test_key_abc123"


class TestEnvVarSubstitution:
    """${VAR} patterns replaced correctly."""

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_CUSTOM_VAR", "custom_value")
        monkeypatch.setenv("POLYGON_API_KEY", "key_xyz")

        config_dir = _create_config_dir(
            tmp_path,
            sources={
                "polygon": {
                    "api_key": "${POLYGON_API_KEY}",
                    "custom": "${MY_CUSTOM_VAR}",
                }
            },
        )

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "nonexistent.env"))
        config = loader.load()

        assert config["polygon"]["api_key"] == "key_xyz"
        assert config["polygon"]["custom"] == "custom_value"

    def test_env_var_inline_substitution(self, tmp_path, monkeypatch):
        """${VAR} embedded in a larger string is replaced."""
        monkeypatch.setenv("POLYGON_API_KEY", "pk_abc")

        config_dir = _create_config_dir(
            tmp_path,
            sources={
                "polygon": {
                    "api_key": "${POLYGON_API_KEY}",
                    "url": "https://api.polygon.io?key=${POLYGON_API_KEY}",
                }
            },
        )

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))
        config = loader.load()

        assert config["polygon"]["url"] == "https://api.polygon.io?key=pk_abc"


class TestMissingEnvVar:
    """${UNDEFINED} raises ConfigError."""

    def test_missing_env_var_raises(self, tmp_path, monkeypatch):
        # Ensure the variable is NOT set
        monkeypatch.delenv("UNDEFINED_VAR", raising=False)
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)

        config_dir = _create_config_dir(
            tmp_path,
            sources={"polygon": {"api_key": "${UNDEFINED_VAR}"}},
        )

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))

        with pytest.raises(ConfigError, match="UNDEFINED_VAR"):
            loader.load()


class TestMissingYaml:
    """Missing YAML file raises FileNotFoundError."""

    def test_missing_yaml_raises(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Only create settings.yaml — missing the others
        _write_yaml(config_dir / "settings.yaml", {"streaming": {}})

        loader = ConfigLoader(config_dir=str(config_dir))

        with pytest.raises(FileNotFoundError, match="sources.yaml"):
            loader.load()

    def test_missing_config_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            ConfigLoader(config_dir=str(tmp_path / "nonexistent"))


class TestInvalidYaml:
    """Bad YAML syntax raises an error."""

    def test_invalid_yaml_raises(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Write invalid YAML to settings.yaml
        (config_dir / "settings.yaml").write_text(":\n  invalid: [unterminated\n")
        _write_yaml(config_dir / "sources.yaml", {"polygon": {"api_key": "x"}})
        _write_yaml(config_dir / "sinks.yaml", {})
        _write_yaml(config_dir / "retry_policy.yaml", {})

        loader = ConfigLoader(config_dir=str(config_dir))

        with pytest.raises(yaml.YAMLError):
            loader.load()


class TestDeepMerge:
    """Nested dicts merge correctly; lists overwrite."""

    def test_deep_merge(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "key_123")

        # settings.yaml has nested monitoring config
        settings = {
            "monitoring": {
                "performance": {"commit_latency_seconds": 300, "error_rate_percent": 1.0}
            },
            "streaming": {"enabled": True},
        }
        # sources.yaml adds polygon key and also has a monitoring override
        sources = {
            "polygon": {"api_key": "${POLYGON_API_KEY}"},
            "monitoring": {"performance": {"commit_latency_seconds": 600}},
        }

        config_dir = _create_config_dir(tmp_path, settings=settings, sources=sources)

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))
        config = loader.load()

        # Deep merge: commit_latency overridden, error_rate preserved
        assert config["monitoring"]["performance"]["commit_latency_seconds"] == 600
        assert config["monitoring"]["performance"]["error_rate_percent"] == 1.0
        # streaming key from settings preserved
        assert config["streaming"]["enabled"] is True

    def test_list_overwrite(self, tmp_path, monkeypatch):
        """Lists from override replace base lists entirely."""
        monkeypatch.setenv("POLYGON_API_KEY", "key_123")

        settings = {"items": [1, 2, 3]}
        sources = {"polygon": {"api_key": "${POLYGON_API_KEY}"}, "items": [4, 5]}

        config_dir = _create_config_dir(tmp_path, settings=settings, sources=sources)
        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))
        config = loader.load()

        assert config["items"] == [4, 5]


class TestReload:
    """reload() returns updated config after file change."""

    def test_reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "key_123")
        config_dir = _create_config_dir(tmp_path)

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))
        config1 = loader.load()

        # Modify settings.yaml
        _write_yaml(config_dir / "settings.yaml", {
            "streaming": {"market_hours": {"timezone": "UTC"}},
            "monitoring": {"performance": {"commit_latency_seconds": 999}},
        })

        config2 = loader.reload()

        assert config2["streaming"]["market_hours"]["timezone"] == "UTC"
        assert config2["monitoring"]["performance"]["commit_latency_seconds"] == 999


class TestDefaultValuesAccessible:
    """Nested .get() patterns work as expected by downstream modules."""

    def test_default_values_accessible(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "key_123")
        config_dir = _create_config_dir(tmp_path)

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))
        config = loader.load()

        # Downstream code accesses config like this:
        timezone = config.get("streaming", {}).get("market_hours", {}).get("timezone", "America/New_York")
        assert timezone == "America/New_York"

        # Missing keys return defaults via .get()
        missing = config.get("nonexistent", {}).get("deep", {}).get("key", "fallback")
        assert missing == "fallback"

        # Parquet sink settings
        compression = config.get("sinks", {}).get("parquet", {}).get("compression", "snappy")
        assert compression == "snappy"


class TestApiKeyValidation:
    """Missing POLYGON_API_KEY raises ConfigError."""

    def test_api_key_validation_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)

        config_dir = _create_config_dir(
            tmp_path,
            sources={"polygon": {"api_key": ""}},
        )

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))

        with pytest.raises(ConfigError, match="POLYGON_API_KEY"):
            loader.load()

    def test_api_key_validation_unresolved(self, tmp_path, monkeypatch):
        """An unresolved ${POLYGON_API_KEY} should fail validation if env var missing."""
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)

        config_dir = _create_config_dir(
            tmp_path,
            sources={"polygon": {"api_key": "${POLYGON_API_KEY}"}},
        )

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))

        with pytest.raises(ConfigError):
            loader.load()


class TestSecurityNoKeyInRepr:
    """Config repr doesn't leak secret values."""

    def test_security_no_key_in_repr(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "super_secret_key_12345")
        config_dir = _create_config_dir(tmp_path)

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))
        loader.load()

        repr_str = repr(loader)
        assert "super_secret_key_12345" not in repr_str

    def test_config_dict_not_in_repr(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "secret_abc")
        config_dir = _create_config_dir(tmp_path)

        loader = ConfigLoader(config_dir=str(config_dir), env_file=str(tmp_path / "none.env"))
        loader.load()

        repr_str = repr(loader)
        assert "secret_abc" not in repr_str
        assert "api_key" not in repr_str
