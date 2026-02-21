# © 2026 Pallab Basu Roy. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, or commercial use is strictly prohibited.

"""Configuration loader with YAML parsing, env var substitution, and validation."""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised for configuration validation failures."""
    pass


class ConfigLoader:
    """
    Load and merge YAML configuration files with environment variable substitution.

    Merge order: settings.yaml (base) <- sources.yaml <- sinks.yaml <- retry_policy.yaml
                 <- any additional *.yaml files found in config_dir (alphabetical order)
    Environment variables referenced as ${VAR} in YAML are resolved from os.environ.
    """

    # Regex to match ${VAR_NAME} patterns
    _ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

    # Required YAML files loaded first in this exact order (base → override)
    _YAML_FILES = ["settings.yaml", "sources.yaml", "sinks.yaml", "retry_policy.yaml"]

    # Keys that must be present and non-empty after loading
    _REQUIRED_KEYS = ["POLYGON_API_KEY"]

    def __init__(self, config_dir: str = "config", env_file: str = ".env"):
        """
        Initialize the config loader.

        Args:
            config_dir: Path to the directory containing YAML config files.
            env_file: Path to the .env file for secret injection.

        Raises:
            FileNotFoundError: If config_dir does not exist.
        """
        self.config_dir = Path(config_dir)
        self.env_file = Path(env_file)
        self.config: dict = {}

        if not self.config_dir.is_dir():
            raise FileNotFoundError(
                f"Configuration directory not found: {self.config_dir}"
            )

    def load(self) -> dict:
        """
        Load .env, parse all YAML files, substitute env vars, merge, and validate.

        Returns:
            Merged configuration dictionary.

        Raises:
            FileNotFoundError: If a required YAML file is missing.
            yaml.YAMLError: If a YAML file has invalid syntax.
            ConfigError: If env var substitution or validation fails.
        """
        # Load .env into os.environ (no-op if file missing)
        if self.env_file.exists():
            load_dotenv(self.env_file, override=True)

        # Parse and merge required YAML files in defined order
        merged: dict = {}
        for filename in self._YAML_FILES:
            filepath = self.config_dir / filename
            data = self._load_yaml(filepath)
            merged = self._deep_merge(merged, data)

        # Auto-discover and merge any additional *.yaml files (e.g. ml_settings.yaml)
        _base_set = set(self._YAML_FILES)
        extra_files = sorted(
            f.name for f in self.config_dir.glob("*.yaml") if f.name not in _base_set
        )
        for filename in extra_files:
            data = self._load_yaml(self.config_dir / filename)
            merged = self._deep_merge(merged, data)

        # Substitute ${VAR} references
        merged = self._substitute_env_vars(merged)

        # Validate required keys
        self._validate(merged)

        self.config = merged
        return self.config

    def reload(self) -> dict:
        """
        Re-read all config files and return updated configuration.

        Returns:
            Updated merged configuration dictionary.
        """
        return self.load()

    def _load_yaml(self, path: Path) -> dict:
        """
        Parse a single YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed dictionary (empty dict if file is empty).

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the YAML syntax is invalid.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # yaml.safe_load returns None for empty files
        return data if isinstance(data, dict) else {}

    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively replace ${VAR} patterns with values from os.environ.

        Args:
            obj: Configuration object (dict, list, or scalar).

        Returns:
            Object with all ${VAR} patterns resolved.

        Raises:
            ConfigError: If a referenced environment variable is not set.
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._replace_env_vars_in_string(obj)
        return obj

    def _replace_env_vars_in_string(self, value: str) -> str:
        """
        Replace all ${VAR} occurrences in a string with env var values.

        Args:
            value: String potentially containing ${VAR} patterns.

        Returns:
            String with env vars resolved.

        Raises:
            ConfigError: If a referenced variable is not set.
        """
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                raise ConfigError(f"Environment variable '{var_name}' not set")
            return env_value

        return self._ENV_VAR_PATTERN.sub(replacer, value)

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """
        Deep merge override into base. Nested dicts are merged recursively;
        all other types (including lists) are overwritten by override.

        Args:
            base: Base dictionary.
            override: Dictionary to merge into base.

        Returns:
            New merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _validate(self, config: dict) -> None:
        """
        Validate that required configuration keys are present and non-empty.

        Checks for POLYGON_API_KEY in the nested polygon.api_key path.

        Args:
            config: Merged configuration dictionary.

        Raises:
            ConfigError: If a required key is missing or empty.
        """
        # Check polygon.api_key
        api_key = config.get("polygon", {}).get("api_key", "")
        if not api_key or api_key.startswith("${"):
            raise ConfigError(
                "POLYGON_API_KEY required but not set. "
                "Set it in .env or as an environment variable."
            )

    def __repr__(self) -> str:
        """Safe repr that never exposes secret values."""
        return f"ConfigLoader(config_dir='{self.config_dir}', loaded={bool(self.config)})"
