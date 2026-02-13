"""Configuration management for the recommendation engine.

Loads application settings from environment variables and YAML config files,
providing a unified interface for accessing configuration values throughout
the application.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        redis_host: Hostname for Redis connection.
        redis_port: Port for Redis connection.
        config_path: Path to the YAML configuration file.
    """

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    config_path: str = Field(default="configs/config.yaml")

    model_config = {"env_file": ".env", "extra": "ignore"}


def _resolve_env_vars(value: str) -> str:
    """Resolve environment variable references in config values.

    Supports the syntax ${VAR_NAME:default_value}.

    Args:
        value: The string potentially containing env var references.

    Returns:
        The resolved string with env var values substituted.
    """
    if not isinstance(value, str) or "${" not in value:
        return value

    import re

    pattern = r"\$\{(\w+):([^}]*)\}"
    match = re.match(pattern, value)
    if match:
        env_var, default = match.groups()
        return os.environ.get(env_var, default)
    return value


def _resolve_config(config: dict) -> dict:
    """Recursively resolve environment variables in config dict.

    Args:
        config: Configuration dictionary with potential env var references.

    Returns:
        Configuration dictionary with all env vars resolved.
    """
    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_config(value)
        elif isinstance(value, str):
            resolved[key] = _resolve_env_vars(value)
        else:
            resolved[key] = value
    return resolved


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load and parse the YAML configuration file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file contains invalid YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    return _resolve_config(raw_config)


settings = Settings()
config = load_config(settings.config_path)
