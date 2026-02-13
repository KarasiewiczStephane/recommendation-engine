"""Shared test fixtures for the recommendation engine test suite."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_ratings() -> pd.DataFrame:
    """Create a small ratings DataFrame for testing."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "item_id": [1, 2, 3, 1, 4, 2, 3, 4, 5],
            "rating": [5.0, 3.0, 4.0, 4.0, 5.0, 2.0, 5.0, 4.0, 3.0],
            "timestamp": list(range(9)),
        }
    )


@pytest.fixture
def sample_movies() -> pd.DataFrame:
    """Create a small movies DataFrame for testing."""
    return pd.DataFrame(
        {
            "item_id": [1, 2, 3, 4, 5],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
            "Action": [1, 0, 1, 0, 1],
            "Comedy": [0, 1, 0, 1, 0],
            "Drama": [0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def tmp_dir() -> Path:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_yaml(tmp_dir: Path) -> Path:
    """Create a temporary config YAML file for testing."""
    config_path = tmp_dir / "config.yaml"
    config_path.write_text(
        """
app:
  name: test-engine
  version: 0.1.0

data:
  raw_path: data/raw/
  processed_path: data/processed/

redis:
  host: localhost
  port: 6379
  db: 0
  recommendation_ttl: 3600
  similarity_ttl: 86400

sqlite:
  path: data/test.db

api:
  host: 0.0.0.0
  port: 8000
"""
    )
    return config_path
