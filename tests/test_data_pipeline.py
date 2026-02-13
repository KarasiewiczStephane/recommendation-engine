"""Tests for the data download and preprocessing pipeline."""

import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.downloader import download_movielens
from src.data.preprocessor import (
    GENRE_COLUMNS,
    load_movies,
    load_ratings,
    validate_ratings,
)

SAMPLE_DATA_PATH = Path("data/sample/ml-100k")


class TestDownloader:
    """Tests for the dataset download functionality."""

    def test_download_skips_if_exists(self, tmp_dir: Path) -> None:
        """Download is skipped when the dataset already exists."""
        ml_dir = tmp_dir / "ml-100k"
        ml_dir.mkdir()
        (ml_dir / "u.data").write_text("1\t1\t5\t1000\n")

        result = download_movielens(str(tmp_dir))
        assert result == ml_dir

    def test_download_with_mock(self, tmp_dir: Path) -> None:
        """Download and extraction work with mocked network call."""
        # Create a fake zip file
        zip_path = tmp_dir / "source" / "ml-100k.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        inner_dir = tmp_dir / "temp_inner" / "ml-100k"
        inner_dir.mkdir(parents=True)
        (inner_dir / "u.data").write_text("1\t1\t5\t1000\n")

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(inner_dir / "u.data", "ml-100k/u.data")

        dest = tmp_dir / "dest"

        def fake_urlretrieve(url: str, filename: str) -> None:
            import shutil

            shutil.copy(str(zip_path), filename)

        with patch("src.data.downloader.urllib.request.urlretrieve", fake_urlretrieve):
            result = download_movielens(str(dest))

        assert result.exists()
        assert (result / "u.data").exists()


class TestLoadRatings:
    """Tests for loading and parsing ratings data."""

    def test_load_ratings_from_sample(self) -> None:
        """Sample ratings file loads with correct columns."""
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")
        ratings = load_ratings(SAMPLE_DATA_PATH)
        assert list(ratings.columns) == ["user_id", "item_id", "rating", "timestamp"]
        assert len(ratings) > 0

    def test_load_ratings_missing_file(self, tmp_dir: Path) -> None:
        """Loading from non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_ratings(tmp_dir / "nonexistent")

    def test_ratings_data_types(self) -> None:
        """Ratings columns have expected numeric types."""
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")
        ratings = load_ratings(SAMPLE_DATA_PATH)
        assert ratings["rating"].dtype in ("int64", "float64")
        assert ratings["user_id"].dtype == "int64"


class TestLoadMovies:
    """Tests for loading movie metadata."""

    def test_load_movies_from_sample(self) -> None:
        """Sample movies file loads with genre columns."""
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")
        movies = load_movies(SAMPLE_DATA_PATH)
        assert "title" in movies.columns
        assert "item_id" in movies.columns
        for genre in GENRE_COLUMNS:
            assert genre in movies.columns

    def test_genre_values_are_binary(self) -> None:
        """Genre columns contain only 0 and 1 values."""
        if not SAMPLE_DATA_PATH.exists():
            pytest.skip("Sample data not available")
        movies = load_movies(SAMPLE_DATA_PATH)
        for genre in GENRE_COLUMNS:
            assert set(movies[genre].unique()).issubset({0, 1})

    def test_load_movies_missing_file(self, tmp_dir: Path) -> None:
        """Loading from non-existent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_movies(tmp_dir / "nonexistent")


class TestValidateRatings:
    """Tests for the ratings validation function."""

    def test_valid_ratings(self) -> None:
        """Valid ratings pass all checks."""
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 3],
                "rating": [1.0, 5.0, 3.0, 4.0],
                "timestamp": [100, 200, 300, 400],
            }
        )
        result = validate_ratings(df)
        assert result["total_ratings"] == 4
        assert result["unique_users"] == 2
        assert result["unique_items"] == 3
        assert result["rating_range_valid"] is True

    def test_invalid_ratings_raises(self) -> None:
        """Ratings outside 1-5 range raise ValueError."""
        df = pd.DataFrame(
            {
                "user_id": [1, 1],
                "item_id": [1, 2],
                "rating": [1.0, 6.0],
                "timestamp": [100, 200],
            }
        )
        with pytest.raises(ValueError, match="outside the valid range"):
            validate_ratings(df)

    def test_sparsity_calculation(self) -> None:
        """Sparsity is computed correctly for a known matrix."""
        # 2 users, 3 items, 4 ratings => sparsity = 1 - 4/(2*3) = 1/3
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 3],
                "rating": [5.0, 4.0, 3.0, 2.0],
                "timestamp": [100, 200, 300, 400],
            }
        )
        result = validate_ratings(df)
        expected_sparsity = 1 - 4 / (2 * 3)
        assert abs(result["sparsity"] - expected_sparsity) < 1e-10

    def test_missing_values_detected(self) -> None:
        """Missing values are counted per column."""
        df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": [1, None],
                "rating": [5.0, 3.0],
                "timestamp": [100, 200],
            }
        )
        # This will still fail validation since None in item_id doesn't affect rating range
        result = validate_ratings(df)
        assert result["missing_values"]["item_id"] == 1
