"""Data loading, validation, and preprocessing for MovieLens 100K.

Provides functions to load ratings and movie metadata from the raw dataset
files, validate data integrity, and compute dataset statistics.
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def load_ratings(data_path: Path) -> pd.DataFrame:
    """Load and parse the ratings data file.

    Args:
        data_path: Path to the ml-100k directory.

    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp.

    Raises:
        FileNotFoundError: If the ratings file does not exist.
    """
    ratings_file = data_path / "u.data"
    if not ratings_file.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

    ratings = pd.read_csv(
        ratings_file,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    logger.info("Loaded %d ratings", len(ratings))
    return ratings


def load_movies(data_path: Path) -> pd.DataFrame:
    """Load movie metadata including genre information.

    Args:
        data_path: Path to the ml-100k directory.

    Returns:
        DataFrame with item_id, title, release_date, and genre columns.

    Raises:
        FileNotFoundError: If the items file does not exist.
    """
    items_file = data_path / "u.item"
    if not items_file.exists():
        raise FileNotFoundError(f"Items file not found: {items_file}")

    column_names = [
        "item_id",
        "title",
        "release_date",
        "video_release",
        "imdb_url",
    ] + GENRE_COLUMNS

    movies = pd.read_csv(
        items_file,
        sep="|",
        encoding="latin-1",
        names=column_names,
    )
    logger.info("Loaded %d movies", len(movies))
    return movies


def validate_ratings(df: pd.DataFrame) -> dict:
    """Validate rating data integrity and compute statistics.

    Checks that ratings are within the valid range, counts unique users
    and items, and computes the sparsity of the user-item matrix.

    Args:
        df: Ratings DataFrame with user_id, item_id, and rating columns.

    Returns:
        Dictionary containing validation results and statistics.

    Raises:
        ValueError: If ratings contain values outside the 1-5 range.
    """
    rating_range_valid = bool(df["rating"].between(1, 5).all())

    checks = {
        "total_ratings": len(df),
        "unique_users": int(df["user_id"].nunique()),
        "unique_items": int(df["item_id"].nunique()),
        "rating_range_valid": rating_range_valid,
        "sparsity": 1 - len(df) / (df["user_id"].nunique() * df["item_id"].nunique()),
        "missing_values": df.isnull().sum().to_dict(),
    }

    if not rating_range_valid:
        raise ValueError("Ratings contain values outside the valid range [1, 5]")

    logger.info(
        "Validation complete: %d ratings, %d users, %d items, sparsity=%.4f",
        checks["total_ratings"],
        checks["unique_users"],
        checks["unique_items"],
        checks["sparsity"],
    )
    return checks
