"""Data loading, validation, and preprocessing for MovieLens 100K.

Provides functions to load ratings and movie metadata from the raw dataset
files, validate data integrity, and compute dataset statistics.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

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


def extract_genre_features(
    movies: pd.DataFrame,
) -> tuple[spmatrix, TfidfVectorizer]:
    """Convert genre columns to TF-IDF weighted features.

    Creates a text representation of each movie's genres and applies
    TF-IDF vectorization to produce weighted feature vectors.

    Args:
        movies: DataFrame with binary genre columns.

    Returns:
        A tuple of (tfidf_matrix, fitted_vectorizer).
    """
    genre_cols = [c for c in movies.columns if c in GENRE_COLUMNS]
    genre_strings = movies[genre_cols].apply(
        lambda row: " ".join([col for col, val in row.items() if val == 1]),
        axis=1,
    )
    # Handle empty genre strings
    genre_strings = genre_strings.replace("", "unknown")

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(genre_strings)

    logger.info(
        "Extracted TF-IDF features: %d items x %d features",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
    )
    return tfidf_matrix, vectorizer


def save_processed_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    features: Any,
    output_path: str,
) -> None:
    """Save processed datasets to disk in efficient formats.

    Args:
        train: Training ratings DataFrame.
        test: Test ratings DataFrame.
        movies: Movie metadata DataFrame.
        features: Sparse or dense feature matrix.
        output_path: Directory to save all processed files.
    """
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    train.to_parquet(output / "train.parquet")
    test.to_parquet(output / "test.parquet")
    movies.to_parquet(output / "movies.parquet")

    if hasattr(features, "toarray"):
        np.save(output / "item_features.npy", features.toarray())
    else:
        np.save(output / "item_features.npy", features)

    logger.info("Saved processed data to %s", output)
