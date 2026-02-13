"""SQLite database utilities for persistent storage.

Manages the SQLite database connection and schema for storing user ratings
and item metadata used by the recommendation engine.
"""

import sqlite3
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_connection(db_path: str = "data/app.db") -> sqlite3.Connection:
    """Create or open a SQLite database connection.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        An open SQLite connection with row factory enabled.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = "data/app.db") -> None:
    """Initialize the database schema.

    Creates the ratings and items tables if they do not already exist.

    Args:
        db_path: Path to the SQLite database file.
    """
    conn = get_connection(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                item_id INTEGER NOT NULL,
                rating REAL NOT NULL CHECK(rating >= 1.0 AND rating <= 5.0),
                timestamp INTEGER NOT NULL,
                UNIQUE(user_id, item_id)
            );

            CREATE INDEX IF NOT EXISTS idx_ratings_user
                ON ratings(user_id);

            CREATE INDEX IF NOT EXISTS idx_ratings_item
                ON ratings(item_id);

            CREATE TABLE IF NOT EXISTS items (
                item_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                release_date TEXT,
                genres TEXT
            );
            """
        )
        conn.commit()
        logger.info("Database schema initialized at %s", db_path)
    finally:
        conn.close()


def insert_rating(
    db_path: str, user_id: int, item_id: int, rating: float, timestamp: int
) -> None:
    """Insert or update a user rating.

    Args:
        db_path: Path to the SQLite database file.
        user_id: The user identifier.
        item_id: The item identifier.
        rating: The rating value (1.0-5.0).
        timestamp: Unix timestamp of the rating.
    """
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO ratings (user_id, item_id, rating, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, item_id)
            DO UPDATE SET rating = excluded.rating, timestamp = excluded.timestamp
            """,
            (user_id, item_id, rating, timestamp),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_ratings(db_path: str, user_id: int) -> list[tuple[int, float]]:
    """Retrieve all ratings for a given user.

    Args:
        db_path: Path to the SQLite database file.
        user_id: The user identifier.

    Returns:
        A list of (item_id, rating) tuples for the user.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT item_id, rating FROM ratings WHERE user_id = ?",
            (user_id,),
        )
        return [(row["item_id"], row["rating"]) for row in cursor.fetchall()]
    finally:
        conn.close()
