"""Tests for SQLite database utilities."""

from pathlib import Path

import pytest

from src.utils.database import get_connection, get_user_ratings, init_db, insert_rating


class TestDatabase:
    """Tests for database connection and CRUD operations."""

    def test_init_creates_tables(self, tmp_dir: Path) -> None:
        """init_db creates ratings and items tables."""
        db_path = str(tmp_dir / "test.db")
        init_db(db_path)
        conn = get_connection(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}
        conn.close()
        assert "ratings" in tables
        assert "items" in tables

    def test_insert_and_retrieve_rating(self, tmp_dir: Path) -> None:
        """Inserting a rating can be retrieved by user."""
        db_path = str(tmp_dir / "test.db")
        init_db(db_path)
        insert_rating(db_path, user_id=1, item_id=10, rating=4.5, timestamp=1000)
        ratings = get_user_ratings(db_path, user_id=1)
        assert len(ratings) == 1
        assert ratings[0] == (10, 4.5)

    def test_upsert_rating(self, tmp_dir: Path) -> None:
        """Inserting a duplicate user-item pair updates the rating."""
        db_path = str(tmp_dir / "test.db")
        init_db(db_path)
        insert_rating(db_path, user_id=1, item_id=10, rating=3.0, timestamp=1000)
        insert_rating(db_path, user_id=1, item_id=10, rating=5.0, timestamp=2000)
        ratings = get_user_ratings(db_path, user_id=1)
        assert len(ratings) == 1
        assert ratings[0] == (10, 5.0)

    def test_get_user_ratings_empty(self, tmp_dir: Path) -> None:
        """Querying a user with no ratings returns an empty list."""
        db_path = str(tmp_dir / "test.db")
        init_db(db_path)
        ratings = get_user_ratings(db_path, user_id=999)
        assert ratings == []

    def test_rating_constraint(self, tmp_dir: Path) -> None:
        """Ratings outside 1.0-5.0 range are rejected."""
        db_path = str(tmp_dir / "test.db")
        init_db(db_path)
        with pytest.raises(Exception):
            insert_rating(db_path, user_id=1, item_id=1, rating=6.0, timestamp=1000)

    def test_creates_parent_directory(self, tmp_dir: Path) -> None:
        """get_connection creates parent directories if needed."""
        db_path = str(tmp_dir / "subdir" / "nested" / "test.db")
        conn = get_connection(db_path)
        conn.close()
        assert Path(db_path).exists()
