"""Tests for the temporal train/test splitting module."""

import pandas as pd
import pytest

from src.data.splitter import temporal_split


class TestTemporalSplit:
    """Tests for the temporal_split function."""

    def test_split_preserves_all_rows(self, sample_ratings: pd.DataFrame) -> None:
        """Total rows in train + test equals original."""
        train, test = temporal_split(sample_ratings, test_ratio=0.2)
        assert len(train) + len(test) == len(sample_ratings)

    def test_split_ratio_approximate(self, sample_ratings: pd.DataFrame) -> None:
        """Test set size is approximately the requested ratio."""
        train, test = temporal_split(sample_ratings, test_ratio=0.3)
        actual_ratio = len(test) / len(sample_ratings)
        assert abs(actual_ratio - 0.3) < 0.15  # Within 15% tolerance

    def test_user_coverage(self, sample_ratings: pd.DataFrame) -> None:
        """All users with 2+ ratings appear in both train and test sets."""
        train, test = temporal_split(sample_ratings, test_ratio=0.3)
        users_with_multiple = sample_ratings.groupby("user_id").size()
        users_multi = users_with_multiple[users_with_multiple >= 2].index
        for user_id in users_multi:
            assert user_id in train["user_id"].values
            assert user_id in test["user_id"].values

    def test_temporal_ordering(self, sample_ratings: pd.DataFrame) -> None:
        """Test timestamps are always after train timestamps for each user."""
        train, test = temporal_split(sample_ratings, test_ratio=0.3)
        for user_id in train["user_id"].unique():
            if user_id in test["user_id"].values:
                max_train_ts = train[train["user_id"] == user_id]["timestamp"].max()
                min_test_ts = test[test["user_id"] == user_id]["timestamp"].min()
                assert min_test_ts >= max_train_ts

    def test_no_data_leakage(self, sample_ratings: pd.DataFrame) -> None:
        """No identical (user_id, item_id, timestamp) rows in both sets."""
        train, test = temporal_split(sample_ratings, test_ratio=0.2)
        train_keys = set(zip(train["user_id"], train["item_id"], train["timestamp"]))
        test_keys = set(zip(test["user_id"], test["item_id"], test["timestamp"]))
        assert len(train_keys & test_keys) == 0

    def test_invalid_ratio_raises(self, sample_ratings: pd.DataFrame) -> None:
        """Invalid test_ratio values raise ValueError."""
        with pytest.raises(ValueError):
            temporal_split(sample_ratings, test_ratio=0.0)
        with pytest.raises(ValueError):
            temporal_split(sample_ratings, test_ratio=1.0)

    def test_columns_preserved(self, sample_ratings: pd.DataFrame) -> None:
        """Output DataFrames have the same columns as input."""
        train, test = temporal_split(sample_ratings, test_ratio=0.2)
        assert list(train.columns) == list(sample_ratings.columns)
        assert list(test.columns) == list(sample_ratings.columns)
