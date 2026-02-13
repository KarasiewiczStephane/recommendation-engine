"""Temporal train/test splitting for recommendation evaluation.

Implements a time-based splitting strategy that respects the chronological
order of user interactions, preventing data leakage from future ratings
into the training set.
"""

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def temporal_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings by taking the last portion of each user's ratings for test.

    For each user, ratings are sorted by timestamp and the most recent
    `test_ratio` fraction is assigned to the test set. This preserves
    temporal ordering and prevents future information leakage.

    Args:
        ratings: DataFrame with user_id, item_id, rating, timestamp columns.
        test_ratio: Fraction of each user's ratings to hold out for testing.

    Returns:
        A tuple of (train_df, test_df) DataFrames.

    Raises:
        ValueError: If test_ratio is not between 0 and 1.
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    ratings = ratings.sort_values(["user_id", "timestamp"]).copy()

    train_indices = []
    test_indices = []

    for _user_id, group in ratings.groupby("user_id"):
        n = len(group)
        split_idx = int(n * (1 - test_ratio))
        if split_idx == 0 and n > 1:
            split_idx = 1
        if split_idx == n and n > 1:
            split_idx = n - 1
        idx_list = group.index.tolist()
        train_indices.extend(idx_list[:split_idx])
        test_indices.extend(idx_list[split_idx:])

    train = ratings.loc[train_indices]
    test = ratings.loc[test_indices]

    logger.info(
        "Temporal split: %d train, %d test (ratio=%.2f)",
        len(train),
        len(test),
        test_ratio,
    )
    return train, test
