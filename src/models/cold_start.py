"""Cold-start handling for new users and new items.

Provides popularity-based fallback recommendations for new users
and content similarity matching for new items not yet in the catalog.
"""

import numpy as np
import pandas as pd

from src.models.content_based import ContentBasedFilter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ColdStartHandler:
    """Handles recommendations for cold-start users and items.

    New users with zero or few ratings receive popularity-based
    recommendations. New items are matched against existing catalog
    items using content similarity.

    Attributes:
        content_model: A fitted ContentBasedFilter instance.
        popularity_scores: Precomputed item popularity scores.
        threshold: Rating count below which a user is considered cold.
    """

    def __init__(
        self,
        content_model: ContentBasedFilter,
        popularity_scores: dict[int, float] | None = None,
        threshold: int = 5,
    ) -> None:
        """Initialize the cold-start handler.

        Args:
            content_model: A fitted content-based filtering model.
            popularity_scores: Precomputed item popularity scores.
            threshold: Minimum ratings for a user to be non-cold.
        """
        self.content_model = content_model
        self.popularity_scores = popularity_scores or {}
        self.threshold = threshold

    @staticmethod
    def compute_popularity_scores(
        ratings_df: pd.DataFrame,
        method: str = "count_weighted",
    ) -> dict[int, float]:
        """Compute item popularity from ratings data.

        Supports simple count-based and Bayesian average-weighted methods.

        Args:
            ratings_df: DataFrame with item_id and rating columns.
            method: Scoring method - "count" or "count_weighted".

        Returns:
            Dictionary mapping item_id to normalized popularity score.

        Raises:
            ValueError: If an unknown method is specified.
        """
        if method == "count":
            scores = ratings_df.groupby("item_id").size().to_dict()
        elif method == "count_weighted":
            stats = ratings_df.groupby("item_id")["rating"].agg(["count", "mean"])
            c = stats["count"].mean()
            m = stats["mean"].mean()
            scores = (
                (stats["count"] * stats["mean"] + c * m) / (stats["count"] + c)
            ).to_dict()
        else:
            raise ValueError(f"Unknown popularity method: {method}")

        max_score = max(scores.values()) if scores else 1.0
        normalized = {k: v / max_score for k, v in scores.items()}

        logger.info(
            "Computed popularity scores for %d items (method=%s)",
            len(normalized),
            method,
        )
        return normalized

    def recommend_new_user(
        self,
        n: int = 10,
        exclude_items: set | None = None,
    ) -> list[tuple[int, float]]:
        """Generate popularity-based recommendations for new users.

        Args:
            n: Number of recommendations to return.
            exclude_items: Set of item IDs to exclude.

        Returns:
            List of (item_id, score) tuples sorted by popularity.
        """
        exclude = exclude_items or set()
        candidates = [
            (item, score)
            for item, score in self.popularity_scores.items()
            if item not in exclude
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def recommend_for_new_item(
        self,
        item_features: np.ndarray,
        n: int = 10,
    ) -> list[tuple[int, float]]:
        """Find similar existing items for a new item.

        Args:
            item_features: Feature vector for the new item.
            n: Number of similar items to return.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        similarities = np.dot(
            self.content_model.item_features,
            item_features.flatten(),
        )
        top_indices = np.argsort(similarities)[::-1][:n]
        return [
            (int(self.content_model.item_ids[i]), float(similarities[i]))
            for i in top_indices
        ]

    def is_cold_user(self, num_ratings: int) -> bool:
        """Check if a user is a cold-start user.

        Args:
            num_ratings: Number of ratings the user has.

        Returns:
            True if the user has fewer ratings than the threshold.
        """
        return num_ratings < self.threshold

    def is_cold_item(self, item_id: int) -> bool:
        """Check if an item is new (not in the content model's catalog).

        Args:
            item_id: The item identifier.

        Returns:
            True if the item is not in the content model's index.
        """
        if self.content_model.item_id_to_idx is None:
            return True
        return item_id not in self.content_model.item_id_to_idx
