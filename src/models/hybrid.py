"""Hybrid recommendation model combining collaborative and content-based filtering.

Provides weighted average and switching strategies to leverage both
collaborative filtering and content-based approaches, with automatic
fallback for cold-start scenarios.
"""

from src.models.collaborative import CollaborativeFilter
from src.models.content_based import ContentBasedFilter
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRecommender:
    """Hybrid recommender combining collaborative and content-based models.

    Supports weighted combination and switching strategies based on
    user activity level, with configurable blending parameters.

    Attributes:
        collaborative: The collaborative filtering model.
        content_based: The content-based filtering model.
        alpha: Weight for collaborative scores (1-alpha for content-based).
        cold_start_threshold: Minimum ratings for a user to be considered warm.
    """

    def __init__(
        self,
        collaborative: CollaborativeFilter,
        content_based: ContentBasedFilter,
        alpha: float = 0.7,
        cold_start_threshold: int = 5,
    ) -> None:
        """Initialize the hybrid recommender.

        Args:
            collaborative: A trained CollaborativeFilter instance.
            content_based: A fitted ContentBasedFilter instance.
            alpha: Weight for collaborative scores (0.0 to 1.0).
            cold_start_threshold: Number of ratings below which a user is cold.
        """
        self.collaborative = collaborative
        self.content_based = content_based
        self.alpha = alpha
        self.cold_start_threshold = cold_start_threshold

    def _is_cold_user(self, user_ratings: list[tuple[int, float]]) -> bool:
        """Check if user has fewer ratings than the cold-start threshold.

        Args:
            user_ratings: List of (item_id, rating) tuples.

        Returns:
            True if the user is a cold-start user.
        """
        return len(user_ratings) < self.cold_start_threshold

    @staticmethod
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        """Normalize score dictionary values to [0, 1] range.

        Args:
            scores: Dictionary mapping item_id to raw score.

        Returns:
            Dictionary with min-max normalized scores.
        """
        if not scores:
            return {}
        min_s = min(scores.values())
        max_s = max(scores.values())
        if max_s == min_s:
            return {k: 0.5 for k in scores}
        return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}

    def recommend_weighted(
        self,
        user_id: int,
        user_ratings: list[tuple[int, float]],
        n: int = 10,
        exclude_items: set | None = None,
    ) -> list[tuple[int, float]]:
        """Generate recommendations using weighted score combination.

        Combines normalized collaborative and content-based scores using
        the configured alpha weight.

        Args:
            user_id: The user identifier.
            user_ratings: List of (item_id, rating) tuples.
            n: Number of recommendations to return.
            exclude_items: Set of item IDs to exclude.

        Returns:
            List of (item_id, combined_score) tuples sorted by score.
        """
        if exclude_items is None:
            exclude_items = {item_id for item_id, _ in user_ratings}

        # Get collaborative recommendations
        collab_recs = self.collaborative.recommend(user_id, n=n * 3)
        collab_scores = dict(collab_recs)

        # Get content-based recommendations
        user_profile = self.content_based.build_user_profile(user_ratings)
        content_recs = self.content_based.recommend(
            user_profile, n=n * 3, exclude_items=exclude_items
        )
        content_scores = dict(content_recs)

        # Normalize and combine
        collab_norm = self._normalize_scores(collab_scores)
        content_norm = self._normalize_scores(content_scores)

        all_items = set(collab_scores.keys()) | set(content_scores.keys())
        hybrid_scores = []
        for item in all_items:
            if item in exclude_items:
                continue
            c_score = collab_norm.get(item, 0)
            cb_score = content_norm.get(item, 0)
            combined = self.alpha * c_score + (1 - self.alpha) * cb_score
            hybrid_scores.append((item, combined))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:n]

    def recommend_switching(
        self,
        user_id: int,
        user_ratings: list[tuple[int, float]],
        n: int = 10,
        exclude_items: set | None = None,
    ) -> list[tuple[int, float]]:
        """Switch between strategies based on user activity level.

        Cold-start users receive content-based recommendations while
        warm users get collaborative filtering recommendations.

        Args:
            user_id: The user identifier.
            user_ratings: List of (item_id, rating) tuples.
            n: Number of recommendations to return.
            exclude_items: Set of item IDs to exclude.

        Returns:
            List of (item_id, score) tuples sorted by score.
        """
        if self._is_cold_user(user_ratings):
            user_profile = self.content_based.build_user_profile(user_ratings)
            return self.content_based.recommend(
                user_profile, n=n, exclude_items=exclude_items or set()
            )
        else:
            return self.collaborative.recommend(user_id, n=n)

    def recommend(
        self,
        user_id: int,
        user_ratings: list[tuple[int, float]],
        n: int = 10,
        strategy: str = "weighted",
    ) -> list[tuple[int, float]]:
        """Main recommendation interface supporting multiple strategies.

        Args:
            user_id: The user identifier.
            user_ratings: List of (item_id, rating) tuples.
            n: Number of recommendations to return.
            strategy: One of "weighted" or "switching".

        Returns:
            List of (item_id, score) tuples sorted by score.

        Raises:
            ValueError: If an unknown strategy is specified.
        """
        exclude = {item_id for item_id, _ in user_ratings}
        if strategy == "weighted":
            return self.recommend_weighted(user_id, user_ratings, n, exclude)
        elif strategy == "switching":
            return self.recommend_switching(user_id, user_ratings, n, exclude)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
