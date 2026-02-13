"""Content-based filtering using TF-IDF features and cosine similarity.

Implements item-item similarity computation and user profile construction
from rated items to generate personalized content-based recommendations.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContentBasedFilter:
    """Content-based recommender using item feature similarity.

    Uses TF-IDF-weighted genre features and cosine similarity to find
    similar items and build user preference profiles.

    Attributes:
        item_features: Feature matrix (items x features).
        item_ids: Array of item identifiers.
        similarity_matrix: Precomputed item-item cosine similarity matrix.
        item_id_to_idx: Mapping from item ID to matrix row index.
    """

    def __init__(self) -> None:
        """Initialize the content-based filter."""
        self.item_features: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.similarity_matrix: np.ndarray | None = None
        self.item_id_to_idx: dict[int, int] | None = None

    def fit(
        self, item_features: np.ndarray, item_ids: np.ndarray
    ) -> "ContentBasedFilter":
        """Compute cosine similarity matrix between all items.

        Args:
            item_features: Feature matrix of shape (n_items, n_features).
            item_ids: Array of item identifiers matching feature rows.

        Returns:
            Self, for method chaining.
        """
        self.item_features = np.asarray(
            item_features.toarray()
            if hasattr(item_features, "toarray")
            else item_features
        )
        self.item_ids = np.asarray(item_ids)
        self.item_id_to_idx = {int(id_): idx for idx, id_ in enumerate(self.item_ids)}
        self.similarity_matrix = cosine_similarity(self.item_features)
        logger.info(
            "Fitted content-based model: %d items, %d features",
            len(self.item_ids),
            self.item_features.shape[1],
        )
        return self

    def get_similar_items(self, item_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Find the n most similar items to a given item.

        Args:
            item_id: The reference item identifier.
            n: Number of similar items to return.

        Returns:
            List of (item_id, similarity_score) tuples sorted by similarity.
        """
        if self.item_id_to_idx is None or item_id not in self.item_id_to_idx:
            return []

        idx = self.item_id_to_idx[item_id]
        similarities = self.similarity_matrix[idx]
        similar_indices = np.argsort(similarities)[::-1][1 : n + 1]

        return [
            (int(self.item_ids[i]), float(similarities[i])) for i in similar_indices
        ]

    def build_user_profile(
        self,
        user_ratings: list[tuple[int, float]],
        rating_threshold: float = 3.5,
    ) -> np.ndarray:
        """Construct a user preference vector from highly-rated items.

        Builds a weighted average of item feature vectors, where weights
        are the user's ratings for items above the threshold.

        Args:
            user_ratings: List of (item_id, rating) tuples.
            rating_threshold: Minimum rating to include in the profile.

        Returns:
            User profile vector of shape (n_features,).
        """
        weighted_features = []
        weights = []

        for item_id, rating in user_ratings:
            if (
                rating >= rating_threshold
                and self.item_id_to_idx is not None
                and item_id in self.item_id_to_idx
            ):
                idx = self.item_id_to_idx[item_id]
                weighted_features.append(self.item_features[idx] * rating)
                weights.append(rating)

        if not weighted_features:
            return np.zeros(self.item_features.shape[1])

        profile = np.sum(weighted_features, axis=0) / sum(weights)
        return np.asarray(profile).flatten()

    def recommend(
        self,
        user_profile: np.ndarray,
        n: int = 10,
        exclude_items: set | None = None,
    ) -> list[tuple[int, float]]:
        """Recommend items similar to the user's preference profile.

        Args:
            user_profile: User preference vector of shape (n_features,).
            n: Number of recommendations to return.
            exclude_items: Set of item IDs to exclude from recommendations.

        Returns:
            List of (item_id, score) tuples sorted by similarity.
        """
        if exclude_items is None:
            exclude_items = set()

        user_profile_2d = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile_2d, self.item_features)[0]

        item_scores = [
            (int(self.item_ids[i]), float(similarities[i]))
            for i in range(len(self.item_ids))
            if int(self.item_ids[i]) not in exclude_items
        ]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n]

    def save(self, path: str) -> None:
        """Serialize the model to disk.

        Args:
            path: File path for the pickle output.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Content-based model saved to %s", path)

    @staticmethod
    def load(path: str) -> "ContentBasedFilter":
        """Load a serialized model from disk.

        Args:
            path: Path to the pickle file.

        Returns:
            The deserialized ContentBasedFilter instance.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        logger.info("Content-based model loaded from %s", path)
        return model
