"""Tests for the hybrid recommendation model."""

import random

import pandas as pd
import pytest

from src.data.preprocessor import extract_genre_features
from src.models.collaborative import CollaborativeFilter
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender


@pytest.fixture
def hybrid_models(sample_movies: pd.DataFrame) -> dict:
    """Create trained collaborative and content-based models for hybrid testing."""
    random.seed(42)
    rows = []
    for user_id in range(1, 11):
        for item_id in random.sample(range(1, 6), min(4, 5)):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": float(random.randint(1, 5)),
                }
            )
    ratings_df = pd.DataFrame(rows)

    collab = CollaborativeFilter(algorithm="SVD")
    collab.fit(ratings_df, n_factors=5, n_epochs=10)

    tfidf_matrix, _ = extract_genre_features(sample_movies)
    content = ContentBasedFilter()
    content.fit(tfidf_matrix.toarray(), sample_movies["item_id"].values)

    return {"collaborative": collab, "content_based": content, "ratings_df": ratings_df}


@pytest.fixture
def hybrid_recommender(hybrid_models: dict) -> HybridRecommender:
    """Create a configured hybrid recommender."""
    return HybridRecommender(
        collaborative=hybrid_models["collaborative"],
        content_based=hybrid_models["content_based"],
        alpha=0.7,
        cold_start_threshold=5,
    )


class TestHybridRecommender:
    """Tests for the HybridRecommender class."""

    def test_weighted_returns_results(
        self, hybrid_recommender: HybridRecommender
    ) -> None:
        """Weighted strategy produces recommendations."""
        recs = hybrid_recommender.recommend(
            user_id=1,
            user_ratings=[(1, 5.0), (2, 4.0)],
            n=3,
            strategy="weighted",
        )
        assert len(recs) > 0

    def test_switching_cold_user_uses_content(
        self, hybrid_recommender: HybridRecommender
    ) -> None:
        """Cold users (< threshold ratings) get content-based recommendations."""
        # User with only 2 ratings (below threshold of 5)
        recs = hybrid_recommender.recommend(
            user_id=999,
            user_ratings=[(1, 5.0), (2, 4.0)],
            n=3,
            strategy="switching",
        )
        assert len(recs) > 0

    def test_switching_warm_user_uses_collaborative(
        self, hybrid_recommender: HybridRecommender
    ) -> None:
        """Warm users (>= threshold ratings) get collaborative recommendations."""
        warm_ratings = [(i, 4.0) for i in range(1, 7)]
        recs = hybrid_recommender.recommend(
            user_id=1,
            user_ratings=warm_ratings,
            n=3,
            strategy="switching",
        )
        assert len(recs) > 0

    def test_unknown_strategy_raises(
        self, hybrid_recommender: HybridRecommender
    ) -> None:
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            hybrid_recommender.recommend(
                user_id=1,
                user_ratings=[(1, 5.0)],
                strategy="invalid",
            )

    def test_alpha_one_is_collaborative_only(self, hybrid_models: dict) -> None:
        """Alpha=1.0 produces collaborative-only scores."""
        recommender = HybridRecommender(
            collaborative=hybrid_models["collaborative"],
            content_based=hybrid_models["content_based"],
            alpha=1.0,
        )
        recs = recommender.recommend_weighted(
            user_id=1,
            user_ratings=[(1, 5.0)],
            n=3,
        )
        # With alpha=1.0, scores should be purely from collaborative
        for _, score in recs:
            assert 0.0 <= score <= 1.0

    def test_alpha_zero_is_content_only(self, hybrid_models: dict) -> None:
        """Alpha=0.0 produces content-based-only scores."""
        recommender = HybridRecommender(
            collaborative=hybrid_models["collaborative"],
            content_based=hybrid_models["content_based"],
            alpha=0.0,
        )
        recs = recommender.recommend_weighted(
            user_id=1,
            user_ratings=[(1, 5.0)],
            n=3,
        )
        for _, score in recs:
            assert 0.0 <= score <= 1.0

    def test_normalize_scores_edge_case(self) -> None:
        """Normalization handles all-same scores."""
        result = HybridRecommender._normalize_scores({1: 3.0, 2: 3.0, 3: 3.0})
        for v in result.values():
            assert v == 0.5

    def test_normalize_scores_empty(self) -> None:
        """Normalization handles empty dict."""
        result = HybridRecommender._normalize_scores({})
        assert result == {}

    def test_exclude_items_in_weighted(
        self, hybrid_recommender: HybridRecommender
    ) -> None:
        """Excluded items are not in weighted recommendations."""
        recs = hybrid_recommender.recommend(
            user_id=1,
            user_ratings=[(1, 5.0), (2, 4.0)],
            n=5,
            strategy="weighted",
        )
        rec_ids = {item_id for item_id, _ in recs}
        assert 1 not in rec_ids
        assert 2 not in rec_ids

    def test_cold_user_detection(self, hybrid_recommender: HybridRecommender) -> None:
        """Cold user check works at boundary."""
        assert hybrid_recommender._is_cold_user([(i, 4.0) for i in range(4)])
        assert not hybrid_recommender._is_cold_user([(i, 4.0) for i in range(5)])
