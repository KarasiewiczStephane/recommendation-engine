"""Tests for the cold-start handling module."""

import pandas as pd
import pytest

from src.data.preprocessor import extract_genre_features
from src.models.cold_start import ColdStartHandler
from src.models.content_based import ContentBasedFilter


@pytest.fixture
def content_model(sample_movies: pd.DataFrame) -> ContentBasedFilter:
    """Create a fitted content model for cold-start testing."""
    tfidf_matrix, _ = extract_genre_features(sample_movies)
    model = ContentBasedFilter()
    model.fit(tfidf_matrix.toarray(), sample_movies["item_id"].values)
    return model


@pytest.fixture
def popularity_scores() -> dict[int, float]:
    """Create sample popularity scores."""
    return {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}


@pytest.fixture
def cold_handler(
    content_model: ContentBasedFilter,
    popularity_scores: dict[int, float],
) -> ColdStartHandler:
    """Create a ColdStartHandler instance."""
    return ColdStartHandler(
        content_model=content_model,
        popularity_scores=popularity_scores,
        threshold=5,
    )


class TestComputePopularityScores:
    """Tests for popularity score computation."""

    def test_count_method(self) -> None:
        """Count method returns rating counts normalized."""
        df = pd.DataFrame(
            {
                "item_id": [1, 1, 1, 2, 2, 3],
                "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 3.0],
            }
        )
        scores = ColdStartHandler.compute_popularity_scores(df, method="count")
        assert scores[1] == 1.0  # Most rated
        assert scores[3] < scores[1]

    def test_count_weighted_method(self) -> None:
        """Weighted method produces Bayesian average scores."""
        df = pd.DataFrame(
            {
                "item_id": [1, 1, 1, 2, 2, 3],
                "rating": [5.0, 5.0, 5.0, 1.0, 1.0, 5.0],
            }
        )
        scores = ColdStartHandler.compute_popularity_scores(df, method="count_weighted")
        assert len(scores) == 3
        assert all(0 <= s <= 1.0 for s in scores.values())

    def test_unknown_method_raises(self) -> None:
        """Unknown method raises ValueError."""
        df = pd.DataFrame({"item_id": [1], "rating": [5.0]})
        with pytest.raises(ValueError, match="Unknown popularity method"):
            ColdStartHandler.compute_popularity_scores(df, method="invalid")

    def test_scores_are_normalized(self) -> None:
        """All scores are normalized to [0, 1] with max=1."""
        df = pd.DataFrame(
            {
                "item_id": [1, 1, 2, 3, 3, 3],
                "rating": [4.0, 5.0, 3.0, 4.0, 5.0, 4.0],
            }
        )
        scores = ColdStartHandler.compute_popularity_scores(df)
        assert max(scores.values()) == 1.0
        assert all(0 <= s <= 1.0 for s in scores.values())


class TestColdStartHandler:
    """Tests for the ColdStartHandler class."""

    def test_new_user_gets_popular_items(self, cold_handler: ColdStartHandler) -> None:
        """New users receive popularity-based recommendations."""
        recs = cold_handler.recommend_new_user(n=3)
        assert len(recs) == 3
        # Top item should be most popular
        assert recs[0][0] == 1

    def test_new_user_excludes_items(self, cold_handler: ColdStartHandler) -> None:
        """Excluded items don't appear in new user recommendations."""
        recs = cold_handler.recommend_new_user(n=5, exclude_items={1, 2})
        rec_ids = {item_id for item_id, _ in recs}
        assert 1 not in rec_ids
        assert 2 not in rec_ids

    def test_new_item_similarity(self, cold_handler: ColdStartHandler) -> None:
        """New item finds similar existing items."""
        new_features = cold_handler.content_model.item_features[0]
        recs = cold_handler.recommend_for_new_item(new_features, n=3)
        assert len(recs) == 3

    def test_is_cold_user_boundary(self, cold_handler: ColdStartHandler) -> None:
        """Cold user detection works at threshold boundary."""
        assert cold_handler.is_cold_user(0)
        assert cold_handler.is_cold_user(4)
        assert not cold_handler.is_cold_user(5)
        assert not cold_handler.is_cold_user(10)

    def test_is_cold_item(self, cold_handler: ColdStartHandler) -> None:
        """Cold item detection identifies items not in catalog."""
        assert not cold_handler.is_cold_item(1)
        assert cold_handler.is_cold_item(9999)

    def test_popularity_sorted_descending(self, cold_handler: ColdStartHandler) -> None:
        """Popularity recommendations are sorted highest to lowest."""
        recs = cold_handler.recommend_new_user(n=5)
        scores = [score for _, score in recs]
        assert scores == sorted(scores, reverse=True)
