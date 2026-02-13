"""Tests for the collaborative filtering module."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.collaborative import CollaborativeFilter


@pytest.fixture
def large_sample_ratings() -> pd.DataFrame:
    """Create a larger ratings DataFrame needed for Surprise library."""
    import random

    random.seed(42)
    rows = []
    for user_id in range(1, 21):
        for item_id in random.sample(range(1, 51), 10):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": float(random.randint(1, 5)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_model(large_sample_ratings: pd.DataFrame) -> CollaborativeFilter:
    """Create a trained SVD model on sample data."""
    model = CollaborativeFilter(algorithm="SVD")
    model.fit(large_sample_ratings, n_factors=10, n_epochs=10)
    return model


class TestCollaborativeFilter:
    """Tests for the CollaborativeFilter class."""

    def test_fit_creates_model(self, large_sample_ratings: pd.DataFrame) -> None:
        """Training creates a non-null model."""
        cf = CollaborativeFilter(algorithm="SVD")
        cf.fit(large_sample_ratings, n_factors=10, n_epochs=10)
        assert cf.model is not None
        assert cf.trainset is not None

    def test_prediction_in_valid_range(
        self, trained_model: CollaborativeFilter
    ) -> None:
        """Predictions fall within the valid rating range."""
        pred = trained_model.predict(1, 1)
        assert 1.0 <= pred <= 5.0

    def test_recommend_returns_correct_count(
        self, trained_model: CollaborativeFilter
    ) -> None:
        """Recommendations return the requested number of items."""
        recs = trained_model.recommend(1, n=5)
        assert len(recs) == 5

    def test_recommend_excludes_rated(
        self, trained_model: CollaborativeFilter, large_sample_ratings: pd.DataFrame
    ) -> None:
        """Recommendations exclude already-rated items by default."""
        recs = trained_model.recommend(1, n=10, exclude_rated=True)
        rec_items = {item for item, _ in recs}
        rated_items = set(
            large_sample_ratings[large_sample_ratings["user_id"] == 1]["item_id"]
        )
        assert len(rec_items & rated_items) == 0

    def test_recommend_includes_rated_when_disabled(
        self, trained_model: CollaborativeFilter
    ) -> None:
        """Recommendations include rated items when exclude_rated=False."""
        recs = trained_model.recommend(1, n=50, exclude_rated=False)
        assert len(recs) > 0

    def test_recommendations_are_sorted(
        self, trained_model: CollaborativeFilter
    ) -> None:
        """Recommendations are sorted by predicted score descending."""
        recs = trained_model.recommend(1, n=10)
        scores = [score for _, score in recs]
        assert scores == sorted(scores, reverse=True)

    def test_save_and_load(
        self, trained_model: CollaborativeFilter, tmp_dir: Path
    ) -> None:
        """Model can be serialized and deserialized."""
        model_path = str(tmp_dir / "model.pkl")
        trained_model.save(model_path)

        loaded = CollaborativeFilter.load(model_path)
        assert loaded.algorithm == trained_model.algorithm
        pred_original = trained_model.predict(1, 1)
        pred_loaded = loaded.predict(1, 1)
        assert abs(pred_original - pred_loaded) < 1e-6

    def test_predict_without_training_raises(self) -> None:
        """Predicting before training raises RuntimeError."""
        cf = CollaborativeFilter()
        with pytest.raises(RuntimeError, match="not been trained"):
            cf.predict(1, 1)

    def test_recommend_without_training_raises(self) -> None:
        """Recommending before training raises RuntimeError."""
        cf = CollaborativeFilter()
        with pytest.raises(RuntimeError, match="not been trained"):
            cf.recommend(1)

    @pytest.mark.parametrize("algo", ["SVD", "NMF"])
    def test_algorithm_variants(
        self, algo: str, large_sample_ratings: pd.DataFrame
    ) -> None:
        """Multiple algorithm variants train and predict successfully."""
        cf = CollaborativeFilter(algorithm=algo)
        cf.fit(large_sample_ratings, n_factors=10, n_epochs=5)
        pred = cf.predict(1, 1)
        assert 1.0 <= pred <= 5.0

    def test_unknown_user_recommendation(
        self, trained_model: CollaborativeFilter
    ) -> None:
        """Recommending for an unknown user returns results gracefully."""
        recs = trained_model.recommend(9999, n=5)
        assert len(recs) == 5
