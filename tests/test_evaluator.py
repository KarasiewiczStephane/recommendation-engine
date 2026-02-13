"""Tests for the evaluation metrics module."""

import numpy as np
import pytest

from src.models.evaluator import Evaluator


@pytest.fixture
def evaluator() -> Evaluator:
    """Create an evaluator with default K values."""
    return Evaluator(k_values=[5, 10])


class TestRatingMetrics:
    """Tests for RMSE and MAE metrics."""

    def test_rmse_known_values(self) -> None:
        """RMSE computed correctly for known predictions."""
        predictions = [(3.0, 3.0), (4.0, 4.0)]
        assert Evaluator.rmse(predictions) == 0.0

    def test_rmse_nonzero(self) -> None:
        """RMSE is non-zero for imperfect predictions."""
        predictions = [(3.0, 4.0), (2.0, 4.0)]
        rmse = Evaluator.rmse(predictions)
        expected = np.sqrt((1.0 + 4.0) / 2)
        assert abs(rmse - expected) < 1e-6

    def test_rmse_empty(self) -> None:
        """RMSE of empty predictions is 0."""
        assert Evaluator.rmse([]) == 0.0

    def test_mae_known_values(self) -> None:
        """MAE computed correctly for known predictions."""
        predictions = [(3.0, 4.0), (5.0, 3.0)]
        assert Evaluator.mae(predictions) == 1.5

    def test_mae_perfect(self) -> None:
        """MAE is 0 for perfect predictions."""
        predictions = [(3.0, 3.0), (4.0, 4.0)]
        assert Evaluator.mae(predictions) == 0.0


class TestRankingMetrics:
    """Tests for Precision@K, Recall@K, and NDCG@K."""

    def test_precision_perfect(self) -> None:
        """Precision@K is 1.0 when all recommendations are relevant."""
        recommended = [1, 2, 3]
        relevant = {1, 2, 3}
        assert Evaluator.precision_at_k(recommended, relevant, k=3) == 1.0

    def test_precision_none_relevant(self) -> None:
        """Precision@K is 0.0 when no recommendations are relevant."""
        recommended = [1, 2, 3]
        relevant = {4, 5, 6}
        assert Evaluator.precision_at_k(recommended, relevant, k=3) == 0.0

    def test_precision_partial(self) -> None:
        """Precision@K with partial overlap."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5, 7}
        assert Evaluator.precision_at_k(recommended, relevant, k=5) == 3 / 5

    def test_recall_perfect(self) -> None:
        """Recall@K is 1.0 when all relevant items are in top-K."""
        recommended = [1, 2, 3, 4, 5]
        relevant = {1, 2}
        assert Evaluator.recall_at_k(recommended, relevant, k=5) == 1.0

    def test_recall_empty_relevant(self) -> None:
        """Recall@K is 0.0 when there are no relevant items."""
        recommended = [1, 2, 3]
        assert Evaluator.recall_at_k(recommended, set(), k=3) == 0.0

    def test_recall_partial(self) -> None:
        """Recall@K with partial coverage."""
        recommended = [1, 2, 3]
        relevant = {1, 4, 5, 6}
        assert Evaluator.recall_at_k(recommended, relevant, k=3) == 1 / 4

    def test_ndcg_perfect_ranking(self) -> None:
        """NDCG@K is 1.0 for a perfect ranking."""
        recommended = [1, 2, 3]
        relevant = {1: 5.0, 2: 4.0, 3: 3.0}
        assert abs(Evaluator.ndcg_at_k(recommended, relevant, k=3) - 1.0) < 1e-6

    def test_ndcg_reverse_ranking(self) -> None:
        """NDCG@K is less than 1.0 for a reversed ranking."""
        recommended = [3, 2, 1]
        relevant = {1: 5.0, 2: 4.0, 3: 3.0}
        ndcg = Evaluator.ndcg_at_k(recommended, relevant, k=3)
        assert ndcg < 1.0

    def test_ndcg_empty_relevant(self) -> None:
        """NDCG@K is 0.0 when there are no relevant items."""
        recommended = [1, 2, 3]
        assert Evaluator.ndcg_at_k(recommended, {}, k=3) == 0.0

    def test_precision_k_zero(self) -> None:
        """Precision@0 returns 0."""
        assert Evaluator.precision_at_k([1, 2], {1}, k=0) == 0.0


class TestModelComparison:
    """Tests for the model comparison functionality."""

    def test_compare_returns_all_models(self, evaluator: Evaluator) -> None:
        """compare_models returns results for each input model."""

        class MockModel:
            def recommend(self, user_id, user_ratings, n=10):
                return [(i, 4.0 - i * 0.1) for i in range(1, n + 1)]

        models = {"model_a": MockModel(), "model_b": MockModel()}
        test_data = {1: [(1, 5.0), (2, 4.0)]}
        train_data = {1: [(3, 3.0)]}

        results = evaluator.compare_models(models, test_data, train_data)
        assert "model_a" in results
        assert "model_b" in results

    def test_evaluate_ranking_metrics(self, evaluator: Evaluator) -> None:
        """evaluate_ranking returns expected metric keys."""

        class MockModel:
            def recommend(self, user_id, user_ratings, n=10):
                return [(i, 4.0) for i in range(1, n + 1)]

        test_data = {1: [(1, 5.0), (2, 3.0)]}
        train_data = {1: [(3, 4.0)]}

        metrics = evaluator.evaluate_ranking(MockModel(), test_data, train_data)
        assert "precision@5" in metrics
        assert "recall@10" in metrics
        assert "ndcg@5" in metrics
