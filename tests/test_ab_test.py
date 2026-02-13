"""Tests for the A/B testing simulation module."""

import pytest

from src.models.ab_test import ABTestSimulator


class MockModel:
    """Mock recommendation model for A/B testing."""

    def __init__(self, items: list[tuple[int, float]]) -> None:
        self._items = items

    def recommend(
        self,
        user_id: int,
        user_ratings: list[tuple[int, float]],
        n: int = 10,
    ) -> list[tuple[int, float]]:
        return self._items[:n]


@pytest.fixture
def simulator() -> ABTestSimulator:
    """Create an A/B test simulator with two mock models."""
    model_a = MockModel([(1, 5.0), (2, 4.0), (3, 3.0)])
    model_b = MockModel([(4, 5.0), (5, 4.0), (6, 3.0)])
    return ABTestSimulator(
        models={"model_a": model_a, "model_b": model_b},
        click_threshold=4.0,
    )


class TestABTestSimulator:
    """Tests for the ABTestSimulator class."""

    def test_simulate_session(self, simulator: ABTestSimulator) -> None:
        """Simulating a session produces per-variant results."""
        results = simulator.simulate_user_session(
            user_id=1,
            user_train_ratings=[(10, 3.0)],
            user_test_ratings=[(1, 5.0), (2, 3.0), (4, 5.0)],
            n_recommendations=3,
        )
        assert "model_a" in results
        assert "model_b" in results
        assert "clicks" in results["model_a"]
        assert "ndcg" in results["model_a"]

    def test_ctr_calculation(self, simulator: ABTestSimulator) -> None:
        """CTR is computed correctly from clicks and impressions."""
        simulator.simulate_user_session(
            user_id=1,
            user_train_ratings=[],
            user_test_ratings=[(1, 5.0), (2, 5.0)],
            n_recommendations=3,
        )
        results = simulator.get_results()
        for r in results:
            if r.variant == "model_a":
                # Model A recommends items 1, 2, 3
                # Test has item 1=5.0 (click), item 2=5.0 (click)
                assert r.ctr > 0

    def test_confidence_interval(self, simulator: ABTestSimulator) -> None:
        """Confidence intervals are within [0, 1]."""
        simulator.simulate_user_session(
            user_id=1,
            user_train_ratings=[],
            user_test_ratings=[(1, 5.0)],
            n_recommendations=3,
        )
        results = simulator.get_results()
        for r in results:
            assert 0 <= r.confidence_interval[0] <= 1
            assert 0 <= r.confidence_interval[1] <= 1

    def test_run_simulation(self, simulator: ABTestSimulator) -> None:
        """Full simulation runs without errors."""
        train_data = {1: [(10, 3.0)], 2: [(20, 4.0)]}
        test_data = {1: [(1, 5.0)], 2: [(2, 4.0)]}
        simulator.run_simulation(train_data, test_data)
        results = simulator.get_results()
        assert len(results) == 2

    def test_statistical_comparison(self, simulator: ABTestSimulator) -> None:
        """Statistical comparison returns valid p-value."""
        train = {i: [(10, 3.0)] for i in range(1, 11)}
        test = {i: [(1, 5.0), (4, 5.0)] for i in range(1, 11)}
        simulator.run_simulation(train, test)

        comparison = simulator.statistical_comparison("model_a", "model_b")
        assert "t_statistic" in comparison
        assert "p_value" in comparison
        assert 0 <= comparison["p_value"] <= 1

    def test_statistical_comparison_insufficient_data(self) -> None:
        """Comparison with insufficient data returns safe defaults."""
        sim = ABTestSimulator(
            models={"a": MockModel([]), "b": MockModel([])},
        )
        comparison = sim.statistical_comparison("a", "b")
        assert comparison["winner"] == "insufficient_data"

    def test_ndcg_perfect(self, simulator: ABTestSimulator) -> None:
        """NDCG is non-negative."""
        simulator.simulate_user_session(
            user_id=1,
            user_train_ratings=[],
            user_test_ratings=[(1, 5.0), (2, 4.0), (3, 3.0)],
            n_recommendations=3,
        )
        results = simulator.get_results()
        for r in results:
            assert r.ndcg >= 0

    def test_results_reproducible(self) -> None:
        """Same inputs produce the same results."""
        model = MockModel([(1, 5.0), (2, 4.0)])
        sim1 = ABTestSimulator(models={"m": model})
        sim2 = ABTestSimulator(models={"m": model})

        train = {1: [(10, 3.0)]}
        test = {1: [(1, 5.0)]}
        sim1.run_simulation(train, test)
        sim2.run_simulation(train, test)

        r1 = sim1.get_results()[0]
        r2 = sim2.get_results()[0]
        assert r1.ctr == r2.ctr
        assert r1.ndcg == r2.ndcg
