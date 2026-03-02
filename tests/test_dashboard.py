"""Tests for the recommendation engine dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    generate_ab_test_data,
    generate_coverage_stats,
    generate_ranking_metrics,
    generate_rating_accuracy,
)


class TestRankingMetrics:
    def test_returns_dataframe(self) -> None:
        df = generate_ranking_metrics()
        assert isinstance(df, pd.DataFrame)

    def test_has_entries(self) -> None:
        df = generate_ranking_metrics()
        assert len(df) == 9  # 3 models x 3 K values

    def test_has_required_columns(self) -> None:
        df = generate_ranking_metrics()
        for col in ["model", "k", "precision_at_k", "recall_at_k", "ndcg_at_k"]:
            assert col in df.columns

    def test_metrics_positive(self) -> None:
        df = generate_ranking_metrics()
        assert (df["precision_at_k"] > 0).all()
        assert (df["ndcg_at_k"] > 0).all()

    def test_three_k_values(self) -> None:
        df = generate_ranking_metrics()
        assert set(df["k"].unique()) == {5, 10, 20}

    def test_reproducible(self) -> None:
        df1 = generate_ranking_metrics(seed=99)
        df2 = generate_ranking_metrics(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestRatingAccuracy:
    def test_returns_dataframe(self) -> None:
        df = generate_rating_accuracy()
        assert isinstance(df, pd.DataFrame)

    def test_has_four_models(self) -> None:
        df = generate_rating_accuracy()
        assert len(df) == 4

    def test_rmse_positive(self) -> None:
        df = generate_rating_accuracy()
        assert (df["rmse"] > 0).all()

    def test_mae_positive(self) -> None:
        df = generate_rating_accuracy()
        assert (df["mae"] > 0).all()

    def test_has_required_columns(self) -> None:
        df = generate_rating_accuracy()
        for col in ["model", "rmse", "mae"]:
            assert col in df.columns


class TestAbTestData:
    def test_returns_dataframe(self) -> None:
        df = generate_ab_test_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_entries(self) -> None:
        df = generate_ab_test_data()
        assert len(df) == 56  # 28 days x 2 variants

    def test_ctr_bounded(self) -> None:
        df = generate_ab_test_data()
        assert (df["ctr"] >= 0).all()
        assert (df["ctr"] <= 1).all()

    def test_has_both_variants(self) -> None:
        df = generate_ab_test_data()
        assert df["variant"].nunique() == 2


class TestCoverageStats:
    def test_returns_dict(self) -> None:
        coverage = generate_coverage_stats()
        assert isinstance(coverage, dict)

    def test_has_required_keys(self) -> None:
        coverage = generate_coverage_stats()
        for key in [
            "user_coverage",
            "item_coverage",
            "cold_start_users",
            "cold_start_items",
        ]:
            assert key in coverage

    def test_coverage_bounded(self) -> None:
        coverage = generate_coverage_stats()
        assert 0 <= coverage["user_coverage"] <= 1
        assert 0 <= coverage["item_coverage"] <= 1

    def test_counts_positive(self) -> None:
        coverage = generate_coverage_stats()
        assert coverage["cold_start_users"] > 0
        assert coverage["total_users"] > 0
