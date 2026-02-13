"""A/B testing simulation for comparing recommendation strategies.

Simulates user sessions across multiple recommendation model variants
to compare click-through rates and NDCG metrics with statistical
significance testing.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ABTestResult:
    """Results for a single A/B test variant.

    Attributes:
        variant: Name of the model variant.
        impressions: Total number of recommendations shown.
        clicks: Simulated clicks (items rated >= threshold in test set).
        ctr: Click-through rate.
        ndcg: Average NDCG across all user sessions.
        confidence_interval: 95% confidence interval for CTR.
    """

    variant: str
    impressions: int
    clicks: int
    ctr: float
    ndcg: float
    confidence_interval: tuple[float, float]


class ABTestSimulator:
    """Simulation-based A/B testing for recommendation models.

    Compares multiple recommendation strategies by simulating user
    sessions using held-out test ratings as ground truth for
    engagement signals.

    Attributes:
        models: Dictionary mapping variant names to model instances.
        click_threshold: Rating threshold to consider as a click.
        results: Accumulated test results per variant.
    """

    def __init__(
        self,
        models: dict[str, object],
        click_threshold: float = 4.0,
    ) -> None:
        """Initialize the A/B test simulator.

        Args:
            models: Dictionary mapping variant name to recommendation model.
            click_threshold: Minimum test rating to count as a simulated click.
        """
        self.models = models
        self.click_threshold = click_threshold
        self.results: dict[str, dict] = defaultdict(
            lambda: {"impressions": 0, "clicks": 0, "ndcgs": []}
        )

    def _compute_ndcg(self, relevances: list[float], k: int) -> float:
        """Compute NDCG for a single ranking.

        Args:
            relevances: Relevance scores in recommendation order.
            k: Number of positions to evaluate.

        Returns:
            NDCG value between 0 and 1.
        """
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevances[:k]))
        ideal = sorted(relevances, reverse=True)[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
        return float(dcg / idcg) if idcg > 0 else 0.0

    def simulate_user_session(
        self,
        user_id: int,
        user_train_ratings: list[tuple[int, float]],
        user_test_ratings: list[tuple[int, float]],
        n_recommendations: int = 10,
    ) -> dict[str, dict]:
        """Simulate one user session across all model variants.

        Args:
            user_id: The user identifier.
            user_train_ratings: User's training ratings.
            user_test_ratings: User's held-out test ratings.
            n_recommendations: Number of recommendations per variant.

        Returns:
            Per-variant session results with clicks and NDCG.
        """
        test_items = dict(user_test_ratings)
        session_results = {}

        for variant, model in self.models.items():
            recs = model.recommend(user_id, user_train_ratings, n=n_recommendations)
            rec_items = [item for item, _ in recs]

            clicks = sum(
                1
                for item in rec_items
                if item in test_items and test_items[item] >= self.click_threshold
            )

            relevances = [test_items.get(item, 0) for item in rec_items]
            ndcg = self._compute_ndcg(relevances, n_recommendations)

            self.results[variant]["impressions"] += n_recommendations
            self.results[variant]["clicks"] += clicks
            self.results[variant]["ndcgs"].append(ndcg)

            session_results[variant] = {"clicks": clicks, "ndcg": ndcg}

        return session_results

    def run_simulation(
        self,
        train_data: dict[int, list[tuple[int, float]]],
        test_data: dict[int, list[tuple[int, float]]],
    ) -> None:
        """Run the full A/B test simulation across all users.

        Args:
            train_data: Per-user training ratings.
            test_data: Per-user test ratings.
        """
        for user_id in train_data:
            if user_id in test_data and len(test_data[user_id]) > 0:
                self.simulate_user_session(
                    user_id,
                    train_data[user_id],
                    test_data[user_id],
                )
        logger.info("A/B simulation complete for %d variants", len(self.models))

    def get_results(self) -> list[ABTestResult]:
        """Compute final aggregated results with confidence intervals.

        Returns:
            List of ABTestResult dataclass instances.
        """
        results = []
        for variant, data in self.results.items():
            impressions = data["impressions"]
            clicks = data["clicks"]
            ctr = clicks / impressions if impressions > 0 else 0
            ndcg = float(np.mean(data["ndcgs"])) if data["ndcgs"] else 0

            if impressions > 0:
                se = np.sqrt(ctr * (1 - ctr) / impressions)
                ci = (max(0.0, ctr - 1.96 * se), min(1.0, ctr + 1.96 * se))
            else:
                ci = (0.0, 0.0)

            results.append(
                ABTestResult(
                    variant=variant,
                    impressions=impressions,
                    clicks=clicks,
                    ctr=round(ctr, 6),
                    ndcg=round(ndcg, 6),
                    confidence_interval=(round(ci[0], 6), round(ci[1], 6)),
                )
            )
        return results

    def statistical_comparison(
        self, variant_a: str, variant_b: str
    ) -> dict[str, object]:
        """Perform a paired t-test comparing two variants on NDCG.

        Args:
            variant_a: Name of the first variant.
            variant_b: Name of the second variant.

        Returns:
            Dictionary with t-statistic, p-value, significance, and winner.
        """
        ndcgs_a = self.results[variant_a]["ndcgs"]
        ndcgs_b = self.results[variant_b]["ndcgs"]

        min_len = min(len(ndcgs_a), len(ndcgs_b))
        ndcgs_a = ndcgs_a[:min_len]
        ndcgs_b = ndcgs_b[:min_len]

        if min_len < 2:
            return {
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "winner": "insufficient_data",
            }

        t_stat, p_value = stats.ttest_rel(ndcgs_a, ndcgs_b)

        if np.isnan(p_value):
            t_stat = 0.0
            p_value = 1.0

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "winner": variant_a if np.mean(ndcgs_a) > np.mean(ndcgs_b) else variant_b,
        }
