"""Evaluation metrics and model comparison for recommendation systems.

Implements standard recommendation evaluation metrics including RMSE, MAE,
Precision@K, Recall@K, and NDCG@K with support for multi-model comparison.
"""

from collections import defaultdict

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Comprehensive evaluation suite for recommendation models.

    Computes rating accuracy and ranking quality metrics at configurable
    K values, with support for comparing multiple models.

    Attributes:
        k_values: List of K values for top-K metric evaluation.
    """

    def __init__(self, k_values: list[int] | None = None) -> None:
        """Initialize the evaluator.

        Args:
            k_values: K values for Precision@K, Recall@K, NDCG@K.
        """
        self.k_values = k_values or [5, 10, 20]

    @staticmethod
    def rmse(predictions: list[tuple[float, float]]) -> float:
        """Compute Root Mean Square Error.

        Args:
            predictions: List of (predicted, actual) value pairs.

        Returns:
            RMSE value.
        """
        if not predictions:
            return 0.0
        errors = [(pred - actual) ** 2 for pred, actual in predictions]
        return float(np.sqrt(np.mean(errors)))

    @staticmethod
    def mae(predictions: list[tuple[float, float]]) -> float:
        """Compute Mean Absolute Error.

        Args:
            predictions: List of (predicted, actual) value pairs.

        Returns:
            MAE value.
        """
        if not predictions:
            return 0.0
        errors = [abs(pred - actual) for pred, actual in predictions]
        return float(np.mean(errors))

    @staticmethod
    def precision_at_k(
        recommended: list[int],
        relevant: set,
        k: int,
    ) -> float:
        """Compute Precision@K.

        Measures the fraction of top-K recommendations that are relevant.

        Args:
            recommended: Ordered list of recommended item IDs.
            relevant: Set of relevant (ground-truth positive) item IDs.
            k: Number of top recommendations to evaluate.

        Returns:
            Precision@K value between 0 and 1.
        """
        if k <= 0:
            return 0.0
        top_k = recommended[:k]
        hits = len(set(top_k) & relevant)
        return hits / k

    @staticmethod
    def recall_at_k(
        recommended: list[int],
        relevant: set,
        k: int,
    ) -> float:
        """Compute Recall@K.

        Measures the fraction of relevant items captured in top-K.

        Args:
            recommended: Ordered list of recommended item IDs.
            relevant: Set of relevant (ground-truth positive) item IDs.
            k: Number of top recommendations to evaluate.

        Returns:
            Recall@K value between 0 and 1.
        """
        if not relevant:
            return 0.0
        top_k = recommended[:k]
        hits = len(set(top_k) & relevant)
        return hits / len(relevant)

    @staticmethod
    def ndcg_at_k(
        recommended: list[int],
        relevant: dict[int, float],
        k: int,
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain@K.

        Args:
            recommended: Ordered list of recommended item IDs.
            relevant: Dictionary mapping item_id to relevance score.
            k: Number of top recommendations to evaluate.

        Returns:
            NDCG@K value between 0 and 1.
        """

        def _dcg(scores: list[float], k: int) -> float:
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores[:k]))

        actual_scores = [relevant.get(item, 0) for item in recommended[:k]]
        actual_dcg = _dcg(actual_scores, k)

        ideal_scores = sorted(relevant.values(), reverse=True)[:k]
        ideal_dcg = _dcg(ideal_scores, k)

        return float(actual_dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    def evaluate_ranking(
        self,
        model: object,
        test_ratings: dict[int, list[tuple[int, float]]],
        train_ratings: dict[int, list[tuple[int, float]]],
        relevance_threshold: float = 4.0,
    ) -> dict[str, float]:
        """Evaluate a model's ranking quality across all users.

        Args:
            model: A recommendation model with a recommend() method.
            test_ratings: Per-user test set as {user_id: [(item_id, rating)]}.
            train_ratings: Per-user training set as {user_id: [(item_id, rating)]}.
            relevance_threshold: Minimum rating to consider an item relevant.

        Returns:
            Dictionary of averaged metric values.
        """
        metrics: dict[str, list[float]] = defaultdict(list)

        for user_id, test_items in test_ratings.items():
            if user_id not in train_ratings:
                continue

            recs = model.recommend(
                user_id,
                train_ratings[user_id],
                n=max(self.k_values),
            )
            rec_items = [item for item, _ in recs]

            relevant = {
                item for item, rating in test_items if rating >= relevance_threshold
            }
            relevant_scores = dict(test_items)

            for k in self.k_values:
                metrics[f"precision@{k}"].append(
                    self.precision_at_k(rec_items, relevant, k)
                )
                metrics[f"recall@{k}"].append(self.recall_at_k(rec_items, relevant, k))
                metrics[f"ndcg@{k}"].append(
                    self.ndcg_at_k(rec_items, relevant_scores, k)
                )

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def compare_models(
        self,
        models: dict[str, object],
        test_data: dict[int, list[tuple[int, float]]],
        train_data: dict[int, list[tuple[int, float]]],
    ) -> dict[str, dict[str, float]]:
        """Compare multiple models on the same evaluation data.

        Args:
            models: Dictionary mapping model name to model instance.
            test_data: Per-user test ratings.
            train_data: Per-user training ratings.

        Returns:
            Nested dictionary of {model_name: {metric: value}}.
        """
        results = {}
        for name, model in models.items():
            logger.info("Evaluating model: %s", name)
            results[name] = self.evaluate_ranking(model, test_data, train_data)
        return results
