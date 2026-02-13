"""Collaborative filtering using matrix factorization.

Implements collaborative filtering recommendation models using the Surprise
library, supporting SVD, SVD++, NMF, and KNN-based algorithms with
hyperparameter tuning capabilities.
"""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from surprise import NMF, SVD, Dataset, KNNBasic, Reader, SVDpp
from surprise.model_selection import GridSearchCV

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CollaborativeFilter:
    """Collaborative filtering recommender using matrix factorization.

    Supports multiple algorithms from the Surprise library and provides
    methods for training, prediction, recommendation, and serialization.

    Attributes:
        algorithm: Name of the algorithm to use (SVD, SVDpp, NMF, KNN_user, KNN_item).
        model: The trained Surprise model instance.
        trainset: The Surprise trainset object.
    """

    SUPPORTED_ALGORITHMS = {"SVD", "SVDpp", "NMF", "KNN_user", "KNN_item"}

    def __init__(self, algorithm: str = "SVD") -> None:
        """Initialize the collaborative filter.

        Args:
            algorithm: The algorithm variant to use.
        """
        self.algorithm = algorithm
        self.model: Any = None
        self.trainset: Any = None

    def _get_model(self, **params: Any) -> Any:
        """Instantiate the specified algorithm with given parameters.

        Args:
            **params: Keyword arguments passed to the algorithm constructor.

        Returns:
            An instantiated Surprise algorithm object.
        """
        models = {
            "SVD": lambda: SVD(**params),
            "SVDpp": lambda: SVDpp(**params),
            "NMF": lambda: NMF(**params),
            "KNN_user": lambda: KNNBasic(sim_options={"user_based": True}, **params),
            "KNN_item": lambda: KNNBasic(sim_options={"user_based": False}, **params),
        }
        factory = models.get(self.algorithm, lambda: SVD(**params))
        return factory()

    def tune_hyperparameters(self, ratings_df: pd.DataFrame) -> dict:
        """Find optimal hyperparameters using grid search cross-validation.

        Args:
            ratings_df: DataFrame with user_id, item_id, rating columns.

        Returns:
            Dictionary of best parameters for RMSE metric.
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[["user_id", "item_id", "rating"]], reader
        )

        param_grid = {
            "n_factors": [50, 100],
            "n_epochs": [20, 30],
            "lr_all": [0.005, 0.01],
            "reg_all": [0.02, 0.1],
        }

        gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
        gs.fit(data)

        best_params = gs.best_params["rmse"]
        logger.info("Best hyperparameters (RMSE): %s", best_params)
        return best_params

    def fit(self, ratings_df: pd.DataFrame, **params: Any) -> "CollaborativeFilter":
        """Train the collaborative filtering model on ratings data.

        Args:
            ratings_df: DataFrame with user_id, item_id, rating columns.
            **params: Algorithm-specific parameters.

        Returns:
            Self, for method chaining.
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_df[["user_id", "item_id", "rating"]], reader
        )
        self.trainset = data.build_full_trainset()
        self.model = self._get_model(**params)
        self.model.fit(self.trainset)
        logger.info(
            "Trained %s model on %d ratings",
            self.algorithm,
            self.trainset.n_ratings,
        )
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict a rating for a user-item pair.

        Args:
            user_id: The user identifier.
            item_id: The item identifier.

        Returns:
            Predicted rating value.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        return self.model.predict(user_id, item_id).est

    def recommend(
        self, user_id: int, n: int = 10, exclude_rated: bool = True
    ) -> list[tuple[int, float]]:
        """Generate top-N recommendations for a user.

        Args:
            user_id: The user to generate recommendations for.
            n: Number of recommendations to return.
            exclude_rated: Whether to exclude items the user has already rated.

        Returns:
            List of (item_id, predicted_rating) tuples sorted by score.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if self.model is None or self.trainset is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        all_items = set(self.trainset.all_items())

        if exclude_rated:
            try:
                inner_uid = self.trainset.to_inner_uid(user_id)
                rated_inner = {iid for iid, _ in self.trainset.ur[inner_uid]}
                candidate_inner = all_items - rated_inner
            except ValueError:
                candidate_inner = all_items
        else:
            candidate_inner = all_items

        candidate_items = [self.trainset.to_raw_iid(i) for i in candidate_inner]

        predictions = [(item, self.predict(user_id, item)) for item in candidate_items]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def save(self, path: str) -> None:
        """Serialize the model to disk.

        Args:
            path: File path for the pickle output.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved to %s", path)

    @staticmethod
    def load(path: str) -> "CollaborativeFilter":
        """Load a serialized model from disk.

        Args:
            path: Path to the pickle file.

        Returns:
            The deserialized CollaborativeFilter instance.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        logger.info("Model loaded from %s", path)
        return model
