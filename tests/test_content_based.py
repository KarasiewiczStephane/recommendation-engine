"""Tests for the content-based filtering module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import extract_genre_features
from src.models.content_based import ContentBasedFilter


@pytest.fixture
def fitted_content_model(sample_movies: pd.DataFrame) -> ContentBasedFilter:
    """Create a fitted content-based model on sample data."""
    tfidf_matrix, _ = extract_genre_features(sample_movies)
    model = ContentBasedFilter()
    model.fit(tfidf_matrix.toarray(), sample_movies["item_id"].values)
    return model


class TestContentBasedFilter:
    """Tests for the ContentBasedFilter class."""

    def test_fit_creates_similarity_matrix(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Fitting creates a similarity matrix."""
        assert fitted_content_model.similarity_matrix is not None

    def test_similarity_matrix_is_symmetric(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Similarity matrix is symmetric (sim[i,j] == sim[j,i])."""
        sim = fitted_content_model.similarity_matrix
        np.testing.assert_array_almost_equal(sim, sim.T)

    def test_similarity_diagonal_is_one(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Diagonal of similarity matrix is all 1.0 (self-similarity)."""
        sim = fitted_content_model.similarity_matrix
        np.testing.assert_array_almost_equal(np.diag(sim), np.ones(len(sim)))

    def test_similar_items_returns_correct_count(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """get_similar_items returns the requested number of results."""
        similar = fitted_content_model.get_similar_items(1, n=3)
        assert len(similar) == 3

    def test_similar_items_excludes_self(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Similar items list does not include the query item."""
        similar = fitted_content_model.get_similar_items(1, n=4)
        similar_ids = {item_id for item_id, _ in similar}
        assert 1 not in similar_ids

    def test_similar_items_unknown_item(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Unknown item returns empty list."""
        similar = fitted_content_model.get_similar_items(9999, n=5)
        assert similar == []

    def test_user_profile_shape(self, fitted_content_model: ContentBasedFilter) -> None:
        """User profile has the correct number of features."""
        profile = fitted_content_model.build_user_profile([(1, 5.0), (2, 4.0)])
        assert profile.shape == (fitted_content_model.item_features.shape[1],)

    def test_user_profile_empty_ratings(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Empty user ratings return a zero profile vector."""
        profile = fitted_content_model.build_user_profile([])
        np.testing.assert_array_equal(
            profile, np.zeros(fitted_content_model.item_features.shape[1])
        )

    def test_user_profile_low_ratings_ignored(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Ratings below threshold are excluded from profile."""
        profile = fitted_content_model.build_user_profile(
            [(1, 1.0), (2, 2.0)], rating_threshold=3.5
        )
        np.testing.assert_array_equal(
            profile, np.zeros(fitted_content_model.item_features.shape[1])
        )

    def test_recommend_returns_correct_count(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Recommendations return the requested number of items."""
        profile = fitted_content_model.build_user_profile([(1, 5.0)])
        recs = fitted_content_model.recommend(profile, n=3)
        assert len(recs) == 3

    def test_recommend_excludes_items(
        self, fitted_content_model: ContentBasedFilter
    ) -> None:
        """Excluded items do not appear in recommendations."""
        profile = fitted_content_model.build_user_profile([(1, 5.0)])
        recs = fitted_content_model.recommend(profile, n=5, exclude_items={1, 2})
        rec_ids = {item_id for item_id, _ in recs}
        assert 1 not in rec_ids
        assert 2 not in rec_ids

    def test_save_and_load(
        self, fitted_content_model: ContentBasedFilter, tmp_dir: Path
    ) -> None:
        """Model can be serialized and deserialized."""
        model_path = str(tmp_dir / "cb_model.pkl")
        fitted_content_model.save(model_path)

        loaded = ContentBasedFilter.load(model_path)
        np.testing.assert_array_equal(
            loaded.item_features, fitted_content_model.item_features
        )
        np.testing.assert_array_equal(
            loaded.similarity_matrix, fitted_content_model.similarity_matrix
        )

    def test_fit_with_sparse_input(self, sample_movies: pd.DataFrame) -> None:
        """Model accepts sparse matrices as input."""
        tfidf_matrix, _ = extract_genre_features(sample_movies)
        model = ContentBasedFilter()
        model.fit(tfidf_matrix, sample_movies["item_id"].values)
        assert model.similarity_matrix is not None
