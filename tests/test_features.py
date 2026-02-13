"""Tests for TF-IDF feature extraction and data saving."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessor import extract_genre_features, save_processed_data


class TestExtractGenreFeatures:
    """Tests for the TF-IDF genre feature extraction."""

    def test_feature_matrix_shape(self, sample_movies: pd.DataFrame) -> None:
        """Feature matrix has one row per movie."""
        tfidf_matrix, _ = extract_genre_features(sample_movies)
        assert tfidf_matrix.shape[0] == len(sample_movies)

    def test_feature_matrix_nonzero(self, sample_movies: pd.DataFrame) -> None:
        """Feature matrix has non-zero entries."""
        tfidf_matrix, _ = extract_genre_features(sample_movies)
        assert tfidf_matrix.nnz > 0

    def test_vectorizer_vocabulary(self, sample_movies: pd.DataFrame) -> None:
        """Vectorizer learns genre terms from the data."""
        _, vectorizer = extract_genre_features(sample_movies)
        vocab = vectorizer.vocabulary_
        assert len(vocab) > 0

    def test_sparse_to_dense_conversion(self, sample_movies: pd.DataFrame) -> None:
        """Sparse matrix converts to dense array correctly."""
        tfidf_matrix, _ = extract_genre_features(sample_movies)
        dense = tfidf_matrix.toarray()
        assert dense.shape == tfidf_matrix.shape
        assert isinstance(dense, np.ndarray)

    def test_empty_genres_handled(self) -> None:
        """Movies with no genres get fallback features."""
        movies = pd.DataFrame(
            {
                "item_id": [1, 2],
                "title": ["Movie A", "Movie B"],
                "Action": [0, 1],
                "Comedy": [0, 0],
            }
        )
        tfidf_matrix, _ = extract_genre_features(movies)
        assert tfidf_matrix.shape[0] == 2


class TestSaveProcessedData:
    """Tests for the processed data saving functionality."""

    def test_parquet_round_trip(self, tmp_dir: Path) -> None:
        """Saved parquet files can be reloaded with identical data."""
        train = pd.DataFrame(
            {"user_id": [1, 2], "item_id": [1, 2], "rating": [5.0, 4.0]}
        )
        test = pd.DataFrame({"user_id": [1], "item_id": [3], "rating": [3.0]})
        movies = pd.DataFrame({"item_id": [1, 2, 3], "title": ["A", "B", "C"]})
        features = np.eye(3)

        save_processed_data(train, test, movies, features, str(tmp_dir))

        loaded_train = pd.read_parquet(tmp_dir / "train.parquet")
        loaded_test = pd.read_parquet(tmp_dir / "test.parquet")
        loaded_movies = pd.read_parquet(tmp_dir / "movies.parquet")
        loaded_features = np.load(tmp_dir / "item_features.npy")

        pd.testing.assert_frame_equal(loaded_train, train)
        pd.testing.assert_frame_equal(loaded_test, test)
        pd.testing.assert_frame_equal(loaded_movies, movies)
        np.testing.assert_array_equal(loaded_features, features)

    def test_sparse_features_saved(
        self, tmp_dir: Path, sample_movies: pd.DataFrame
    ) -> None:
        """Sparse feature matrices are saved correctly."""
        tfidf_matrix, _ = extract_genre_features(sample_movies)
        train = pd.DataFrame({"user_id": [1], "item_id": [1], "rating": [5.0]})
        test = pd.DataFrame({"user_id": [1], "item_id": [2], "rating": [4.0]})

        save_processed_data(train, test, sample_movies, tfidf_matrix, str(tmp_dir))

        loaded_features = np.load(tmp_dir / "item_features.npy")
        np.testing.assert_array_almost_equal(loaded_features, tfidf_matrix.toarray())

    def test_creates_output_directory(self, tmp_dir: Path) -> None:
        """Output directory is created if it does not exist."""
        out_path = tmp_dir / "nested" / "output"
        train = pd.DataFrame({"a": [1]})
        test = pd.DataFrame({"a": [2]})
        movies = pd.DataFrame({"a": [3]})
        features = np.array([[1.0]])

        save_processed_data(train, test, movies, features, str(out_path))
        assert out_path.exists()
