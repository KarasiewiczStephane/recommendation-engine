"""Tests for the FastAPI application endpoints."""

import random

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.data.preprocessor import extract_genre_features
from src.models.collaborative import CollaborativeFilter
from src.models.content_based import ContentBasedFilter
from src.models.hybrid import HybridRecommender


@pytest.fixture
def test_client(sample_movies: pd.DataFrame) -> TestClient:
    """Create a test client with mocked models loaded."""
    random.seed(42)
    rows = []
    for user_id in range(1, 11):
        for item_id in random.sample(range(1, 6), 4):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": float(random.randint(1, 5)),
                }
            )
    ratings_df = pd.DataFrame(rows)

    collab = CollaborativeFilter(algorithm="SVD")
    collab.fit(ratings_df, n_factors=5, n_epochs=10)

    tfidf_matrix, _ = extract_genre_features(sample_movies)
    content = ContentBasedFilter()
    content.fit(tfidf_matrix.toarray(), sample_movies["item_id"].values)

    hybrid = HybridRecommender(
        collaborative=collab,
        content_based=content,
        alpha=0.7,
        cold_start_threshold=5,
    )

    movies_dict = dict(zip(sample_movies["item_id"], sample_movies["title"]))

    with TestClient(app) as client:
        # Set state after lifespan has initialized
        app.state.recommender = hybrid
        app.state.movies = movies_dict
        app.state.get_user_ratings = lambda uid: []
        yield client


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, test_client: TestClient) -> None:
        """Health check returns 200."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, test_client: TestClient) -> None:
        """Health check returns expected fields."""
        response = test_client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "model_loaded" in data


class TestRecommendEndpoint:
    """Tests for the /recommend/{user_id} endpoint."""

    def test_recommend_returns_200(self, test_client: TestClient) -> None:
        """Recommendation endpoint returns valid response."""
        response = test_client.get("/recommend/1?n=5")
        assert response.status_code == 200

    def test_recommend_response_schema(self, test_client: TestClient) -> None:
        """Response has expected JSON structure."""
        response = test_client.get("/recommend/1?n=5")
        data = response.json()
        assert data["user_id"] == 1
        assert "recommendations" in data
        assert data["strategy"] == "weighted"
        assert len(data["recommendations"]) <= 5

    def test_recommend_with_strategy(self, test_client: TestClient) -> None:
        """Explicit strategy parameter is respected."""
        response = test_client.get("/recommend/1?n=3&strategy=switching")
        data = response.json()
        assert data["strategy"] == "switching"

    def test_recommend_invalid_strategy(self, test_client: TestClient) -> None:
        """Invalid strategy returns 400."""
        response = test_client.get("/recommend/1?strategy=invalid")
        assert response.status_code == 400

    def test_recommend_n_parameter_validation(self, test_client: TestClient) -> None:
        """N parameter is validated (ge=1, le=100)."""
        response = test_client.get("/recommend/1?n=0")
        assert response.status_code == 422

    def test_recommend_unknown_user(self, test_client: TestClient) -> None:
        """Unknown users still get recommendations."""
        response = test_client.get("/recommend/9999?n=3")
        assert response.status_code == 200


class TestSimilarEndpoint:
    """Tests for the /similar/{item_id} endpoint."""

    def test_similar_returns_200(self, test_client: TestClient) -> None:
        """Similar items endpoint returns valid response."""
        response = test_client.get("/similar/1?n=3")
        assert response.status_code == 200

    def test_similar_response_schema(self, test_client: TestClient) -> None:
        """Response has expected JSON structure."""
        response = test_client.get("/similar/1?n=3")
        data = response.json()
        assert data["item_id"] == 1
        assert "similar_items" in data

    def test_similar_unknown_item(self, test_client: TestClient) -> None:
        """Unknown item returns 404."""
        response = test_client.get("/similar/9999")
        assert response.status_code == 404


class TestRateEndpoint:
    """Tests for the /rate endpoint."""

    def test_rate_returns_200(self, test_client: TestClient) -> None:
        """Rating submission returns success."""
        response = test_client.post(
            "/rate",
            json={"user_id": 1, "item_id": 1, "rating": 4.5},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_rate_invalid_rating(self, test_client: TestClient) -> None:
        """Rating outside 1-5 range is rejected."""
        response = test_client.post(
            "/rate",
            json={"user_id": 1, "item_id": 1, "rating": 6.0},
        )
        assert response.status_code == 422

    def test_rate_missing_fields(self, test_client: TestClient) -> None:
        """Missing required fields are rejected."""
        response = test_client.post(
            "/rate",
            json={"user_id": 1},
        )
        assert response.status_code == 422
