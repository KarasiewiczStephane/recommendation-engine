"""FastAPI dependency injection for shared resources.

Provides dependency functions for accessing the recommendation model,
cache layer, and user ratings across API endpoints.
"""

from fastapi import Request

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_recommender(request: Request) -> object:
    """Retrieve the recommendation model from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The loaded HybridRecommender instance.
    """
    return request.app.state.recommender


def get_movies(request: Request) -> dict:
    """Retrieve movie metadata from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        Dictionary mapping item_id to movie title.
    """
    return request.app.state.movies
