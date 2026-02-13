"""Pydantic schemas for API request/response validation.

Defines the data models used for serialization and validation of all
API endpoint inputs and outputs.
"""

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """A single recommended item with its score.

    Attributes:
        item_id: The item identifier.
        score: The recommendation score.
        title: Optional item title.
    """

    item_id: int
    score: float
    title: str | None = None


class RecommendResponse(BaseModel):
    """Response schema for recommendation endpoints.

    Attributes:
        user_id: The user who received recommendations.
        recommendations: List of recommended items.
        strategy: The recommendation strategy used.
        cached: Whether the result was served from cache.
    """

    user_id: int
    recommendations: list[RecommendationItem]
    strategy: str
    cached: bool = False


class SimilarItemsResponse(BaseModel):
    """Response schema for item similarity endpoint.

    Attributes:
        item_id: The query item identifier.
        similar_items: List of similar items with scores.
        cached: Whether the result was served from cache.
    """

    item_id: int
    similar_items: list[RecommendationItem]
    cached: bool = False


class RatingRequest(BaseModel):
    """Request schema for submitting a user rating.

    Attributes:
        user_id: The user submitting the rating.
        item_id: The item being rated.
        rating: The rating value (1.0 to 5.0).
    """

    user_id: int
    item_id: int
    rating: float = Field(ge=1.0, le=5.0)


class RatingResponse(BaseModel):
    """Response schema for rating submission.

    Attributes:
        success: Whether the rating was saved.
        message: Human-readable status message.
    """

    success: bool
    message: str


class HealthResponse(BaseModel):
    """Response schema for health check endpoint.

    Attributes:
        status: Overall health status.
        model_loaded: Whether the recommendation model is loaded.
    """

    status: str
    model_loaded: bool
