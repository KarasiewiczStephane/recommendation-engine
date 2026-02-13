"""Rating submission API endpoints.

Provides endpoints for users to submit ratings for items,
which are persisted to the database.
"""

import time

from fastapi import APIRouter, Request

from src.api.schemas import RatingRequest, RatingResponse
from src.utils.database import insert_rating
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["ratings"])


@router.post("/rate", response_model=RatingResponse)
async def submit_rating(
    request: Request,
    rating_request: RatingRequest,
) -> RatingResponse:
    """Submit a user rating for an item.

    Args:
        request: The incoming request.
        rating_request: The rating data to save.

    Returns:
        RatingResponse indicating success or failure.
    """
    db_path = request.app.state.db_path

    insert_rating(
        db_path,
        user_id=rating_request.user_id,
        item_id=rating_request.item_id,
        rating=rating_request.rating,
        timestamp=int(time.time()),
    )

    logger.info(
        "Rating saved: user=%d, item=%d, rating=%.1f",
        rating_request.user_id,
        rating_request.item_id,
        rating_request.rating,
    )

    return RatingResponse(
        success=True,
        message="Rating saved successfully",
    )
