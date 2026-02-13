"""Recommendation API endpoints.

Provides endpoints for generating personalized recommendations
for users using the hybrid recommendation engine.
"""

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.schemas import RecommendationItem, RecommendResponse

router = APIRouter(prefix="/recommend", tags=["recommendations"])


@router.get("/{user_id}", response_model=RecommendResponse)
async def get_recommendations(
    request: Request,
    user_id: int,
    n: int = Query(default=10, ge=1, le=100),
    strategy: str = Query(default="weighted"),
) -> RecommendResponse:
    """Get top-N recommendations for a user.

    Args:
        request: The incoming request.
        user_id: The user to generate recommendations for.
        n: Number of recommendations to return.
        strategy: Recommendation strategy (weighted or switching).

    Returns:
        RecommendResponse with a list of recommended items.
    """
    recommender = request.app.state.recommender
    movies = request.app.state.movies

    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    user_ratings = request.app.state.get_user_ratings(user_id)

    try:
        recs = recommender.recommend(
            user_id=user_id,
            user_ratings=user_ratings,
            n=n,
            strategy=strategy,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    items = [
        RecommendationItem(
            item_id=item_id,
            score=round(score, 4),
            title=movies.get(item_id),
        )
        for item_id, score in recs
    ]

    return RecommendResponse(
        user_id=user_id,
        recommendations=items,
        strategy=strategy,
        cached=False,
    )
