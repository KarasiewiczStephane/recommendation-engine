"""Item similarity API endpoints.

Provides endpoints for finding items similar to a given item
using content-based cosine similarity.
"""

from fastapi import APIRouter, HTTPException, Query, Request

from src.api.schemas import RecommendationItem, SimilarItemsResponse

router = APIRouter(prefix="/similar", tags=["similarity"])


@router.get("/{item_id}", response_model=SimilarItemsResponse)
async def get_similar_items(
    request: Request,
    item_id: int,
    n: int = Query(default=10, ge=1, le=100),
) -> SimilarItemsResponse:
    """Find items most similar to a given item.

    Args:
        request: The incoming request.
        item_id: The reference item to find similar items for.
        n: Number of similar items to return.

    Returns:
        SimilarItemsResponse with a list of similar items.
    """
    recommender = request.app.state.recommender
    movies = request.app.state.movies

    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    similar = recommender.content_based.get_similar_items(item_id, n=n)

    if not similar:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    items = [
        RecommendationItem(
            item_id=sid,
            score=round(score, 4),
            title=movies.get(sid),
        )
        for sid, score in similar
    ]

    return SimilarItemsResponse(
        item_id=item_id,
        similar_items=items,
        cached=False,
    )
