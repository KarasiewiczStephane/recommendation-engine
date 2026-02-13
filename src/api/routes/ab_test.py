"""A/B test results API endpoints.

Provides endpoints for viewing A/B test simulation results
and statistical comparisons between recommendation strategies.
"""

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/ab-test", tags=["ab-testing"])


@router.get("/results")
async def get_ab_results(request: Request) -> dict:
    """Get the latest A/B test simulation results.

    Args:
        request: The incoming request.

    Returns:
        Dictionary containing results for all tested variants.
    """
    simulator = getattr(request.app.state, "ab_simulator", None)
    if simulator is None:
        raise HTTPException(status_code=404, detail="No A/B test results available")

    results = simulator.get_results()
    return {
        "results": [
            {
                "variant": r.variant,
                "impressions": r.impressions,
                "clicks": r.clicks,
                "ctr": r.ctr,
                "ndcg": r.ndcg,
                "confidence_interval": r.confidence_interval,
            }
            for r in results
        ]
    }
