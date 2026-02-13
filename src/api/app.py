"""FastAPI application setup and configuration.

Defines the FastAPI application with lifespan management for model loading,
database initialization, and route registration.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import ab_test, rate, recommend, similar
from src.api.schemas import HealthResponse
from src.utils.database import get_user_ratings as db_get_user_ratings
from src.utils.database import init_db
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle: startup and shutdown.

    Initializes the database and sets up default state for the
    recommendation model and movie metadata.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application runtime.
    """
    db_path = "data/app.db"
    init_db(db_path)
    app.state.db_path = db_path
    app.state.recommender = None
    app.state.movies = {}

    def _get_user_ratings(user_id: int) -> list[tuple[int, float]]:
        return db_get_user_ratings(db_path, user_id)

    app.state.get_user_ratings = _get_user_ratings

    logger.info("Application started")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="Recommendation Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(recommend.router)
app.include_router(similar.router)
app.include_router(rate.router)
app.include_router(ab_test.router)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check application health status.

    Returns:
        HealthResponse with system status information.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=app.state.recommender is not None,
    )
