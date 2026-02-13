"""Application entry point for the recommendation engine API server."""

import uvicorn

from src.utils.config import config


def main() -> None:
    """Start the uvicorn server with configuration from config.yaml."""
    uvicorn.run(
        "src.api.app:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=False,
    )


if __name__ == "__main__":
    main()
