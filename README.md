# Recommendation Engine

A hybrid movie recommendation system built with collaborative filtering, content-based filtering, and cold-start strategies. Trained on the MovieLens 100K dataset with a FastAPI serving layer, Redis caching, and A/B testing simulation.

## Features

- **Collaborative Filtering** — SVD, SVDpp, NMF, and KNN-based matrix factorization via scikit-surprise
- **Content-Based Filtering** — TF-IDF feature extraction with cosine similarity ranking
- **Hybrid Model** — Weighted and switching strategies combining both approaches
- **Cold-Start Handling** — Popularity-based recommendations for new users, content similarity for new items
- **Evaluation Suite** — RMSE, MAE, Precision@K, Recall@K, NDCG@K with model comparison
- **A/B Testing** — Simulation framework with CTR metrics and paired t-test significance testing
- **REST API** — FastAPI endpoints for recommendations, similar items, and rating submission
- **Redis Caching** — TTL-based caching with pattern-based invalidation
- **Docker Support** — Multi-service Docker Compose with Redis

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI     │────▶│  Hybrid      │────▶│ Collaborative│
│   Endpoints   │     │  Recommender │     │ Filter (SVD) │
└──────┬───────┘     └──────┬───────┘     └──────────────┘
       │                    │
       │                    └────────────▶┌──────────────┐
       │                                  │ Content-Based│
       ▼                                  │ Filter (TF-IDF)│
┌──────────────┐                          └──────────────┘
│  Redis Cache │
└──────────────┘     ┌──────────────┐
                     │ Cold-Start   │
┌──────────────┐     │ Handler      │
│  SQLite DB   │     └──────────────┘
└──────────────┘
```

## Quick Start

### Local Development

```bash
# Clone the repository
git clone git@github.com:KarasiewiczStephane/recommendation-engine.git
cd recommendation-engine

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the API server
make run
```

### Docker

```bash
# Start all services (API + Redis)
make docker

# Or run in detached mode
make docker-up

# Stop services
make docker-down
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with model status |
| GET | `/recommend/{user_id}` | Get recommendations (params: `n`, `strategy`) |
| GET | `/similar/{item_id}` | Get similar items (params: `n`) |
| POST | `/rate` | Submit a user rating |
| GET | `/ab-test/results` | View A/B test simulation results |

### Examples

```bash
# Get 5 recommendations for user 1
curl http://localhost:8000/recommend/1?n=5&strategy=weighted

# Find similar items to item 42
curl http://localhost:8000/similar/42?n=10

# Submit a rating
curl -X POST http://localhost:8000/rate \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "item_id": 42, "rating": 4.5}'

# Health check
curl http://localhost:8000/health
```

## Project Structure

```
recommendation-engine/
├── src/
│   ├── api/              # FastAPI application and routes
│   │   ├── app.py        # App setup with lifespan management
│   │   ├── routes/       # Endpoint handlers
│   │   ├── cache.py      # Redis caching layer
│   │   ├── schemas.py    # Pydantic request/response models
│   │   └── dependencies.py
│   ├── data/             # Data loading and preprocessing
│   │   ├── downloader.py # MovieLens dataset downloader
│   │   ├── preprocessor.py # Ratings/movies loading, TF-IDF features
│   │   └── splitter.py   # Temporal train/test split
│   ├── models/           # Recommendation models
│   │   ├── collaborative.py  # SVD/NMF/KNN collaborative filtering
│   │   ├── content_based.py  # TF-IDF + cosine similarity
│   │   ├── hybrid.py         # Weighted/switching hybrid
│   │   ├── cold_start.py     # New user/item strategies
│   │   ├── evaluator.py      # Metrics and model comparison
│   │   └── ab_test.py        # A/B testing simulation
│   ├── utils/            # Configuration, logging, database
│   └── main.py           # Application entry point
├── tests/                # Test suite (148 tests, 95% coverage)
├── configs/              # YAML configuration
├── data/sample/          # Sample MovieLens data for testing
├── .github/workflows/    # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── pyproject.toml
```

## Testing

```bash
# Run all tests with coverage
make test

# Run with verbose output
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing
```

## Linting

```bash
# Lint and format
make lint

# Check only (CI mode)
ruff check src/ tests/
ruff format --check src/ tests/
```

## Configuration

Application settings are in `configs/config.yaml`:

- **Model hyperparameters** — SVD factors, learning rate, regularization
- **Hybrid weights** — Alpha blending between collaborative and content-based
- **Redis TTLs** — Cache expiration for recommendations and similarity results
- **API settings** — Host, port configuration

Environment variables (`.env.example`):

```
REDIS_HOST=localhost
REDIS_PORT=6379
CONFIG_PATH=configs/config.yaml
```

## Tech Stack

- **Python 3.11+** with type annotations
- **scikit-surprise** for collaborative filtering
- **scikit-learn** for TF-IDF and cosine similarity
- **FastAPI** + **Uvicorn** for the REST API
- **Redis** for caching with TTL
- **SQLite** for persistent rating storage
- **pandas** + **NumPy** for data processing
- **pytest** with 95% code coverage
- **ruff** for linting and formatting
- **Docker** + **Docker Compose** for containerization
- **GitHub Actions** for CI/CD

## License

MIT
