# Recommendation Engine

A hybrid movie recommendation system built with collaborative filtering, content-based filtering, and cold-start strategies. Trained on the MovieLens 100K dataset with a FastAPI serving layer, Redis caching, and A/B testing simulation.

## Features

- **Collaborative Filtering** -- SVD, SVDpp, NMF, and KNN-based matrix factorization via scikit-surprise
- **Content-Based Filtering** -- TF-IDF feature extraction with cosine similarity ranking
- **Hybrid Model** -- Weighted and switching strategies combining both approaches
- **Cold-Start Handling** -- Popularity-based recommendations for new users, content similarity for new items
- **Evaluation Suite** -- RMSE, MAE, Precision@K, Recall@K, NDCG@K with model comparison
- **A/B Testing** -- Simulation framework with CTR metrics and paired t-test significance testing
- **REST API** -- FastAPI endpoints for recommendations, similar items, and rating submission
- **Redis Caching** -- TTL-based caching with pattern-based invalidation
- **Streamlit Dashboard** -- Interactive visualization of ranking metrics, model comparison, A/B test results, and coverage analysis
- **Docker Support** -- Multi-service Docker Compose with Redis

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI     │────>│  Hybrid      │────>│ Collaborative│
│   Endpoints   │     │  Recommender │     │ Filter (SVD) │
└──────┬───────┘     └──────┬───────┘     └──────────────┘
       │                    │
       │                    └────────────>┌──────────────┐
       │                                  │ Content-Based│
       v                                  │ Filter (TF-IDF)│
┌──────────────┐                          └──────────────┘
│  Redis Cache │
└──────────────┘     ┌──────────────┐
                     │ Cold-Start   │
┌──────────────┐     │ Handler      │
│  SQLite DB   │     └──────────────┘
└──────────────┘
                     ┌──────────────┐
                     │  Streamlit   │
                     │  Dashboard   │
                     └──────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
git clone git@github.com:KarasiewiczStephane/recommendation-engine.git
cd recommendation-engine
make install
```

This runs `pip install -r requirements.txt` and `pip install -e .` (editable install).

### 2. Download the MovieLens 100K dataset

```bash
make download-data
```

This downloads and extracts the MovieLens 100K dataset into `data/raw/ml-100k/`. The download is skipped if the data already exists. A small sample dataset is also included in `data/sample/ml-100k/` for testing.

### 3. Launch the API server

```bash
make run
```

The FastAPI server starts at [http://localhost:8000](http://localhost:8000). See the API Endpoints section below for usage.

### 4. Launch the dashboard

```bash
make dashboard
```

Opens the Streamlit dashboard at [http://localhost:8501](http://localhost:8501), displaying ranking metrics, model accuracy comparison, A/B test CTR trends, and coverage/cold-start analysis.

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
│   ├── api/                 # FastAPI application and routes
│   │   ├── app.py           # App setup with lifespan management
│   │   ├── routes/          # Endpoint handlers
│   │   │   ├── recommend.py
│   │   │   ├── similar.py
│   │   │   ├── rate.py
│   │   │   └── ab_test.py
│   │   ├── cache.py         # Redis caching layer
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── dependencies.py
│   ├── data/                # Data loading and preprocessing
│   │   ├── downloader.py    # MovieLens dataset downloader
│   │   ├── preprocessor.py  # Ratings/movies loading, TF-IDF features
│   │   └── splitter.py      # Temporal train/test split
│   ├── models/              # Recommendation models
│   │   ├── collaborative.py # SVD/NMF/KNN collaborative filtering
│   │   ├── content_based.py # TF-IDF + cosine similarity
│   │   ├── hybrid.py        # Weighted/switching hybrid
│   │   ├── cold_start.py    # New user/item strategies
│   │   ├── evaluator.py     # Metrics and model comparison
│   │   └── ab_test.py       # A/B testing simulation
│   ├── dashboard/
│   │   └── app.py           # Streamlit dashboard entry point
│   ├── utils/               # Configuration, logging, database
│   └── main.py              # API server entry point (uvicorn)
├── tests/                   # Test suite
├── configs/
│   └── config.yaml          # Model, Redis, API configuration
├── data/
│   └── sample/              # Sample MovieLens data for testing
├── .github/workflows/
│   └── ci.yml               # CI pipeline: lint, test, docker
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

- **Model hyperparameters** -- SVD factors, learning rate, regularization
- **Hybrid weights** -- Alpha blending between collaborative and content-based
- **Redis TTLs** -- Cache expiration for recommendations and similarity results
- **API settings** -- Host, port configuration

Environment variables (`.env.example`):

```
REDIS_HOST=localhost
REDIS_PORT=6379
CONFIG_PATH=configs/config.yaml
```

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies and editable package |
| `make run` | Start the FastAPI server |
| `make dashboard` | Launch the Streamlit dashboard |
| `make download-data` | Download the MovieLens 100K dataset |
| `make test` | Run tests with coverage |
| `make lint` | Lint and format with ruff |
| `make docker` | Build and run all services |
| `make docker-up` | Start services in detached mode |
| `make docker-down` | Stop all services |
| `make clean` | Remove __pycache__ and .pyc files |

## Tech Stack

- **Python 3.11+** with type annotations
- **scikit-surprise** for collaborative filtering
- **scikit-learn** for TF-IDF and cosine similarity
- **FastAPI** + **Uvicorn** for the REST API
- **Redis** for caching with TTL
- **SQLite** for persistent rating storage
- **Streamlit** + **Plotly** for the interactive dashboard
- **pandas** + **NumPy** for data processing
- **pytest** for testing
- **ruff** for linting and formatting
- **Docker** + **Docker Compose** for containerization
- **GitHub Actions** for CI/CD

## License

MIT
