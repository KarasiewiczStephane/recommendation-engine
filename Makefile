.PHONY: install test lint clean run docker docker-up docker-down docker-build dashboard download-data

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker:
	docker compose up --build

dashboard:
	streamlit run src/dashboard/app.py

download-data:
	python -c "from src.data.downloader import download_movielens; download_movielens()"
