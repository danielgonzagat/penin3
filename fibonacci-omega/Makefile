.PHONY: install test lint format clean docker-build docker-run run-example help

help:
	@echo "Fibonacci Engine - Makefile Commands"
	@echo "===================================="
	@echo "install         Install in development mode"
	@echo "test            Run all tests"
	@echo "lint            Run linters"
	@echo "format          Format code with black"
	@echo "clean           Clean build artifacts"
	@echo "docker-build    Build Docker image"
	@echo "docker-run      Run in Docker container"
	@echo "run-example     Run example script"

install:
	pip install -e .

test:
	pytest fibonacci_engine/tests/ -v --cov=fibonacci_engine --cov-report=term-missing

lint:
	flake8 fibonacci_engine/ --max-line-length=100 --ignore=E203,W503 || true
	mypy fibonacci_engine/ --ignore-missing-imports || true

format:
	black fibonacci_engine/ --line-length=100

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f fibonacci_engine/persistence/*.json
	rm -f fibonacci_engine/reports/*.md

docker-build:
	docker build -t fibonacci-engine:latest .

docker-run:
	docker run -it --rm fibonacci-engine:latest fib run --adapter rl --generations 30

run-example:
	python fibonacci_engine/examples/run_example.py

# Quick start - install and run example
quickstart: install run-example

# Full workflow - install, test, run
all: install test run-example
	@echo "✅ Full workflow complete!"
