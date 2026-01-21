SHELL := /bin/bash

.PHONY: lint format typecheck test all

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run black src tests scripts

typecheck:
	uv run mypy src tests scripts

test:
	uv run pytest

all: lint typecheck test
