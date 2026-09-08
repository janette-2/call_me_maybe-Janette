install:
	uv sync

run: src/main.py
	uv run python -m src

debug: src/main.py
	uv run python -m pdb src/main.py

clean:
# Remove cache directories created by Python and mypy
	rm -rf __pycache__ .mypy_cache .pytest_cache
# .pyc files = compiled Python code that Python auto-generates when
# you import a module. They are safe to delete.
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -empty -delete

lint:
	uv run flake8 src/. 
	uv run mypy src/. --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/.
	uv run mypy src/. --strict

.PHONY: install run debug clean lint lint-strict