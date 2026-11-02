.PHONY: setup dev check test e2e

setup:
	python3 scripts/setup.py

dev:
	python3 scripts/dev.py

check:
	backend/.venv/bin/ruff check --config backend/pyproject.toml backend/app backend/tests scripts/dev.py scripts/setup.py
	cd frontend && node node_modules/eslint/bin/eslint.js .
	cd frontend && node node_modules/typescript/bin/tsc -b && node node_modules/vite/bin/vite.js build

test:
	cd backend && .venv/bin/python -m pytest -q
	cd frontend && node node_modules/vitest/vitest.mjs run

e2e:
	cd frontend && node node_modules/@playwright/test/cli.js test
