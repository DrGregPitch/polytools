.PHONY: install test lint bench clean

install:
	uv venv && uv pip install -e ".[all]"

test:
	pytest tests -v

lint:
	ruff check src tests scripts

bench:
	python scripts/run_benchmark.py --outdir results

clean:
	rm -rf results .pytest_cache .ruff_cache .coverage __pycache__
