.PHONY: quality format test test-coverage docs

# Check that source code meets quality standards

quality:
	black --check mislabeled examples
	flake8 mislabeled examples
	isort --check-only mislabeled examples

# Format source code automatically

format:
	black examples mislabeled
	isort examples mislabeled

# Run tests for the library

test:
	pytest -n auto --maxprocesses=8 -s -v mislabeled --ignore=mislabeled/datasets --ignore=mislabeled/tests/test_cache.py

test-datasets:
	pytest -n auto --maxprocesses=8 -s -v mislabeled/datasets

# Run code coverage

test-coverage:
	pytest --cov --cov-report term --cov-report xml --junitxml=junit.xml -n auto --maxprocesses=8 -s -v mislabeled --ignore=mislabeled/datasets/tests --ignore=mislabeled/tests/test_cache.py

# Check that docs can build

docs:
	cd docs && make clean && make html