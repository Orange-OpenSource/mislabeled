.PHONY: quality format test test-coverage docs

# Check that source code meets quality standards

quality:
	black --check examples mislabeled
	flake8 mislabeled examples
	isort --check-only mislabeled examples

# Format source code automatically

format:
	black examples mislabeled
	isort examples mislabeled

# Run tests for the library

test:
	pytest -n auto --maxprocesses=8 -s -v mislabeled --ignore=mislabeled/datasets/tests/test_wrench.py

# Run code coverage

test-coverage:
	pytest --cov --cov-report term --cov-report xml --junitxml=junit.xml -n auto --maxprocesses=8 -s -v mislabeled --ignore=mislabeled/datasets/tests/test_wrench.py

# Check that docs can build

docs:
	cd docs && make clean && make html