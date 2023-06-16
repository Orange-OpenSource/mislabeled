.PHONY: quality format test test-examples coverage docs

# Check that source code meets quality standards

quality:
	black --check examples mislabeled
	flake8 mislabeled examples
	isort --check-only mislabeled examples

# Format source code automatically

format:
	isort examples mislabeled
	black examples mislabeled

# Run tests for the library

test:
	pytest -n auto --dist loadfile -s -v mislabeled

# Run tests for examples

test-examples:
	pytest -n auto --dist loadfile -s -v examples

# Run code coverage

coverage:
	pytest --cov -n auto --cov-report xml --dist loadfile -s -v mislabeled

# Check that docs can build

docs:
	cd docs && make clean && make html