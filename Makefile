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
	pytest -n logical --dist loadfile -s -v mislabeled

# Run code coverage

test-coverage:
	pytest --cov --cov-report term --cov-report xml --junitxml=junit.xml -n logical --dist loadfile -s -v mislabeled

# Check that docs can build

docs:
	cd docs && make clean && make html