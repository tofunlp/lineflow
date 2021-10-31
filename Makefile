.PYTHON: init
init:
	poetry install --no-root

.PYTHON: lint
lint:
	poetry run flake8

.PYTHON: isort
isort:
	poetry run isort -c .

.PYTHON: test
test:
	poetry run pytest --cov=lineflow --cov-report=term-missing tests

.PYTHON: testall
testall:
	poetry run pytest --cov=lineflow --cov-report=term-missing --cov-report=xml tests -m "slow or not slow"
