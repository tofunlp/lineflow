.PHONY: init
init:
	poetry install --no-root

.PHONY: lint
lint:
	poetry run flake8

.PHONY: isort
isort:
	poetry run isort -c .

.PHONY: test
test:
	poetry run pytest --cov=lineflow --cov-report=term-missing tests

.PHONY: testall
testall:
	poetry run pytest --cov=lineflow --cov-report=term-missing --cov-report=xml tests -m "slow or not slow"
