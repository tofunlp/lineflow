init:
	pipenv install --skip-lock --dev
test:
	pipenv run pytest --cov=lineflow --cov-report=term-missing tests
test-all:
	pipenv run pytest --cov=lineflow --cov-report=term-missing --cov-report=xml tests -m "slow or not slow"
