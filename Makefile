init:
	pipenv install --skip-lock --dev
test:
	pipenv run pytest --cov=lineflow --cov-report=term-missing tests
