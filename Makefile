init:
	pipenv install --skip-lock --dev
test:
	pipenv run pytest -n 4 --cov=lineflow --cov-report=term-missing tests
