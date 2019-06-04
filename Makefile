test: init
	pipenv run pytest tests
init:
	pipenv install --skip-lock --dev
