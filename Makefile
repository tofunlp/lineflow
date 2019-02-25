test: init
	pipenv run pytest tests
init:
	pipenv install --skip-lock --dev
publish:
	pip install 'twine>=1.5.0'
	python setup.py sdist bdist_wheel
	twine upload dist/*
	rm -fr build dist .egg lineflow.egg-info
