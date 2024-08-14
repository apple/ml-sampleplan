.PHONY: format black isort test

black:
	black --line-length 120 sampleplan/ tests/ experiments/

isort:
	isort --profile black sampleplan/ tests/ experiments/

format: black isort

requirements:
	poetry update

install:
	poetry install

test:
	 python -m pytest tests
