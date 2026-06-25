.PHONY: install build clean test

install:
	pip install -r requirements.txt
	pip install -e .

build: clean
	pip install build
	python -m build

clean:
	rm -rf dist/ build/ src/*.egg-info .pytest_cache
	find . -type d -name "__pycache__" -exec rm -r {} +

test:
	pytest tests/ -v
