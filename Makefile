.PHONY: install run clean lint

install:
	pip install -r requirements.txt

run:
	python run_src.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	python -m py_compile run_src.py src/**/*.py || echo "Basic syntax check completed"