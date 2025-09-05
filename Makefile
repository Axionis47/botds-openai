.PHONY: help install test test-quick clean run-iris run-breast-cancer run-diabetes

help:
	@echo "Bot Data Scientist - Available commands:"
	@echo ""
	@echo "  install          Install dependencies"
	@echo "  test             Run full test suite"
	@echo "  test-quick       Run quick system test"
	@echo "  clean            Clean cache and artifacts"
	@echo ""
	@echo "  run-iris         Run on Iris dataset"
	@echo "  run-breast-cancer Run on Breast Cancer dataset"
	@echo "  run-diabetes     Run on Diabetes dataset"
	@echo ""
	@echo "Environment setup:"
	@echo "  export OPENAI_API_KEY=sk-your-key-here"

install:
	pip install -r requirements.txt
	pip install pytest

test:
	python -m pytest tests/ -v

test-quick:
	python test_system.py

clean:
	rm -rf cache/*
	rm -rf artifacts/*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

run-iris:
	python -m cli.run --config configs/iris.yaml

run-breast-cancer:
	python -m cli.run --config configs/breast_cancer.yaml

run-diabetes:
	python -m cli.run --config configs/diabetes.yaml
