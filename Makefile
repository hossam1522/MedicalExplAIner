.PHONY: help check install test install-uv run run-nodiv dev clean

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help                                    Show this help message"
	@echo "  install-uv                              Install the uv package manager (required)"
	@echo "  install                                 Install the package and its dependencies"
	@echo "  test                                    Run the test suite"
	@echo "  run MODELS='model1 model2 ...'          Run the program with subtasks (space-separated models)"
	@echo "  run-nodiv MODELS='model1 model2 ...'    Run the program without subtasks division"
	@echo "  dev                                     Create a development virtual environment"
	@echo "  clean                                   Remove build artifacts and caches"
	@echo "  check                                   Check source files for syntax errors"

install-uv:
	wget -qO- https://astral.sh/uv/install.sh | sh

check:
	python3 -m py_compile medicalexplainer/*.py

install:
	uv pip install -e .

test:
	uv run pytest tests/

run:
	uv run python -m medicalexplainer \
		--dataset data/test.final.json \
		--models $(MODELS) --subtasks

run-nodiv:
	uv run python -m medicalexplainer \
		--dataset data/test.final.json \
		--models $(MODELS)

dev:
	uv venv .venv
	uv pip install -e .[dev]

clean:
	rm -rf *.egg-info/ .pytest_cache/ __pycache__/ build/ dist/ medicalexplainer/__pycache__/ tests/__pycache__/
