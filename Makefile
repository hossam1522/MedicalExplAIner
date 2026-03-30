.PHONY: help check install test install-uv run run-nodiv run-nothink run-nodiv-nothink dev clean download-demo-data

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help                                    Show this help message"
	@echo "  install-uv                              Install the uv package manager (required)"
	@echo "  install                                 Install the package and its dependencies"
	@echo "  test                                    Run the test suite"
	@echo "  download-demo-data                      Download MIMIC-IV-ED demo CSV files"
	@echo "  run MODELS='model1 model2 ...'               Run evaluation with subtasks"
	@echo "  run-nodiv MODELS='model1 model2 ...'         Run evaluation without subtasks"
	@echo "  run-nothink MODELS='model1 model2 ...'       Run with subtasks, thinking disabled"
	@echo "  run-nodiv-nothink MODELS='model1 model2 ...' Run without subtasks, thinking disabled"
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

download-demo-data:
	@echo "Downloading MIMIC-IV-ED demo data..."
	@mkdir -p data
	wget -r -np -nd -A "*.csv.gz" -P data/ \
		https://physionet.org/files/mimic-iv-ed-demo/2.2/ed/
	@echo "Decompressing..."
	gunzip -f data/*.csv.gz
	@echo "Removing unnecessary files (medrecon, pyxis)..."
	rm -f data/medrecon.csv data/pyxis.csv
	@echo "Done. CSV files in data/:"
	@ls data/*.csv

run:
	uv run python -m medicalexplainer \
		--models $(MODELS) --subtasks

run-nodiv:
	uv run python -m medicalexplainer \
		--models $(MODELS)

run-nothink:
	uv run python -m medicalexplainer \
		--models $(MODELS) --subtasks --no-think

run-nodiv-nothink:
	uv run python -m medicalexplainer \
		--models $(MODELS) --no-think

dev:
	uv venv .venv
	uv pip install -e .[dev]

clean:
	rm -rf *.egg-info/ .pytest_cache/ __pycache__/ build/ dist/ medicalexplainer/__pycache__/ tests/__pycache__/
