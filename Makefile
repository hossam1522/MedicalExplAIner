.PHONY: help check install test install-uv run dev clean download-data clean-data delete-data

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  help          	Show this help message"
	@echo "  install-uv    	Install the uv package manager (required)"
	@echo "  install       	Install the package and its dependencies"
	@echo "  test          	Run the tests"
	@echo "  download-data  	Download network files from Wireshark samples"
	@echo "  clean-data N=<number>	Keep network files with a maximum of <number> packets"
	@echo "  delete-data   	Delete all network files"
	@echo "  run MODELS='model1 model2 ...'	Run the program with specified models (space-separated)"
	@echo "  dev           	Create a development environment"
	@echo "  clean         	Remove build artifacts"
	@echo "  check 	    	Check the code for syntax errors"

install-uv:
	wget -qO- https://astral.sh/uv/install.sh | sh

check:
	python3 -m py_compile medicalexplainer/*.py

install:
	uv run pip install .

run:
	uv run python -m medicalexplainer \
		--dataset medicalexplainer/data/test.final.json \
		--models $(MODELS)

dev:
	uv venv dev
	. dev/bin/activate
	uv run pip install -e .[dev]

clean:
	rm -rf *.egg-info/ .pytest_cache/ __pycache__/ build/ dist/ medicalexplainer/__pycache__/ tests/__pycache__/
