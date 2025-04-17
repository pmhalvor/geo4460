PYTHON = python

SRC_DIR = src/
TEST_DIR = tests/
OUTPUT_DIR = output_py_$(shell date +%Y%m%d_%H%M%S)

.PHONY: all run clean test install list_output help

all: help

run:
	@echo "--- Running DEM Analysis Workflow ---"
	$(PYTHON) -m src.workflow
	@echo "--- Workflow Finished ---"

clean:
	@echo "--- Available Output Directories ---"
	@ls -d output_py_* 2>/dev/null || echo "No output directories found."
	@echo "Enter the directory to clean (or type 'all' to clean all):"
	@read dir; \
	if [ "$$dir" = "all" ]; then \
		echo "--- Cleaning All Output Directories ---"; \
		rm -rf output_py_*; \
		echo "--- All Output Directories Cleaned ---"; \
	elif [ -d "$$dir" ]; then \
		echo "--- Cleaning Output Directory ($$dir) ---"; \
		rm -rf "$$dir"; \
		echo "--- Directory ($$dir) Cleaned ---"; \
	else \
		echo "Invalid directory: $$dir"; \
	fi

test:
	@echo "--- Running Tests ---"
	pytest $(TEST_DIR)
	# Alternative if running pytest from lab/GIS5:
	# pytest $(TEST_DIR)
	@echo "--- Testing Complete ---"

install:
	@echo "--- Installing Dependencies (run from project root) ---"
	pip install -e .[dev]
	@echo "--- Installation Command Shown (run manually from project root if needed) ---"

list_output:
	@echo "--- Listing Output Directory Contents (${OUTPUT_DIR}) ---"
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		ls -l $(OUTPUT_DIR); \
	else \
		echo "Output directory not found. Run the workflow first."; \
	fi
	@echo "--- Listing Complete ---"


help:
	@echo "Available commands:"
	@echo "  make run          - Run the main DEM analysis workflow"
	@echo "  make test         - Run pytest tests (expects to be run from project root)"
	@echo "  make clean        - Remove an output directory"
	@echo "  make list_output  - List the contents of the output directory"
	@echo "  make install      - Show command to install dependencies (run from project root)"
	@echo "  make help         - Show this help message"
