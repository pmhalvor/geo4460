# Makefile for GIS5 DEM Analysis Workflow

# Use python from the current environment
# Assumes dependencies are installed (e.g., via pip install -e .[dev] from project root)
PYTHON = python

# Define the source directory relative to this Makefile
SRC_DIR = src/
TEST_DIR = tests/
OUTPUT_DIR = output_py_$(shell date +%Y%m%d_%H%M%S)

# Phony targets don't represent files
.PHONY: all run clean test install list_output help

# Default target
all: help

# Run the main analysis workflow
run:
	@echo "--- Running DEM Analysis Workflow ---"
	$(PYTHON) -m src.workflow
	@echo "--- Workflow Finished ---"

# Clean the output directory
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

# Run tests using pytest
# Assumes pytest is run from the project root directory (containing pyproject.toml)
# Adjust if running from lab/GIS5 directory
test:
	@echo "--- Running Tests ---"
	pytest $(TEST_DIR)
	# Alternative if running pytest from lab/GIS5:
	# pytest $(TEST_DIR)
	@echo "--- Testing Complete ---"

# Install dependencies (useful for setup)
# Assumes running from the project root directory
install:
	@echo "--- Installing Dependencies (run from project root) ---"
	pip install -e .[dev]
	@echo "--- Installation Command Shown (run manually from project root if needed) ---"

# List the contents of the output directory
list_output:
	@echo "--- Listing Output Directory Contents (${OUTPUT_DIR}) ---"
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		ls -l $(OUTPUT_DIR); \
	else \
		echo "Output directory not found. Run the workflow first."; \
	fi
	@echo "--- Listing Complete ---"


# Help target to display available commands
help:
	@echo "Available commands:"
	@echo "  make run          - Run the main DEM analysis workflow"
	@echo "  make test         - Run pytest tests (expects to be run from project root)"
	@echo "  make clean        - Remove an output directory"
	@echo "  make list_output  - List the contents of the output directory"
	@echo "  make install      - Show command to install dependencies (run from project root)"
	@echo "  make help         - Show this help message"
