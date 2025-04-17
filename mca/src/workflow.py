import logging
import sys
import time
from whitebox import WhiteboxTools

from src.config import settings
from src.utils import setup_output_dir


logger = logging.getLogger(__name__)


def main():
    """Runs the complete DEM analysis workflow."""
    start_time = time.time()
    logger.info("--- Starting DEM Analysis Workflow ---")

    # --- Initialization ---
    logger.info("Initializing WhiteboxTools...")
    try:
        wbt = WhiteboxTools()
        logger.info(f"  - WhiteboxTools version: {wbt.version()}")
        wbt.set_verbose_mode(settings.processing.wbt_verbose)
    except Exception as e:
        logger.error(f"Error initializing WhiteboxTools: {e}")
        logger.error(
            "Ensure WhiteboxTools executable is correctly installed and in the system PATH."
        )
        sys.exit(1)

    # --- Setup Output Directory ---
    # Ensure the output directory exists (NOTE removes previous data!)
    logger.info(f"Setting up output directory: {settings.paths.output_dir}")
    setup_output_dir(settings.paths.output_dir)

    try:
        # --- Task 1: Load Data (Main DEM Workflow) ---
        logger.info("--- Starting Task 1: Load Data ---")

        # --- Task 2: Generate Feature Layers ---
        logger.info("--- Starting Task 2: Generate Feature Layers ---")

        # --- Task 3: Geoprocess ---

        # --- Task 4: Assess ---

        # --- Task 5: Plot ---

    except FileNotFoundError as fnf_error:
        logger.error("--- Workflow Halted: Required file not found ---")
        logger.error(f"Error: {fnf_error}")
        sys.exit(1)
    except ValueError as val_error:
        logger.error("--- Workflow Halted: Data error ---")
        logger.error(f"Error: {val_error}")
        sys.exit(1)
    except Exception as e:
        logger.error("--- Workflow Halted: An unexpected error occurred ---")
        # Use exc_info=True to include traceback information automatically
        logger.error(f"Error Details: {e}", exc_info=True)
        sys.exit(1)

    end_time = time.time()
    logger.info("--- Workflow Script Finished ---")
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
