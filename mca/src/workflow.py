import logging
import sys
import time

from dask.distributed import Client, LocalCluster
from whitebox import WhiteboxTools

# Local application/library specific imports
from src.config import settings
from src.utils import setup_output_dir
from src.tasks.build_features import build_features_task
from src.tasks.combine_features import combine_features_task # Import the combine task
from src.tasks.evaluate import EvaluateTask # Import the evaluate task


logger = logging.getLogger(__name__)


def main():
    """Runs the complete DEM analysis workflow."""
    start_time = time.time()
    logger.info("--- Starting DEM Analysis Workflow ---")

    # --- Initialization ---
    # Setup logging (consider moving to a dedicated setup function if complex)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],  # Add FileHandler if needed
    )
    logger.info("--- Starting MCA Workflow ---")

    cluster = LocalCluster(n_workers=settings.processing.dask_workers, threads_per_worker=1)
    client = Client(cluster)

    # Initialize WhiteboxTools
    wbt = None  # Initialize wbt to None
    try:
        logger.info("Initializing WhiteboxTools...")
        wbt = WhiteboxTools()
        logger.info(f"  - WhiteboxTools version: {wbt.version()}")
        # Set verbosity based on config
        wbt.set_verbose_mode(settings.processing.wbt_verbose)
        # Set working directory for WBT if needed (optional)
        # wbt.set_working_dir(str(settings.paths.output_dir))
        logger.info("WhiteboxTools initialized successfully.")
    except Exception as e:
        logger.error(f"Fatal Error initializing WhiteboxTools: {e}")
        logger.error(
            "Ensure WhiteboxTools executable is correctly installed and in the system PATH."
        )
        sys.exit(1)

    # --- Setup Output Directory ---
    # Setup Output Directory
    logger.info(f"Setting up output directory: {settings.paths.output_dir}")
    try:
        setup_output_dir(settings.paths.output_dir)
    except Exception as e:
        logger.error(f"Fatal Error setting up output directory: {e}")
        sys.exit(1)

    # --- Main Workflow Execution ---
    try:
        # --- Task 1: Build Features ---
        # This task loads raw data and generates the primary feature layers/rasters
        logger.info("--- Starting Task 1: Build Features ---")
        # Note: build_features_task needs access to settings for paths and wbt for tools
        # We will need to modify its signature later.
        # For now, assuming it takes settings and wbt
        feature_results = build_features_task(settings=settings, wbt=wbt)
        # Unpack results if needed, e.g.:
        # segments_obj, heatmap_obj, traffic_obj, roads_obj, elevation_obj, build_results = feature_results
        logger.info("--- Task 1: Build Features Completed ---")

        # --- Task 2: Combine Features (Overlays) ---
        logger.info("--- Starting Task 2: Combine Features ---")
        # This task takes outputs from Task 1 and performs overlays
        combined_results = combine_features_task(feature_results, settings=settings, wbt=wbt)
        logger.info("--- Task 2: Combine Features Completed ---")
        # Log the paths of the generated overlays
        logger.info("Combined Feature Results (Overlay Paths):")
        for key, path in combined_results.items():
            if path:
                logger.info(f"  - {key}: {path}")
            else:
                logger.warning(f"  - {key}: Not generated or failed.")


        # --- Task 3: Evaluate Results ---
        logger.info("--- Starting Task 3: Evaluate Results ---")
        # This task generates visualizations, ranks segments, and saves metadata
        eval_task = EvaluateTask(
            settings=settings,
            feature_outputs=feature_results,
            combined_outputs=combined_results
        )
        evaluation_results = eval_task.run()
        logger.info("--- Task 3: Evaluate Results Completed ---")
        # Log the paths of the generated evaluation files
        logger.info("Evaluation Task Results (Output Paths):")
        for key, path in evaluation_results.items():
            if path:
                logger.info(f"  - {key}: {path}")
            else:
                logger.warning(f"  - {key}: Not generated or failed.")


        # --- Error Handling ---
        # Specific errors related to file operations or data processing
    except FileNotFoundError as fnf_error:
        logger.error(
            f"--- Workflow Halted: Required file not found --- \nError: {fnf_error}",
            exc_info=True,
        )
        sys.exit(1)
    except ValueError as val_error:
        logger.error(
            f"--- Workflow Halted: Data or Configuration error --- \nError: {val_error}",
            exc_info=True,
        )
        sys.exit(1)
    except ImportError as imp_error:
        logger.error(
            f"--- Workflow Halted: Missing dependency --- \nError: {imp_error}",
            exc_info=True,
        )
        logger.error(
            "Please ensure all required packages (geopandas, rasterio, whitebox, dask, etc.) are installed."
        )
        sys.exit(1)
    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.error(
            "--- Workflow Halted: An unexpected error occurred ---", exc_info=True
        )
        logger.error(f"Error: {e}")
        sys.exit(1)

    # --- Cleanup ---
    # Close Dask client and cluster
    client.close()
    cluster.close()
    logger.info("Dask client and cluster closed.")

    # --- Workflow Completion ---
    end_time = time.time()
    logger.info(
        f"--- MCA Workflow Finished Successfully in {end_time - start_time:.2f} seconds ---"
    )


if __name__ == "__main__":
    main()
