import sys
import time
import logging
from whitebox import WhiteboxTools

# Ensure the src directory is in the Python path if running as a script
# This might not be needed if run as a module or with `python -m lab.GIS5.src.workflow`
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.config import settings
    from src.utils import setup_output_dir
    from src.tasks.load_data import load_and_prepare_data
    from src.tasks.generate_dems import generate_dems
    from src.tasks.quality_assessment import assess_dem_quality
    from src.tasks.derive_products import generate_derived_products
except ImportError:
    # Fallback for running the script directly from the src directory
    from config import settings
    from utils import setup_output_dir
    from tasks.load_data import load_and_prepare_data
    from tasks.generate_dems import generate_dems
    from tasks.quality_assessment import assess_dem_quality
    from tasks.derive_products import generate_derived_products


# --- Logging Setup ---
# Configure logging to output to console with a specific format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Get a logger for this module
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
        # Optional: Set working directory if needed, though outputs are explicitly pathed
        # wbt.work_dir = str(settings.paths.output_dir / "wbt_temp")
        # logger.info(f"  - WhiteboxTools working directory set to: {wbt.work_dir}")
        wbt.set_verbose_mode(settings.processing.wbt_verbose)
    except Exception as e:
        logger.error(f"Error initializing WhiteboxTools: {e}")
        logger.error(
            "Ensure WhiteboxTools executable is correctly installed and in the system PATH."
        )
        sys.exit(1)

    # --- Setup Output Directory ---
    # Config handles creation, but setup_output_dir ensures it's clean
    logger.info(f"Setting up output directory: {settings.paths.output_dir}")
    setup_output_dir(settings.paths.output_dir)

    try:
        # --- Task 1: Load Data ---
        logger.info("--- Starting Task 1: Load and Prepare Data ---")
        loaded_gdfs, common_extent, common_crs, contour_elev_field, point_elev_field = (
            load_and_prepare_data(settings)
        )
        logger.info("--- Finished Task 1: Load and Prepare Data ---")

        # Get necessary paths for subsequent tasks
        output_dir = settings.paths.output_dir
        output_files = settings.output_files
        contour_shp_path = output_files.get_full_path("contour_shp", output_dir)
        points_shp_path = output_files.get_full_path("points_shp", output_dir)
        # Get the path to the river shapefile saved by load_data
        river_shp_path = output_files.get_full_path("river_shp", output_dir)

        # --- Task 2: Generate DEMs ---
        logger.info("--- Starting Task 2: Generate DEMs ---")
        # Pass the river shapefile path for potential stream burning
        generate_dems(
            settings=settings,
            wbt=wbt,
            contour_shp_path=contour_shp_path,
            contour_elev_field=contour_elev_field,
            river_shp_path=river_shp_path,  # Pass the river path
            stream_extract_threshold=(
                settings.processing.stream_extract_threshold
                if settings.processing.enable_stream_burning
                else None
            ),
        )
        logger.info("--- Finished Task 2: Generate DEMs ---")

        # Get DEM paths for next tasks
        dem_interp_path = output_files.get_full_path("dem_interpolated_tif", output_dir)
        dem_topo_path = output_files.get_full_path("dem_topo_tif", output_dir)
        # Get the path for the INPUT ArcGIS Pro TopoToRaster DEM from PathsConfig
        dem_toporaster_all_path = (
            settings.paths.toporaster_all_input_tif
        )  # Use the direct path from settings
        # Get the path for the generated stream burn DEM
        dem_stream_burn_path = output_files.get_full_path(
            "dem_stream_burned_tif", output_dir
        )

        # --- Task 3: Quality Assessment ---
        logger.info("--- Starting Task 3: Quality Assessment ---")
        assess_dem_quality(
            settings=settings,
            wbt=wbt,
            points_shp_path=points_shp_path,
            dem_interp_path=dem_interp_path,
            dem_topo_path=dem_topo_path,
            dem_toporaster_all_path=dem_toporaster_all_path,
            dem_stream_burn_path=dem_stream_burn_path,  # Added stream burn path
            point_elev_field=point_elev_field,
        )
        logger.info("--- Finished Task 3: Quality Assessment ---")

        # --- Task 4: Generate Derived Products ---
        logger.info("--- Starting Task 4: Generate Derived Products ---")
        # Call generate_derived_products, passing all relevant DEM paths
        # Transect creation and profile analysis are now handled within this function
        generate_derived_products(
            settings=settings,
            wbt=wbt,
            dem_interp_path=dem_interp_path,
            dem_topo_path=dem_topo_path,
            dem_toporaster_all_path=dem_toporaster_all_path,  # Pass ANUDEM path
            dem_stream_burn_path=dem_stream_burn_path,  # Pass Stream Burn path
            common_crs=common_crs,  # Pass the common CRS
            common_extent=common_extent,  # Pass the common extent
        )
        logger.info("--- Finished Task 4: Generate Derived Products ---")

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


if __name__ == "__main__":
    main()
