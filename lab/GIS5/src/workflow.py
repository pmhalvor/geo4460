import sys
import time
import logging
from whitebox import WhiteboxTools

try:
    from src.config import settings
    from src.utils import setup_output_dir
    from src.tasks.load_data import load_and_prepare_data
    from src.tasks.generate_dems import generate_dems
    from src.tasks.quality_assessment import assess_dem_quality
    from lab.GIS5.src.tasks.further_analysis import generate_further_analysis
    from src.tasks.prepare_raw_data import prepare_raw_data
except ImportError:
    from config import settings
    from utils import setup_output_dir

    from tasks.load_data import load_and_prepare_data
    from tasks.generate_dems import generate_dems
    from tasks.quality_assessment import assess_dem_quality
    from lab.GIS5.src.tasks.further_analysis import generate_further_analysis
    from tasks.prepare_raw_data import prepare_raw_data


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
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
        # --- Optional Task: Prepare Raw Data ---
        if settings.processing.run_prepare_raw_data:
            if settings.paths.input_raw_gdb:
                logger.info("--- Starting Optional Task: Prepare Raw Data ---")
                prepare_raw_data(settings)
                logger.info("--- Finished Optional Task: Prepare Raw Data ---")
            else:
                logger.warning("--- Skipping Optional Task: Prepare Raw Data ---")
                logger.warning(
                    "Reason: 'run_prepare_raw_data' is True, but 'paths.input_raw_gdb' (or equivalent) is not configured."
                )
        else:
            logger.info(
                "--- Skipping Optional Task: Prepare Raw Data (disabled in config) ---"
            )

        # --- Task 1: Load Data (Main DEM Workflow) ---
        logger.info(
            "--- Starting Task 1: Load and Prepare Data (Main DEM Workflow) ---"
        )
        loaded_gdfs, common_extent, common_crs, contour_elev_field, point_elev_field = (
            load_and_prepare_data(settings)
        )
        logger.info(
            "--- Finished Task 1: Load and Prepare Data (Main DEM Workflow) ---"
        )

        # Get necessary paths for subsequent tasks (Main DEM Workflow)
        output_dir = settings.paths.output_dir
        output_files = settings.output_files
        contour_shp_path = output_files.get_full_path("contour_shp", output_dir)
        points_shp_path = output_files.get_full_path("points_shp", output_dir)
        river_shp_path = output_files.get_full_path("river_shp", output_dir)

        # --- Task 2: Generate DEMs ---
        logger.info("--- Starting Task 2: Generate DEMs ---")
        # Pass both contour and elevation points paths and contour field
        generate_dems(
            settings=settings,
            wbt=wbt,
            contour_shp_path=contour_shp_path,
            contour_elev_field=contour_elev_field,
            elevation_points_shp_path=points_shp_path,
            river_shp_path=river_shp_path,
            stream_extract_threshold=(
                settings.processing.stream_extract_threshold
                if settings.processing.enable_stream_burning
                else None
            ),
        )
        logger.info("--- Finished Task 2: Generate DEMs ---")

        dem_interp_contour_path = output_files.get_full_path(
            "dem_interpolated_contour_tif", output_dir
        )
        dem_topo_contour_path = output_files.get_full_path(
            "dem_topo_contour_tif", output_dir
        )
        dem_interp_points_path = output_files.get_full_path(
            "dem_interpolated_points_tif", output_dir
        )
        dem_topo_points_path = output_files.get_full_path(
            "dem_topo_points_tif", output_dir
        )
        dem_stream_burn_path = output_files.get_full_path(
            "dem_stream_burned_tif", output_dir
        )
        # Get the path for the INPUT ArcGIS Pro TopoToRaster DEM from PathsConfig
        dem_toporaster_all_path = (
            settings.paths.toporaster_all_input_tif
        )  # Keep for comparison

        # --- Task 3: Quality Assessment ---
        logger.info("--- Starting Task 3: Quality Assessment ---")
        assess_dem_quality(
            settings=settings,
            wbt=wbt,
            points_shp_path=points_shp_path,  # Points file now has 'VALUE' field
            dem_interp_contour_path=dem_interp_contour_path,
            dem_topo_contour_path=dem_topo_contour_path,
            dem_interp_points_path=dem_interp_points_path,
            dem_topo_points_path=dem_topo_points_path,
            dem_stream_burn_path=dem_stream_burn_path,
            dem_toporaster_all_path=dem_toporaster_all_path,
            point_elev_field="VALUE",  # Pass the standardized field name
        )
        logger.info("--- Finished Task 3: Quality Assessment ---")

        # --- Task 4: Generate Derived Products ---
        logger.info("--- Starting Task 4: Generate Derived Products ---")
        generate_further_analysis(
            settings=settings,
            wbt=wbt,
            dem_interp_contour_path=dem_interp_contour_path,
            dem_topo_contour_path=dem_topo_contour_path,
            dem_interp_points_path=dem_interp_points_path,
            dem_topo_points_path=dem_topo_points_path,
            dem_stream_burn_path=dem_stream_burn_path,
            dem_toporaster_all_path=dem_toporaster_all_path,
            common_crs=common_crs,
            common_extent=common_extent,
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
