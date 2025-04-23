import logging
import sys
from pathlib import Path
from dask.distributed import Client, LocalCluster
from whitebox import WhiteboxTools

# Local imports
from src.config import AppConfig, settings as app_settings
from src.tasks.features.segments import Segments
from src.tasks.features.heatmap import Heatmap
from src.tasks.features.traffic import Traffic
from src.tasks.features.roads import Roads
from src.tasks.features.elevation import Elevation
from src.tasks.features.cost_distance import CostDistance


logger = logging.getLogger(__name__)  # Define logger earlier


def build_features_task(settings: AppConfig, wbt: WhiteboxTools):
    """
    Main task function to build all features by initializing and running
    individual feature classes.

    Args:
        settings (AppConfig): Application configuration object.
        wbt (WhiteboxTools): Initialized WhiteboxTools object.

    Returns:
        dict: A dictionary containing the generated feature objects or result paths.
              Keys might include 'segments', 'heatmap', 'traffic', 'roads', 'elevation', 'cost_distance'.
    """
    logger.info("--- Start Feature Building Task ---")

    # Initialize Feature Objects
    segments = Segments(settings, wbt)
    heatmap = Heatmap(settings, wbt)
    traffic = Traffic(settings, wbt)
    roads = Roads(settings, wbt)
    elevation = Elevation(settings, wbt)
    # CostDistance requires outputs from others, initialized later

    # Load data for all features first (can happen in parallel if desired, but sequential is simpler)
    logger.info("Loading data for all features...")
    try:
        segments.load_data()
        heatmap.load_data()  # Will log warnings as it's not implemented
        traffic.load_data()
        roads.load_data()
        elevation.load_data()
    except Exception as e:
        logger.error(f"Fatal error during data loading phase: {e}", exc_info=True)
        # Depending on requirements, might want to exit or continue with available data
        # For now, let's log the error and attempt to build with whatever loaded
        # sys.exit(1) # Or exit if loading is critical

    logger.info("Data loading phase complete.")

    # Build features using Dask where applicable (build methods handle dask.delayed)
    logger.info("Building feature layers...")
    # TODO make sure dask gets all tasks into single list for compute()
    # in other words, rewrite this part
    segment_raster_paths = segments.build()  # Returns list of paths
    speed_raster_path = heatmap.build()  # Returns path or None
    traffic_raster_path = traffic.build()  # Returns path or None
    road_vector_paths = roads.build()  # Returns dict of paths
    dem_path, slope_path = elevation.build()  # Returns two paths or None

    # Initialize and build Cost distance (depends on slope)
    cost_distance = CostDistance(
        settings,
        wbt,
        slope_raster_path=Path(slope_path),
        roads_raster_path=Path(road_vector_paths.get("roads_rasterized")), # If roads are rasterized
        speed_raster_path=(
            Path(speed_raster_path) if speed_raster_path else None
        ),
    )

    # Consolidate results, including paths to saved intermediate files
    results = {
        "segments": segments,  # Keep object for potential later use
        "segment_rasters": segment_raster_paths,
        "heatmap": heatmap,
        "speed_raster": speed_raster_path,
        "traffic": traffic,
        "traffic_raster": traffic_raster_path,
        "roads": roads,
        "road_vectors": road_vector_paths,
        "elevation": elevation,
        "dem_raster": dem_path,
        "slope_raster": slope_path,
        "cost_distance": cost_distance,
        # "cost_raster": cost_raster_path,
        # Add paths to prepared vector data as well
        "prepared_segments": segments.output_paths.get("prepared_segments_gpkg"),
        "prepared_activities": heatmap.output_paths.get("prepared_activities_gpkg"),
        "prepared_traffic": traffic.output_paths.get("prepared_traffic_points_gpkg"),
        "prepared_roads": roads.output_paths.get("prepared_roads_gpkg"),
        "prepared_lanes": roads.output_paths.get("prepared_bike_lanes_gpkg"),
        "prepared_roads_no_lanes": roads.output_paths.get(
            "prepared_roads_no_lanes_gpkg"
        ),
        "prepared_contours": elevation.output_paths.get("prepared_contours_gpkg"),
    }

    logger.info("--- Feature Building Task Completed ---")
    # Log generated paths for clarity
    logger.info("Generated outputs (Paths):")
    for key, value in results.items():
        # Log paths, lists of paths, or dicts of paths
        if isinstance(value, Path) or (isinstance(value, str) and Path(value).exists()):
            logger.info(f"  - {key}: {value}")
        elif isinstance(value, list) and all(
            isinstance(item, (str, Path)) for item in value
        ):
            logger.info(f"  - {key}: {[str(p) for p in value]}")
        elif isinstance(value, dict) and all(
            isinstance(item, (str, Path, type(None))) for item in value.values()
        ):
            # Filter out None values before logging dict paths
            valid_paths = {k: str(v) for k, v in value.items() if v}
            if valid_paths:
                logger.info(f"  - {key}: {valid_paths}")
        # Avoid logging entire objects/dataframes
        elif (
            hasattr(value, "__class__")
            and "feature_base" in str(value.__class__).lower()
        ):
            logger.info(f"  - {key}: <{type(value).__name__} object>")
        # Log None values explicitly if they are direct results
        elif value is None and key.endswith(
            ("_path", "_paths", "_raster", "_rasters", "_vector", "_vectors")
        ):
            logger.info(f"  - {key}: None")

    return results


# --- Example Usage ---
if __name__ == "__main__":
    # Basic setup for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("--- Running build_features.py Refactored Standalone Test ---")

    # Use settings loaded from config.py
    settings = app_settings
    # Ensure output directory exists for the test
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using Output Directory: {settings.paths.output_dir}")

    wbt = None
    try:
        logger.info("Initializing WhiteboxTools for test...")
        wbt = WhiteboxTools()
        wbt.set_verbose_mode(settings.processing.wbt_verbose)
        logger.info("WhiteboxTools initialized.")
    except Exception as e:
        logger.error(
            f"Failed to initialize WhiteboxTools: {e}. Cannot proceed with WBT-dependent tests."
        )
        sys.exit(1)  # Exit if WBT fails, as it's needed

    # Initialize Dask client for parallel processing
    # Using LocalCluster for simple standalone execution
    try:
        cluster = LocalCluster()
        client = Client(cluster)
        logger.info(f"Dask client started: {client.dashboard_link}")
    except Exception as e:
        logger.error(f"Failed to start Dask client: {e}")
        client = None  # Allow proceeding without Dask if it fails? Or exit?

    # --- Test build_features_task ---
    try:
        logger.info("--- Testing build_features_task ---")
        # This will run load_data and build for all features defined within it
        task_results = build_features_task(settings, wbt)
        logger.info("--- build_features_task Test Completed ---")
        # Optionally print specific results from the dictionary
        # print("Prepared Segments Path:", task_results.get("prepared_segments"))
        # print("Segment Rasters:", task_results.get("segment_rasters"))

    except Exception as e:
        logger.error(f"Error during build_features_task test: {e}", exc_info=True)

    # Clean up Dask client
    if client:
        try:
            client.close()
            cluster.close()
            logger.info("Dask client and cluster closed.")
        except Exception as e:
            logger.warning(f"Error closing Dask client/cluster: {e}")

    logger.info("--- Standalone Test Finished ---")
