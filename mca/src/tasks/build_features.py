import logging
import sys
from pathlib import Path
import dask # Import dask
from dask.distributed import Client, LocalCluster
from whitebox import WhiteboxTools

# Local imports
from src.config import AppConfig, settings as app_settings
from src.tasks.features.segments import Segments
from src.tasks.features.heatmap import Heatmap
from src.tasks.features.traffic import Traffic
from src.tasks.features.roads import Roads
from src.tasks.features.elevation import Elevation
from tasks.features.cost import CostLayer


logger = logging.getLogger(__name__)  # Define logger earlier


def build_features_task(settings: AppConfig, wbt: WhiteboxTools):
    """
    Main task function to build all features by initializing and running
    individual feature classes. Uses Dask to parallelize independent features.

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
    logger.info("Collecting Dask build tasks for independent features...")
    # Collect delayed tasks from independent features
    # elevation.build() returns a single delayed object representing a tuple
    segments_task = segments.build()
    heatmap_task = heatmap.build()
    traffic_task = traffic.build()
    roads_task = roads.build()
    elevation_task = elevation.build() # This task computes (dem_path, slope_path)

    delayed_tasks = [
        segments_task,
        heatmap_task,
        traffic_task,
        roads_task,
        elevation_task, # Add the single elevation task object
    ]

    logger.info(f"Collected {len(delayed_tasks)} Dask tasks. Computing...")
    # Compute independent tasks in parallel
    # Use scheduler="distributed" if a Dask client is running, otherwise defaults might work
    computed_results = dask.compute(*delayed_tasks, scheduler="distributed")
    logger.info("Independent feature computation complete.")

    # Extract results (order matches delayed_tasks)
    segment_raster_paths = computed_results[0] # List of paths
    speed_raster_path = computed_results[1]    # Path or None
    traffic_raster_path = computed_results[2]  # Path or None
    road_vector_paths = computed_results[3]    # Dict of paths
    # Elevation result is a tuple (dem_path, slope_path) or (None, None)
    elevation_results = computed_results[4]
    dem_path, slope_path = elevation_results if elevation_results else (None, None)

    # --- Build Dependent Features ---
    logger.info("Building dependent feature: Cost Layer...")
    cost_layer = None # Initialize to None
    cost_raster_path = None # Initialize path to None
    cost_raster_delayed = None # Initialize delayed task to None

    roads_vector_input_path = road_vector_paths.get("samferdsel_all") 

    if slope_path and roads_vector_input_path:
        logger.info(f"Inputs found: Slope='{slope_path}', Roads Vector='{roads_vector_input_path}'")
        try:
            # Convert paths to Path objects only if they are not None
            slope_raster_path_obj = Path(slope_path)
            roads_vector_path_obj = Path(roads_vector_input_path)
            speed_raster_path_obj = Path(speed_raster_path) if speed_raster_path else None

            # Initialize CostLayer with VECTOR roads path
            cost_layer = CostLayer(
                settings=settings,
                wbt=wbt,
                slope_raster_path=slope_raster_path_obj,
                roads_vector_path=roads_vector_path_obj, 
                speed_raster_path=speed_raster_path_obj,
            )

            # Get the delayed task for building the cost raster
            cost_raster_delayed = cost_layer.build()

            if cost_raster_delayed is not None:
                logger.info("Cost Layer build task created. Will compute separately.")
                # Compute the cost layer task
                # dask.compute returns a tuple, get the first element
                computed_results = dask.compute(cost_raster_delayed, scheduler="distributed")[0]
                
                # Handle the new dictionary return format
                if computed_results and isinstance(computed_results, dict):
                    # Extract the cost raster path
                    cost_raster_path_str = computed_results.get("cost_raster_path")
                    
                    if cost_raster_path_str and Path(cost_raster_path_str).exists():
                        cost_raster_path = Path(cost_raster_path_str)
                        logger.info(f"Cost Layer build complete. Output: {cost_raster_path}")
                        
                        # Create multi-layer visualization after computation finishes
                        try:
                            logger.info("Creating multi-layer visualization...")
                            slope_path = computed_results.get("slope_raster_path")
                            speed_path = computed_results.get("aligned_speed_path", computed_results.get("speed_raster_path"))
                            
                            visualization_result = cost_layer.create_multi_layer_visualization(
                                cost_raster_path=cost_raster_path,
                                slope_raster_path=Path(slope_path) if slope_path else None,
                                speed_raster_path=Path(speed_path) if speed_path else None
                            )
                            
                            if visualization_result:
                                logger.info(f"Multi-layer visualization created: {visualization_result}")
                        except Exception as vis_e:
                            logger.error(f"Error creating multi-layer visualization: {vis_e}", exc_info=True)
                    else:
                        logger.error("Cost Layer computation finished but output path is invalid or file not found.")
                        cost_raster_path = None # Reset path if computation failed internally
                else:
                    # Handle old return format (string path) for backwards compatibility
                    if computed_results and isinstance(computed_results, str):
                        cost_raster_path = Path(computed_results)
                        if cost_raster_path.exists():
                            logger.info(f"Cost Layer build complete. Output: {cost_raster_path}")
                        else:
                            logger.error("Cost Layer output path is invalid or file not found.")
                            cost_raster_path = None
                    else:
                        logger.error("Cost Layer computation returned unexpected result format.")
                        cost_raster_path = None

        except FileNotFoundError as e:
             logger.error(f"Initialization error for CostLayer: Input file not found - {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to initialize or build Cost Layer: {e}", exc_info=True)
            cost_layer = None # Ensure object is None if init failed
            cost_raster_path = None
    else:
        missing = []
        if not slope_path: missing.append("slope_path")
        if not roads_vector_input_path: 
            missing.append("roads_vector_path (check key in road_vector_paths dict)")
        logger.warning(f"Skipping Cost Layer build because required inputs are missing: {', '.join(missing)}")


    # --- Consolidate Results ---
    logger.info("Consolidating all results...")
    results = {
        # Feature objects
        "segments": segments,
        "heatmap": heatmap,
        "traffic": traffic,
        "roads": roads,
        "elevation": elevation,
        "cost_distance": cost_layer, # Might be None if build failed/skipped
        # Direct outputs from computation
        "segment_rasters": segment_raster_paths, # List of paths
        "speed_raster": speed_raster_path,       # Path or None
        "traffic_raster": traffic_raster_path,     # Path or None
        "road_vectors": road_vector_paths,       # Dict of paths
        "dem_raster": dem_path,                  # Path or None
        "slope_raster": slope_path,                # Path or None
        "cost_raster": cost_raster_path,         # Path or None
        # Paths to prepared vector data (from feature object properties)
        "prepared_segments": segments.output_paths.get("prepared_segments_gpkg"),
        "prepared_activities": heatmap.output_paths.get("prepared_activities_gpkg"),
        "prepared_traffic": traffic.output_paths.get("prepared_traffic_points_gpkg"),
        "prepared_roads": roads.output_paths.get("prepared_roads_gpkg"),
        "prepared_lanes": roads.output_paths.get("prepared_bike_lanes_gpkg"),
        "prepared_roads_no_lanes": roads.output_paths.get("prepared_roads_no_lanes_gpkg"),
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
    client = None
    cluster = None
    try:
        # Using LocalCluster for simple standalone execution
        # Adjust n_workers and threads_per_worker based on your system
        cluster = LocalCluster(n_workers=settings.processing.dask_workers, threads_per_worker=1)
        client = Client(cluster)
        logger.info(f"Dask client started: {client.dashboard_link}")
    except Exception as e:
        logger.error(f"Failed to start Dask client: {e}. Will attempt sequential execution.")
        # If Dask fails, we might want to fall back or exit depending on requirements.
        # For now, the dask.compute call might default to a synchronous scheduler.
        # sys.exit(1) # Optionally exit if Dask is critical

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

    # Clean up Dask client and cluster
    if client:
        try:
            client.close()
            logger.info("Dask client closed.")
        except Exception as e:
            logger.warning(f"Error closing Dask client: {e}")
    if cluster:
        try:
            cluster.close()
            logger.info("Dask cluster closed.")
        except Exception as e:
            logger.warning(f"Error closing Dask cluster: {e}")

    logger.info("--- Standalone Test Finished ---")
