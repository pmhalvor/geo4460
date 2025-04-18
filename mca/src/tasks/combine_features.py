import logging
from pathlib import Path
import geopandas as gpd
import rasterio
import numpy as np
from whitebox import WhiteboxTools

# Local imports
from src.config import AppConfig
from src.utils import (
    load_vector_data,
    save_vector_data,
    load_raster_data,
    save_raster_data,
)

logger = logging.getLogger(__name__)


# --- Docstring from original file (for reference) ---
"""
Combine Features Steps:

3. Combine features:
    1. Overlay A: Popular segments w/o biking lanes
        1. (Iterative preprocessing step) Roads - bike_lanes -> find segments around these spots
        2. All segment lines layer - bike_lanes (keep all segments with part outside of bike lanes)
    2. Overlay B: Popular segments w/o biking lanes w/ high avg speeds
        1. Overlay A + average speed raster
    3. Overlay C: Popular segments w/o biking lanes w/ high avg speeds + high traffic
        1. Overlay B + traffic buffer raster
    4. Overlay D: Popular segments w/o biking lanes w/ high avg speeds + high traffic + high cost
        1. Overlay C + cost function raster

(Cost now handled in build_features_task)

Final map:
    Output from Overlay D
"""


# Helper function (consider moving to utils if generally useful)
def _get_full_output_path(settings: AppConfig, key: str) -> Path:
    """Helper to get a full output path from settings."""
    filename = getattr(settings.output_files, key)
    return settings.paths.output_dir / filename


# --- Overlay Functions ---


def build_overlay_a(
    settings: AppConfig,
    wbt: WhiteboxTools,
    prepared_segments_path: Path,
    prepared_roads_no_lanes_path: Path,
    # segment_popularity_raster_paths: list[Path] # Or use vector attributes
) -> Path | None:
    """
    Overlay A: Identifies popular segments that are (at least partially)
               outside the existing bike lane network.

    Current approach: Vector overlay. Find segments intersecting roads_no_lanes.
    Alternative: Raster overlay using popularity rasters and rasterized roads_no_lanes.

    Args:
        settings: Application config.
        wbt: WhiteboxTools instance.
        prepared_segments_path: Path to the prepared segments GeoPackage.
        prepared_roads_no_lanes_path: Path to the prepared roads_no_lanes GeoPackage.

    Returns:
        Path to the generated Overlay A GeoPackage, or None if failed.
    """
    logger.info("--- Building Overlay A: Popular Segments w/o Bike Lanes ---")
    output_path = _get_full_output_path(settings, "overlay_a_gpkg")

    if not prepared_segments_path or not prepared_segments_path.exists():
        logger.error("Prepared segments file not found. Cannot build Overlay A.")
        return None
    if not prepared_roads_no_lanes_path or not prepared_roads_no_lanes_path.exists():
        logger.error("Prepared roads_no_lanes file not found. Cannot build Overlay A.")
        return None

    try:
        segments_gdf = load_vector_data(prepared_segments_path)
        roads_no_lanes_gdf = load_vector_data(prepared_roads_no_lanes_path)

        # Ensure CRS match before spatial operations
        if segments_gdf.crs != roads_no_lanes_gdf.crs:
            logger.warning(
                f"CRS mismatch: Segments ({segments_gdf.crs}) vs RoadsNoLanes ({roads_no_lanes_gdf.crs}). Reprojecting RoadsNoLanes."
            )
            # Assume segments_gdf has the target CRS from build_features
            roads_no_lanes_gdf = roads_no_lanes_gdf.to_crs(segments_gdf.crs)

        # Perform spatial intersection (find segments that touch roads_no_lanes)
        # 'inner' join keeps only intersecting features.
        # predicate='intersects' is generally suitable.
        overlay_a_gdf = gpd.sjoin(
            segments_gdf, roads_no_lanes_gdf, how="inner", predicate="intersects"
        )

        # The result might contain duplicate segment IDs if a segment intersects multiple
        # 'roads_no_lanes' features. Drop duplicates based on segment ID.
        # TODO: Verify segment ID field name from config
        segment_id_field = settings.input_data.segment_id_field
        if segment_id_field in overlay_a_gdf.columns:
            overlay_a_gdf = overlay_a_gdf.drop_duplicates(subset=[segment_id_field])
            logger.info(
                f"Found {len(overlay_a_gdf)} unique segments intersecting roads without lanes."
            )
        else:
            logger.warning(
                f"Segment ID field '{segment_id_field}' not found in sjoin result. Cannot drop duplicates."
            )

        # TODO: Add filtering based on popularity metrics if needed
        # e.g., filter overlay_a_gdf based on pre-calculated popularity columns

        save_vector_data(overlay_a_gdf, output_path, driver="GPKG")
        logger.info(f"Overlay A saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error building Overlay A: {e}", exc_info=True)
        return None


def build_overlay_b(
    settings: AppConfig,
    wbt: WhiteboxTools,
    overlay_a_path: Path,
    average_speed_raster_path: Path,
) -> Path | None:
    """
    Overlay B: Filters Overlay A segments based on average speed.

    Requires raster sampling or zonal statistics.

    Args:
        settings: Application config.
        wbt: WhiteboxTools instance.
        overlay_a_path: Path to the Overlay A GeoPackage.
        average_speed_raster_path: Path to the average speed raster.

    Returns:
        Path to the generated Overlay B GeoPackage, or None if failed.
    """
    logger.info("--- Building Overlay B: Adding Speed Filter ---")
    output_path = _get_full_output_path(settings, "overlay_b_gpkg")

    if not overlay_a_path or not overlay_a_path.exists():
        logger.error("Overlay A file not found. Cannot build Overlay B.")
        return None
    if not average_speed_raster_path or not average_speed_raster_path.exists():
        logger.warning(
            "Average speed raster not found. Skipping Overlay B speed filter."
        )
        # Copy Overlay A as Overlay B if speed raster is missing
        try:
            overlay_a_gdf = load_vector_data(overlay_a_path)
            save_vector_data(overlay_a_gdf, output_path, driver="GPKG")
            logger.info(
                f"Copied Overlay A to Overlay B (no speed filter applied): {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Error copying Overlay A to B: {e}")
            return None

    try:
        overlay_a_gdf = load_vector_data(overlay_a_path)

        # TODO: Implement raster sampling/zonal stats
        # Option 1: Sample raster values at segment centroids or along lines
        # Option 2: Use WBT ZonalStatistics or similar tool
        logger.warning("Overlay B logic (raster sampling/zonal stats) not implemented.")
        # Placeholder: Keep all segments from Overlay A for now
        overlay_b_gdf = overlay_a_gdf

        # Example filtering (if stats were calculated and added to gdf):
        # speed_threshold = settings.processing.overlay_speed_threshold
        # if speed_threshold is not None:
        #     overlay_b_gdf = overlay_a_gdf[overlay_a_gdf['avg_speed_stat'] > speed_threshold]
        #     logger.info(f"Filtered Overlay B based on speed threshold > {speed_threshold}")

        save_vector_data(overlay_b_gdf, output_path, driver="GPKG")
        logger.info(f"Overlay B saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error building Overlay B: {e}", exc_info=True)
        return None


def build_overlay_c(
    settings: AppConfig,
    wbt: WhiteboxTools,
    overlay_b_path: Path,
    traffic_density_raster_path: Path,
) -> Path | None:
    """
    Overlay C: Filters Overlay B segments based on traffic density.

    Requires raster sampling or zonal statistics.

    Args:
        settings: Application config.
        wbt: WhiteboxTools instance.
        overlay_b_path: Path to the Overlay B GeoPackage.
        traffic_density_raster_path: Path to the traffic density raster.

    Returns:
        Path to the generated Overlay C GeoPackage, or None if failed.
    """
    logger.info("--- Building Overlay C: Adding Traffic Filter ---")
    output_path = _get_full_output_path(settings, "overlay_c_gpkg")

    if not overlay_b_path or not overlay_b_path.exists():
        logger.error("Overlay B file not found. Cannot build Overlay C.")
        return None
    if not traffic_density_raster_path or not traffic_density_raster_path.exists():
        logger.warning(
            "Traffic density raster not found. Skipping Overlay C traffic filter."
        )
        # Copy Overlay B as Overlay C
        try:
            overlay_b_gdf = load_vector_data(overlay_b_path)
            save_vector_data(overlay_b_gdf, output_path, driver="GPKG")
            logger.info(
                f"Copied Overlay B to Overlay C (no traffic filter applied): {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Error copying Overlay B to C: {e}")
            return None

    try:
        overlay_b_gdf = load_vector_data(overlay_b_path)

        # TODO: Implement raster sampling/zonal stats for traffic
        logger.warning("Overlay C logic (raster sampling/zonal stats) not implemented.")
        # Placeholder: Keep all segments from Overlay B
        overlay_c_gdf = overlay_b_gdf

        # Example filtering:
        # traffic_threshold = settings.processing.overlay_traffic_threshold
        # if traffic_threshold is not None:
        #     overlay_c_gdf = overlay_b_gdf[overlay_b_gdf['avg_traffic_stat'] > traffic_threshold]
        #     logger.info(f"Filtered Overlay C based on traffic threshold > {traffic_threshold}")

        save_vector_data(overlay_c_gdf, output_path, driver="GPKG")
        logger.info(f"Overlay C saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error building Overlay C: {e}", exc_info=True)
        return None


def build_overlay_d(
    settings: AppConfig,
    wbt: WhiteboxTools,
    overlay_c_path: Path,
    cost_raster_path: Path,
) -> Path | None:
    """
    Overlay D: Filters Overlay C segments based on cost function.

    Requires raster sampling or zonal statistics. This is the final
    recommendation layer before evaluation ranking.

    Args:
        settings: Application config.
        wbt: WhiteboxTools instance.
        overlay_c_path: Path to the Overlay C GeoPackage.
        cost_raster_path: Path to the cost function raster.

    Returns:
        Path to the generated Overlay D GeoPackage, or None if failed.
    """
    logger.info("--- Building Overlay D: Adding Cost Filter ---")
    output_path = _get_full_output_path(settings, "overlay_d_gpkg")

    if not overlay_c_path or not overlay_c_path.exists():
        logger.error("Overlay C file not found. Cannot build Overlay D.")
        return None
    if not cost_raster_path or not cost_raster_path.exists():
        logger.warning("Cost raster not found. Skipping Overlay D cost filter.")
        # Copy Overlay C as Overlay D
        try:
            overlay_c_gdf = load_vector_data(overlay_c_path)
            save_vector_data(overlay_c_gdf, output_path, driver="GPKG")
            logger.info(
                f"Copied Overlay C to Overlay D (no cost filter applied): {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Error copying Overlay C to D: {e}")
            return None

    try:
        overlay_c_gdf = load_vector_data(overlay_c_path)

        # TODO: Implement raster sampling/zonal stats for cost
        logger.warning("Overlay D logic (raster sampling/zonal stats) not implemented.")
        # Placeholder: Keep all segments from Overlay C
        overlay_d_gdf = overlay_c_gdf

        # Example filtering (lower cost is better):
        # cost_threshold = settings.processing.overlay_cost_threshold
        # if cost_threshold is not None:
        #     overlay_d_gdf = overlay_c_gdf[overlay_c_gdf['avg_cost_stat'] < cost_threshold]
        #     logger.info(f"Filtered Overlay D based on cost threshold < {cost_threshold}")
        # Alternatively, keep the cost stat column for ranking in evaluation step.

        save_vector_data(overlay_d_gdf, output_path, driver="GPKG")
        logger.info(f"Overlay D saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error building Overlay D: {e}", exc_info=True)
        return None


# --- Main Task Function ---


def combine_features_task(
    feature_results: dict, settings: AppConfig, wbt: WhiteboxTools
):
    """
    Executes the feature combination (overlay) workflow.

    Args:
        feature_results (dict): Dictionary containing results from build_features_task.
                                Expected keys include paths to prepared vector data
                                and generated rasters.
        settings (AppConfig): Application configuration object.
        wbt (WhiteboxTools): Initialized WhiteboxTools object.

    Returns:
        dict: Dictionary containing paths to the generated overlay files.
    """
    logger.info("--- Starting Task 2: Combine Features ---")

    # Extract necessary paths from feature_results
    prepared_segments = feature_results.get("prepared_segments")
    prepared_roads_no_lanes = feature_results.get("prepared_roads_no_lanes")
    avg_speed_raster = feature_results.get("speed_raster")
    traffic_raster = feature_results.get("traffic_raster")
    cost_raster = feature_results.get("cost_raster")

    # --- Execute Overlays Sequentially ---
    overlay_a_result_path = build_overlay_a(
        settings, wbt, prepared_segments, prepared_roads_no_lanes
    )

    overlay_b_result_path = None
    if overlay_a_result_path:
        overlay_b_result_path = build_overlay_b(
            settings, wbt, overlay_a_result_path, avg_speed_raster
        )

    overlay_c_result_path = None
    if overlay_b_result_path:
        overlay_c_result_path = build_overlay_c(
            settings, wbt, overlay_b_result_path, traffic_raster
        )

    overlay_d_result_path = None
    if overlay_c_result_path:
        overlay_d_result_path = build_overlay_d(
            settings, wbt, overlay_c_result_path, cost_raster
        )

    combined_results = {
        "overlay_a": overlay_a_result_path,
        "overlay_b": overlay_b_result_path,
        "overlay_c": overlay_c_result_path,
        "overlay_d": overlay_d_result_path,  # Final layer before evaluation ranking
    }

    logger.info("--- Task 2: Combine Features Completed ---")
    logger.info("Generated overlay outputs:")
    for key, value in combined_results.items():
        if value:
            logger.info(f"  - {key}: {value}")
        else:
            logger.warning(f"  - {key}: Not generated or failed.")

    return combined_results
