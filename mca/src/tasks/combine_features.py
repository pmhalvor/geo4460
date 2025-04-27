import logging
from pathlib import Path
import geopandas as gpd
import rasterio
import rasterio.sample
import numpy as np
import pandas as pd
from whitebox import WhiteboxTools

# Local imports
from src.config import AppConfig, settings as app_settings # Import settings for main block
from src.utils import (
    load_vector_data,
    save_vector_data,
    load_raster_data,
    save_raster_data,
    reproject_gdf,
    display_overlay_folium_map, 
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

        # Filter based on popularity threshold from config
        # TODO: Verify the actual normalized popularity column name exists in segments_gdf
        popularity_col = "popularity_norm" # Placeholder name
        popularity_threshold = settings.processing.overlay_popularity_threshold
        if popularity_threshold is not None and popularity_col in overlay_a_gdf.columns:
            initial_count_pop = len(overlay_a_gdf)
            # Keep segments where popularity is ABOVE threshold OR popularity is NaN
            overlay_a_gdf = overlay_a_gdf[
                (overlay_a_gdf[popularity_col] > popularity_threshold) | overlay_a_gdf[popularity_col].isna()
            ]
            filtered_count_pop = len(overlay_a_gdf)
            logger.info(
                f"Filtered Overlay A based on {popularity_col} > {popularity_threshold}. "
                f"Retained {filtered_count_pop}/{initial_count_pop} segments (including NaN popularity)."
            )
        elif popularity_threshold is not None:
            logger.warning(f"Popularity threshold ({popularity_threshold}) defined, but column '{popularity_col}' not found for filtering.")


        if overlay_a_gdf.empty:
            logger.warning("Overlay A resulted in an empty GeoDataFrame after spatial join and popularity filtering.")
            # Save empty file to indicate completion but no results
            save_vector_data(overlay_a_gdf, output_path, driver="GPKG")
            return output_path # Return path to empty file

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

    overlay_a_gdf = load_vector_data(overlay_a_path)
    if overlay_a_gdf.empty:
        logger.warning("Overlay A input is empty. Skipping Overlay B.")
        save_vector_data(overlay_a_gdf, output_path, driver="GPKG") # Save empty file
        return output_path

    # Sample raster values along the segment lines
    stat_col_name = "avg_speed"
    # Removed try...except block around sampling
    overlay_b_gdf = _sample_raster_along_lines(
        overlay_a_gdf, average_speed_raster_path, stat_col_name, stat="mean"
    )

    # Filter based on speed threshold from config
    speed_threshold = settings.processing.overlay_speed_threshold
    if speed_threshold is not None and stat_col_name in overlay_b_gdf.columns:
        initial_count_speed = len(overlay_b_gdf)
        # Keep segments where speed is ABOVE threshold OR speed is NaN
        overlay_b_gdf = overlay_b_gdf[
            (overlay_b_gdf[stat_col_name] > speed_threshold) | overlay_b_gdf[stat_col_name].isna()
        ]
        filtered_count_speed = len(overlay_b_gdf)
        logger.info(
            f"Filtered Overlay B based on {stat_col_name} > {speed_threshold} m/s. "
            f"Retained {filtered_count_speed}/{initial_count_speed} segments (including NaN speeds)."
        )
    elif speed_threshold is not None:
         logger.warning(f"Speed threshold ({speed_threshold} m/s) defined, but column '{stat_col_name}' not found for filtering.")


    if overlay_b_gdf.empty:
        logger.warning("Overlay B resulted in an empty GeoDataFrame after filtering.")

    save_vector_data(overlay_b_gdf, output_path, driver="GPKG")
    logger.info(f"Overlay B saved to: {output_path}")
    return output_path


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

    overlay_b_gdf = load_vector_data(overlay_b_path)
    if overlay_b_gdf.empty:
        logger.warning("Overlay B input is empty. Skipping Overlay C.")
        save_vector_data(overlay_b_gdf, output_path, driver="GPKG") # Save empty file
        return output_path

    # Sample traffic raster values along the segment lines
    stat_col_name = "avg_traffic"
    # Removed try...except block around sampling
    overlay_c_gdf = _sample_raster_along_lines(
        overlay_b_gdf, traffic_density_raster_path, stat_col_name, stat="mean"
    )

    # Filter based on traffic threshold from config
    # Note: Traffic density interpretation depends on how the raster was created (e.g., vehicles/hour, total count)
    traffic_threshold = settings.processing.overlay_traffic_threshold
    if traffic_threshold is not None and stat_col_name in overlay_c_gdf.columns:
        initial_count_traffic = len(overlay_c_gdf)
        # Keep segments where traffic is ABOVE threshold OR traffic is NaN
        overlay_c_gdf = overlay_c_gdf[
            (overlay_c_gdf[stat_col_name] > traffic_threshold) | overlay_c_gdf[stat_col_name].isna()
        ]
        filtered_count_traffic = len(overlay_c_gdf)
        logger.info(
            f"Filtered Overlay C based on {stat_col_name} > {traffic_threshold}. "
            f"Retained {filtered_count_traffic}/{initial_count_traffic} segments (including NaN traffic)."
        )
    elif traffic_threshold is not None:
         logger.warning(f"Traffic threshold ({traffic_threshold}) defined, but column '{stat_col_name}' not found for filtering.")

    if overlay_c_gdf.empty:
        logger.warning("Overlay C resulted in an empty GeoDataFrame after filtering.")

    save_vector_data(overlay_c_gdf, output_path, driver="GPKG")
    logger.info(f"Overlay C saved to: {output_path}")
    return output_path


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
    if "4326" not in str(cost_raster_path):
        cost_raster_path = cost_raster_path.with_name(cost_raster_path.stem + "_4326").with_suffix(".tif")
    logger.info(f"Using reprojected cost raster: {cost_raster_path}")

    overlay_c_gdf = load_vector_data(overlay_c_path)
    if overlay_c_gdf.empty:
        logger.warning("Overlay C input is empty. Skipping Overlay D.")
        # save_vector_data(overlay_c_gdf, output_path, driver="GPKG") # Save empty file
        return None # Return None to indicate no output

    # Sample cost raster values along the segment lines
    stat_col_name = "avg_cost"
    # Removed try...except block around sampling
    overlay_d_gdf = _sample_raster_along_lines(
        overlay_c_gdf, cost_raster_path, stat_col_name, stat="mean"
    )

    # --- Debug: Log avg_cost stats before filtering ---
    if stat_col_name in overlay_d_gdf.columns and not overlay_d_gdf[stat_col_name].isnull().all():
        logger.info(f"Debug: Stats for '{stat_col_name}' before filtering:")
        logger.info(f"  Min: {overlay_d_gdf[stat_col_name].min():.4f}")
        logger.info(f"  Max: {overlay_d_gdf[stat_col_name].max():.4f}")
        logger.info(f"  Mean: {overlay_d_gdf[stat_col_name].mean():.4f}")
        logger.info(f"  Median: {overlay_d_gdf[stat_col_name].median():.4f}")
        logger.info(f"  NaN count: {overlay_d_gdf[stat_col_name].isnull().sum()}")
    elif stat_col_name in overlay_d_gdf.columns:
         logger.warning(f"Debug: Column '{stat_col_name}' found but contains only NaN values.")
    else:
        logger.warning(f"Debug: Column '{stat_col_name}' not found before filtering.")
    # --- End Debug ---


    # Filter based on cost threshold from config (lower cost is better)
    cost_threshold = settings.processing.overlay_cost_threshold
    if cost_threshold is not None and stat_col_name in overlay_d_gdf.columns:
        initial_count_cost = len(overlay_d_gdf)
        # Keep segments where cost is BELOW threshold (implicitly drops NaN)
        # Note: NaN comparisons like NaN < threshold evaluate to False
        overlay_d_gdf = overlay_d_gdf[overlay_d_gdf[stat_col_name] < cost_threshold]
        filtered_count_cost = len(overlay_d_gdf)
        logger.info(
            f"Filtered Overlay D based on {stat_col_name} < {cost_threshold} (normalized). "
            f"Retained {filtered_count_cost}/{initial_count_cost} segments (NaN costs excluded)."
        )
    elif cost_threshold is not None:
         logger.warning(f"Cost threshold ({cost_threshold}) defined, but column '{stat_col_name}' not found for filtering.")

    # Note: Even without filtering, the avg_cost column is added for potential ranking later.

    if overlay_d_gdf.empty:
        logger.warning("Overlay D resulted in an empty GeoDataFrame after filtering.")

    save_vector_data(overlay_d_gdf, output_path, driver="GPKG")
    logger.info(f"Overlay D saved to: {output_path}")
    return output_path


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
    logger.info("--- Starting Task: Combine Features ---")

    # Extract necessary paths from feature_results, converting to Path objects
    def _to_path(p):
        return Path(p) if p else None

    prepared_segments_path = _to_path(feature_results.get("prepared_segments"))
    # Use the specific 'no lanes' output from roads task
    prepared_roads_no_lanes_path = _to_path(feature_results.get("road_vectors", {}).get("roads_simple_diff_lanes"))
    avg_speed_raster_path = _to_path(feature_results.get("speed_raster"))
    cost_raster_path = _to_path(feature_results.get("cost_raster"))

    # Check if the cost raster is in EPSG:4326
    if cost_raster_path and "4326" not in str(cost_raster_path):
        # Reproject the cost raster to EPSG:4326
        cost_raster_path = cost_raster_path.with_name(cost_raster_path.stem + "_4326").with_suffix(".tif")
        logger.info(f"Using reprojected cost raster: {cost_raster_path}")

    # Handle potential list of traffic rasters from build_features_task
    traffic_raster_input = feature_results.get("traffic_raster")
    traffic_raster_path = None
    if isinstance(traffic_raster_input, list):
        logger.info(f"Received a list for 'traffic_raster': {traffic_raster_input}. Selecting daytime raster.")
        # Select the daytime raster specifically, based on config filename
        daytime_raster_filename = settings.output_files.traffic_density_raster_daytime
        found_daytime = False
        for p in traffic_raster_input:
            path_obj = _to_path(p) # Convert potential string path to Path object
            if path_obj and path_obj.name == daytime_raster_filename:
                traffic_raster_path = path_obj
                logger.info(f"Using daytime traffic raster: {traffic_raster_path}")
                found_daytime = True
                break
        if not found_daytime:
            logger.error(f"Could not find daytime traffic raster ('{daytime_raster_filename}') in the provided list: {traffic_raster_input}. Overlay C might fail or skip traffic filter.")
            # traffic_raster_path remains None
    elif traffic_raster_input:
        # If it's not a list, assume it's a single path (string or Path)
        traffic_raster_path = _to_path(traffic_raster_input)
    else:
         logger.warning("No 'traffic_raster' key/value found in feature_results.")


    # Log extracted paths for verification
    logger.info("Inputs for Combine Features:")
    logger.info(f"  - Segments: {prepared_segments_path}")
    logger.info(f"  - Roads w/o Lanes: {prepared_roads_no_lanes_path}")
    logger.info(f"  - Avg Speed Raster: {avg_speed_raster_path}")
    logger.info(f"  - Traffic Raster: {traffic_raster_path}")
    logger.info(f"  - Cost Raster: {cost_raster_path}")


    # --- Execute Overlays Sequentially ---
    overlay_a_result_path = build_overlay_a(
        settings, wbt, prepared_segments_path, prepared_roads_no_lanes_path
    )

    overlay_b_result_path = None
    if overlay_a_result_path:
        overlay_b_result_path = build_overlay_b(
            settings, wbt, overlay_a_result_path, avg_speed_raster_path
        )

    overlay_c_result_path = None
    if overlay_b_result_path:
        overlay_c_result_path = build_overlay_c(
            settings, wbt, overlay_b_result_path, traffic_raster_path
        )

    overlay_d_result_path = None
    if overlay_c_result_path:
        overlay_d_result_path = build_overlay_d(
            settings, wbt, overlay_c_result_path, cost_raster_path
        )

    combined_results = {
        "overlay_a": overlay_a_result_path,
        "overlay_b": overlay_b_result_path,
        "overlay_c": overlay_c_result_path,
        "overlay_d": overlay_d_result_path,  # Final layer before evaluation ranking
    }

    logger.info("--- Task: Combine Features Completed ---")
    logger.info("Generated overlay outputs:")
    for key, value in combined_results.items():
        if value and value.exists():
            try:
                gdf_check = load_vector_data(value)
                logger.info(f"  - {key}: {value} ({len(gdf_check)} features)")
            except Exception:
                 logger.info(f"  - {key}: {value} (exists, but failed quick load check)")
        elif value:
             logger.warning(f"  - {key}: Path generated ({value}), but file not found.")
        else:
            logger.warning(f"  - {key}: Not generated or failed.")

    return combined_results


# --- Helper for Raster Sampling ---

def _sample_raster_along_lines(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    stat_col_name: str,
    stat: str = "mean",
    sampling_distance: float = 10.0, # Sample every 10 meters along the line
) -> gpd.GeoDataFrame:
    """
    Samples a raster along LineString geometries in a GeoDataFrame and adds a statistic
    (mean, median, max, min) of the sampled values as a new column.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with LineString geometries.
        raster_path (Path): Path to the raster file to sample.
        stat_col_name (str): Name of the new column to add with the calculated statistic.
        stat (str): Statistic to calculate ('mean', 'median', 'max', 'min'). Defaults to 'mean'.
        sampling_distance (float): Distance between sampling points along the line (in gdf's CRS units).

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with the new statistic column added.

    Raises:
        FileNotFoundError: If the raster file does not exist.
        ValueError: If input GDF CRS or raster CRS is missing, or if stat is invalid.
    """
    if not raster_path.is_file():
        raise FileNotFoundError(f"Raster file not found for sampling: {raster_path}")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame for sampling must have a CRS defined.")

    logger.info(f"Sampling raster '{raster_path.name}' along {len(gdf)} lines for '{stat_col_name}' (stat: {stat}).")
    logger.info(f"gdf crs: {gdf.crs}")

    # results = []
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if raster_crs is None:
             raise ValueError(f"Raster file {raster_path} is missing CRS information.")
        logger.info(f"Raster crs: {raster_crs}")

        if gdf.crs != raster_crs:
            logger.warning(f"CRS mismatch: gdf ({gdf.crs}) vs raster ({raster_crs}). Skipping.")
            return gdf # Return original GDF if CRS mismatch
        #     logger.warning(f"Reprojecting vector data from {gdf.crs} to {raster_crs} for sampling.")
        #     try:
        #         # Use utils function for robust reprojection
        #         gdf_proj = reproject_gdf(gdf, raster_crs)
        #     except Exception as reproj_e:
        #         logger.error(f"Reprojection failed during sampling: {reproj_e}", exc_info=True)
        #         # Add NaN column and return original GDF structure but with original CRS
        #         gdf_out = gdf.copy()
        #         gdf_out[stat_col_name] = np.nan
        #         return gdf_out

        # Prepare list to store coordinates for sampling
        all_coords = []
        line_indices = [] # Keep track of which line each set of points belongs to

        for index, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty or geom.geom_type != 'LineString':
                # Handle non-LineString or empty geometries - they get NaN later
                continue

            # Generate points along the line
            num_points = max(2, int(geom.length / sampling_distance)) # Ensure at least start/end points
            sampled_points_coords = [
                geom.interpolate(i / (num_points - 1), normalized=True).coords[0]
                for i in range(num_points)
            ]

            if sampled_points_coords:
                all_coords.extend(sampled_points_coords)
                line_indices.extend([index] * len(sampled_points_coords)) # Mark points with original line index

        # Sample the raster at all collected coordinates
        if not all_coords:
             logger.warning("No valid coordinates generated for sampling.")
             sampled_values_flat = []
        else:
            try:
                # rasterio.sample.sample_gen returns an iterator of arrays (one per band)
                # We assume single band, so take the first element.
                sampled_values_gen = rasterio.sample.sample_gen(src, all_coords, indexes=[1])
                # Extract the values, handling potential errors during generation
                sampled_values_flat = [item[0] for item in sampled_values_gen]
            except Exception as rio_sample_e:
                 logger.error(f"Rasterio sampling failed: {rio_sample_e}", exc_info=True)
                 sampled_values_flat = [np.nan] * len(all_coords) # Assign NaN if sampling fails


        # Aggregate results back to the original lines
        sampled_df = pd.DataFrame({
            'line_index': line_indices,
            'value': sampled_values_flat
        })

        # Group by original line index and calculate the statistic
        # Handle potential NaN values during aggregation
        if stat == "mean":
            line_stats = sampled_df.groupby('line_index')['value'].mean()
        elif stat == "median":
            line_stats = sampled_df.groupby('line_index')['value'].median()
        elif stat == "max":
            line_stats = sampled_df.groupby('line_index')['value'].max()
        elif stat == "min":
            line_stats = sampled_df.groupby('line_index')['value'].min()
        else:
            raise ValueError(f"Invalid statistic '{stat}'. Choose from 'mean', 'median', 'max', 'min'.")

        # Map results back to the original GeoDataFrame (using original index)
        # Ensure the output GDF has the same index and CRS as the input GDF
        gdf_out = gdf.copy()
        # Use .map() on the original index, aligning with the calculated stats
        # The line_stats Series index corresponds to the enumerated index used earlier
        # We need to map this back to the original gdf index if they differ (e.g., if gdf_proj was filtered)
        # A safer way is to create a mapping from the enumerated index back to the original gdf index
        index_map = {i: idx for i, idx in enumerate(gdf.index)}
        mapped_stats = pd.Series(line_stats.index).map(index_map).map(line_stats)
        mapped_stats.index = line_stats.index.map(index_map) # Align index for assignment

        # Assign the calculated stats, filling missing lines (e.g., non-LineString, empty) with NaN
        gdf_out[stat_col_name] = mapped_stats.reindex(gdf_out.index)


        num_nan = gdf_out[stat_col_name].isna().sum()
        if num_nan > 0:
            logger.warning(f"{num_nan}/{len(gdf_out)} lines have NaN for '{stat_col_name}' (due to geometry issues, sampling errors, or nodata).")

        logger.info(f"Raster sampling complete. Added column '{stat_col_name}'.")
        return gdf_out

    # Removed outer try...except block for sampling helper


# --- Main Execution Block ---
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("--- Running Combine Features Task Standalone ---")

    # --- Configuration ---
    # Use settings loaded from config.py
    settings = app_settings
    # Define the specific output directory from the previous run to use as input
    previous_run_name = "mca_20250426_2122_workflow"
    previous_run_output_dir = settings.paths.base_dir / "output" / previous_run_name
    logger.info(f"Using input features from: {previous_run_output_dir}")

    # Define the output directory for this test run's overlays and maps
    test_output_dir = settings.paths.base_dir / "output" / "mca_combine_features_test_2"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving test outputs (overlays, maps) to: {test_output_dir}")
    # Override the default output dir in settings for this run
    settings.paths.output_dir = test_output_dir

    # Check if previous run directory exists
    if not previous_run_output_dir.is_dir():
        logger.error(f"Input directory from previous run not found: {previous_run_output_dir}")
        logger.error(f"Please ensure the '{previous_run_name}' output exists or modify the path.")
        sys.exit(1)

    # --- Mock feature_results dictionary ---
    # Construct paths based on the previous run directory and expected filenames from config
    feature_results_mock = {
        "prepared_segments": previous_run_output_dir / settings.output_files.prepared_segments_gpkg,
        "road_vectors": {
            "roads_simple_diff_lanes": previous_run_output_dir / settings.output_files.prepared_roads_simple_diff_lanes_gpkg,
        },
        "speed_raster": previous_run_output_dir / settings.output_files.average_speed_raster,
        "traffic_raster": previous_run_output_dir / settings.output_files.traffic_density_raster_daytime,
        "cost_raster": previous_run_output_dir / settings.output_files.normalized_cost_layer,
    }

    # Verify that input files exist
    missing_inputs = []
    for key, path in feature_results_mock.items():
        if type(path) is dict: # needed for road_vectors
            # Check nested dictionary for road vectors
            for sub_key, sub_path in path.items():
                if not sub_path or not Path(sub_path).exists():
                    missing_inputs.append(f"{key} ({sub_key}) ({sub_path})")
        elif not path or not Path(path).exists():
            missing_inputs.append(f"{key} ({path})")

    if missing_inputs:
        logger.error("Missing input files required for the test:")
        for missing in missing_inputs:
            logger.error(f"  - {missing}")
        logger.error(f"Please check the contents of: {previous_run_output_dir}")
        sys.exit(1)

    # --- Initialize WhiteboxTools ---
    wbt = None
    try:
        logger.info("Initializing WhiteboxTools...")
        wbt = WhiteboxTools()
        wbt.set_verbose_mode(settings.processing.wbt_verbose)
        logger.info("WhiteboxTools initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize WhiteboxTools: {e}", exc_info=True)
        sys.exit(1)

    # --- Run the Combine Features Task ---
    try:
        combined_results = combine_features_task(feature_results_mock, settings, wbt)
    except Exception as e:
        logger.error(f"Error running combine_features_task: {e}", exc_info=True)
        sys.exit(1)

    # --- Display Results on Combined Folium Map ---
    logger.info("--- Generating Combined Folium Map for Overlays and Inputs ---")

    # Load overlay GDFs
    overlay_gdfs_loaded = {}
    for key, path in combined_results.items():
        if path and path.exists():
            try:
                overlay_gdfs_loaded[key] = load_vector_data(path)
                logger.info(f"Loaded {key} GDF ({len(overlay_gdfs_loaded[key])} features).")
            except Exception as load_e:
                logger.error(f"Failed to load GDF for {key} from {path}: {load_e}")
        elif path:
             logger.warning(f"Overlay file not found for {key}: {path}")
        else:
             logger.warning(f"Overlay {key} was not generated.")

    # Define input rasters to display (adjust keys as needed)
    # Assuming 'popularity_raster' is generated by build_features task
    # Using a default popularity metric filename based on config
    pop_metric = settings.processing.segment_popularity_metrics[0] # e.g., 'efforts_per_age'
    pop_raster_filename = f"{settings.output_files.segment_popularity_raster_prefix}_{pop_metric}.tif"
    popularity_raster_path = previous_run_output_dir / pop_raster_filename

    input_rasters_to_display = {
        "popularity": popularity_raster_path if popularity_raster_path.exists() else None,
        "speed": feature_results_mock.get("speed_raster"),
        "traffic": feature_results_mock.get("traffic_raster"), # This might be daytime, adjust if needed
        "cost": feature_results_mock.get("cost_raster"),
    }

    # check cost raster has been reprojected
    if input_rasters_to_display['cost'] and "4326" not in str(input_rasters_to_display['cost']):
        input_rasters_to_display['cost'] = input_rasters_to_display['cost'].with_name(input_rasters_to_display['cost'].stem + "_4326").with_suffix(".tif")
        logger.info(f"Using reprojected cost raster: {input_rasters_to_display['cost']}")


    # Log which input rasters are being included
    logger.info("Input rasters for combined map:")
    for name, path in input_rasters_to_display.items():
        if path and path.exists():
            logger.info(f"  - {name}: {path}")
        elif path:
            logger.warning(f"  - {name}: Path specified ({path}), but file not found.")
        else:
             logger.warning(f"  - {name}: Path not available in feature_results_mock.")


    # Define columns for tooltips
    tooltip_cols_combined = [
        settings.input_data.segment_id_field,
        'popularity_norm', # Added in Overlay A filter logic (verify name)
        'avg_speed',
        'avg_traffic',
        'avg_cost'
    ]
    # Add other relevant segment attributes if available
    # e.g., tooltip_cols_combined.extend(['distance', 'athlete_count'])

    # Define output path for the combined map
    combined_map_output_path = settings.paths.output_dir / "combined_overlays_map.html"

    # Call the new display function
    try:
        display_overlay_folium_map(
            overlay_gdfs=overlay_gdfs_loaded,
            input_rasters=input_rasters_to_display,
            output_html_path_str=str(combined_map_output_path),
            target_crs_epsg=settings.processing.output_crs_epsg,
            tooltip_columns=tooltip_cols_combined,
            zoom_start=11 # Adjust zoom if needed
        )
        logger.info(f"Combined map saved to: {combined_map_output_path}")
    except Exception as combined_map_e:
        logger.error(f"Failed to generate combined overlay map: {combined_map_e}", exc_info=True)

    logger.info("--- Standalone Combine Features Test Finished ---")
