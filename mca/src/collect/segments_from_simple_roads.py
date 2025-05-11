import logging
import time
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString

from src.config import settings
from src.utils import load_vector_data, save_vector_data, reproject_gdf
from src.collect.segments import (
    explore_segments_with_multiple_bounds,
    segments_to_gdf,
)

# Setup logging
logger = logging.getLogger(__name__)

# --- Helper Functions (Adapted from collect_segments_from_diff.py and previous version) ---

def load_geopackage(path: Path) -> gpd.GeoDataFrame | None:
    """Safely loads a GeoPackage file."""
    if path.exists() and path.stat().st_size > 0:
        try:
            gdf = gpd.read_file(path)
            logger.info(f"Successfully loaded GeoPackage: {path}")
            return gdf
        except Exception as e:
            logger.error(f"Error loading GeoPackage {path}: {e}", exc_info=True)
            return None
    else:
        logger.warning(f"GeoPackage not found or empty: {path}")
        return None

def load_n50_simple_roads(target_crs: str) -> gpd.GeoDataFrame | None:
    """Loads and filters N50 Samferdsel layer for simple car roads."""
    logger.info("Loading N50 Samferdsel layer...")
    gdb_path = settings.paths.n50_gdb_path
    layer_name = settings.input_data.n50_samferdsel_layer
    typeveg_field = settings.input_data.n50_samferdsel_typeveg_field
    simple_road_value = settings.input_data.n50_typeveg_road_simple

    try:
        roads_gdf = gpd.read_file(gdb_path, layer=layer_name)
        logger.info(f"Loaded {len(roads_gdf)} features from {layer_name}.")
        filtered_roads = roads_gdf[roads_gdf[typeveg_field] == simple_road_value].copy()
        logger.info(f"Filtered down to {len(filtered_roads)} simple car roads ('{simple_road_value}').")
        if filtered_roads.empty:
            logger.warning("No simple car roads found in the N50 layer.")
            return None
        filtered_roads = reproject_gdf(filtered_roads, target_crs)
        logger.info(f"Ensured CRS {target_crs} for simple roads.")
        return filtered_roads
    except Exception as e:
        logger.error(f"Error loading/filtering N50 roads: {e}", exc_info=True)
        return None

def load_and_combine_segments(target_crs: str) -> gpd.GeoDataFrame | None:
    """Loads and combines segments from GeoJSON and collected GeoPackage."""
    logger.info("Loading and combining all known Strava segments...")
    segment_id_field = settings.input_data.segment_id_field
    all_segments = []

    # Load Original GeoJSON (if exists)
    geojson_path = settings.paths.strava_segments_geojson
    if geojson_path.exists():
        logger.info(f"Loading original segments from: {geojson_path}")
        geojson_gdf = load_vector_data(geojson_path)
        if geojson_gdf is not None:
            if segment_id_field in geojson_gdf.columns:
                all_segments.append(reproject_gdf(geojson_gdf, target_crs))
            else:
                logger.warning(f"Segment ID field '{segment_id_field}' not found in {geojson_path}. Skipping.")
    else:
        logger.warning(f"Original segments file not found: {geojson_path}")

    # Load Segments from Diff Collection (if exists)
    diff_gpkg_path = settings.paths.collected_segments_gpkg
    if diff_gpkg_path.exists():
        logger.info(f"Loading segments collected from diff layer: {diff_gpkg_path}")
        diff_gpkg_gdf = load_vector_data(diff_gpkg_path)
        if diff_gpkg_gdf is not None:
            if segment_id_field in diff_gpkg_gdf.columns:
                 all_segments.append(reproject_gdf(diff_gpkg_gdf, target_crs))
            else:
                 logger.warning(f"Segment ID field '{segment_id_field}' not found in {diff_gpkg_path}. Skipping.")
    else:
        logger.warning(f"Collected segments (diff) file not found: {diff_gpkg_path}")

    # Load Segments from *this* script's previous runs (if exists)
    simple_roads_gpkg_path = settings.paths.collected_segments_from_simple_roads_gpkg
    if simple_roads_gpkg_path.exists():
        logger.info(f"Loading segments collected from simple roads (previous runs): {simple_roads_gpkg_path}")
        simple_roads_gpkg_gdf = load_vector_data(simple_roads_gpkg_path)
        if simple_roads_gpkg_gdf is not None:
            if segment_id_field in simple_roads_gpkg_gdf.columns:
                 all_segments.append(reproject_gdf(simple_roads_gpkg_gdf, target_crs))
            else:
                 logger.warning(f"Segment ID field '{segment_id_field}' not found in {simple_roads_gpkg_path}. Skipping.")
    else:
        logger.info(f"No previously collected segments from simple roads found at: {simple_roads_gpkg_path}")


    if not all_segments:
        logger.error("No segment data could be loaded from any source.")
        return None

    # Combine all loaded segments
    logger.info(f"Combining segments from {len(all_segments)} sources...")
    combined_gdf = pd.concat(all_segments, ignore_index=True, sort=False)
    # Ensure consistent CRS after concat (should be target_crs)
    combined_gdf = reproject_gdf(combined_gdf, target_crs)
    # Drop duplicates based on the actual segment ID
    combined_gdf = combined_gdf.drop_duplicates(subset=[segment_id_field], keep='last')
    logger.info(f"Combined data has {len(combined_gdf)} unique segments (CRS: {combined_gdf.crs}).")

    # Final checks
    if 'geometry' not in combined_gdf.columns or combined_gdf.geometry.isnull().all():
        logger.error("Combined segments lack valid geometry column.")
        return None
    if not combined_gdf.crs or not combined_gdf.crs.equals(target_crs):
        logger.error(f"Combined segments CRS is not set or does not match target {target_crs}.")
        return None

    return combined_gdf[['geometry']] # Only need geometry for buffering


def convert_lines_to_points(gdf: gpd.GeoDataFrame, segment_length: float = 50.0) -> gpd.GeoDataFrame:
    """Converts LineString/MultiLineString geometries to points at regular intervals."""
    points_list = []
    original_crs = gdf.crs
    if not original_crs:
        logger.warning("Input GDF for line-to-point conversion has no CRS. Assuming project default.")
        original_crs = f"EPSG:{settings.processing.output_crs_epsg}"

    logger.info(f"Converting {len(gdf)} line features to points with interval {segment_length}m...")
    for index, row in gdf.iterrows():
        geometry = row.geometry
        lines_to_process = []
        if isinstance(geometry, LineString): 
            lines_to_process.append(geometry)
        elif isinstance(geometry, MultiLineString): 
            lines_to_process.extend(list(geometry.geoms))
        else: 
            logger.warning(f"Skipping non-line geometry at index {index} (Type: {type(geometry)})."); continue

        for i, line in enumerate(lines_to_process):
             if not isinstance(line, LineString) or line.is_empty or line.length == 0: 
                 continue
             distances = np.arange(0, line.length, segment_length)
             if distances.size == 0 or distances[-1] < line.length: 
                 distances = np.append(distances, line.length)
             for dist in distances:
                 point = line.interpolate(dist)
                 sub_line_id = f"_part_{i}" if isinstance(geometry, MultiLineString) else ""
                 # Create a unique ID for the point itself
                 point_id = f"simple_road_{index}{sub_line_id}_dist_{dist:.0f}"
                 points_list.append({'point_id': point_id, 'original_road_index': index, 'geometry': point})

    if not points_list:
        logger.warning("No points generated from lines.")
        return gpd.GeoDataFrame([], columns=['point_id', 'original_road_index', 'geometry'], crs=original_crs)

    points_gdf = gpd.GeoDataFrame(points_list, crs=original_crs)
    logger.info(f"Converted {len(gdf)} line features to {len(points_gdf)} points.")
    return points_gdf

def generate_initial_search_points(target_crs: str) -> gpd.GeoDataFrame | None:
    """Generates the initial set of points on simple roads that lack nearby segments."""
    logger.info("--- Generating Initial Search Points for Simple Roads ---")
    segment_buffer_distance = 20.0 # Meters
    point_interval = 50.0 # Meters

    # 1. Load Simple Roads
    simple_roads_gdf = load_n50_simple_roads(target_crs)
    if simple_roads_gdf is None or simple_roads_gdf.empty: return None

    # 2. Load All Existing Segments (only geometry needed)
    all_segments_geom_gdf = load_and_combine_segments(target_crs)
    if all_segments_geom_gdf is None or all_segments_geom_gdf.empty:
        logger.warning("No existing segments found. Proceeding to generate points from all simple roads.")
        # If no segments exist, all road points are candidates
        road_points_gdf = convert_lines_to_points(simple_roads_gdf, segment_length=point_interval)
        if road_points_gdf.empty: 
            logger.error("Failed to generate points from roads.")
            return None
        logger.info(f"Generated {len(road_points_gdf)} initial search points (no existing segments to filter by).")
        return road_points_gdf[['point_id', 'geometry']] # Keep only essential columns

    # 3. Convert Roads to Points
    road_points_gdf = convert_lines_to_points(simple_roads_gdf, segment_length=point_interval)
    if road_points_gdf.empty: 
        logger.error("Failed to generate points from roads."); return None

    # 4. Buffer Existing Segments
    logger.info(f"Buffering {len(all_segments_geom_gdf)} existing segments by {segment_buffer_distance}m...")
    try:
        # Ensure segments geometry is valid before buffering
        invalid_segments = ~all_segments_geom_gdf.geometry.is_valid
        if invalid_segments.any():
            logger.warning(f"Found {invalid_segments.sum()} invalid existing segment geometries. Attempting buffer(0) fix...")
            all_segments_geom_gdf.loc[invalid_segments, 'geometry'] = all_segments_geom_gdf.loc[invalid_segments].geometry.buffer(0)
            all_segments_geom_gdf = all_segments_geom_gdf[~all_segments_geom_gdf.geometry.is_empty]
            logger.info(f"{len(all_segments_geom_gdf)} existing segments remaining after validity fix.")

        segment_buffers = all_segments_geom_gdf.geometry.buffer(segment_buffer_distance)
        # Dissolve buffers for efficiency
        dissolved_buffers = gpd.GeoDataFrame(geometry=segment_buffers, crs=target_crs).unary_union
        dissolved_gdf = gpd.GeoDataFrame([1], geometry=[dissolved_buffers], crs=target_crs)
        logger.info("Existing segment buffering and dissolving complete.")
    except Exception as e:
        logger.error(f"Error buffering existing segments: {e}", exc_info=True); return None

    # 5. Identify Points Outside Segment Buffers
    logger.info(f"Identifying which of the {len(road_points_gdf)} road points fall outside existing segment buffers...")
    try:
        points_within = gpd.sjoin(road_points_gdf, dissolved_gdf, how='inner', predicate='within')
        points_within_ids = set(points_within['point_id'])
        points_outside_gdf = road_points_gdf[~road_points_gdf['point_id'].isin(points_within_ids)].copy()
        logger.info(f"Found {len(points_outside_gdf)} points on simple roads that are not within {segment_buffer_distance}m of any known segment.")
    except Exception as e:
        logger.error(f"Error during spatial join: {e}", exc_info=True); return None

    if points_outside_gdf.empty:
        logger.info("No points found outside existing segment buffers.")
        return None

    return points_outside_gdf[['point_id', 'geometry']] # Keep only essential columns

# --- Main Script Logic ---

def main():
    """Main script logic for collecting segments near simple roads."""
    logger.info("--- Starting Segment Collection from Simple Roads ---")

    # --- Configuration ---
    target_crs = f"EPSG:{settings.processing.output_crs_epsg}"
    remaining_points_path = settings.paths.remaining_simple_road_points_gpkg
    collected_segments_path = settings.paths.collected_segments_from_simple_roads_gpkg
    sample_size = settings.processing.segment_collection_sample_size
    search_area_km2 = [0.5, 1, 2] # Example search areas

    # --- Load or Generate Search Points ---
    remaining_points_gdf = load_geopackage(remaining_points_path)

    if remaining_points_gdf is None or remaining_points_gdf.empty:
        logger.info(f"No remaining points found at {remaining_points_path}. Generating initial points...")
        initial_points_gdf = generate_initial_search_points(target_crs)

        if initial_points_gdf is None or initial_points_gdf.empty:
            logger.error("Failed to generate initial search points. Exiting.")
            return

        # Ensure point_id column exists (should be guaranteed by generate_initial_search_points)
        if 'point_id' not in initial_points_gdf.columns:
             logger.error("Generated initial points lack 'point_id'. Exiting.")
             return

        save_vector_data(initial_points_gdf, remaining_points_path)
        logger.info(f"Initial search points saved to {remaining_points_path}")
        remaining_points_gdf = initial_points_gdf # Use the newly generated points
    else:
        logger.info(f"Loaded {len(remaining_points_gdf)} remaining points from {remaining_points_path}")
        # Ensure point_id column exists and is suitable
        if 'point_id' not in remaining_points_gdf.columns:
             logger.error("Existing remaining points file lacks 'point_id'. Cannot proceed.")
             # Attempt fallback? For now, exit.
             # remaining_points_gdf['point_id'] = [f"fallback_{i}" for i in range(len(remaining_points_gdf))]
             return

    if remaining_points_gdf.empty:
        logger.info("No points remaining to search. Exiting.")
        return

    # --- Sample Points for Current Run ---
    actual_sample_size = min(sample_size, len(remaining_points_gdf))
    if actual_sample_size <= 0:
        logger.info("Sample size is 0 or no points left. Exiting.")
        return

    logger.info(f"Sampling {actual_sample_size} points for this run...")
    sampled_points_gdf = remaining_points_gdf.sample(n=actual_sample_size, random_state=settings.processing.seed)

    # --- Explore Segments ---
    logger.info("Starting segment exploration around sampled points...")
    # Ensure points are in WGS84 (Lat/Lon) for the Strava API
    sampled_points_wgs84 = reproject_gdf(sampled_points_gdf, "EPSG:4326")

    start_time = time.time()
    collected_segments_list, processed_indices = explore_segments_with_multiple_bounds(
        sampled_points_wgs84,
        area_sizes=search_area_km2,
        point_col='geometry'
    )
    end_time = time.time()
    logger.info(f"Segment exploration completed in {end_time - start_time:.2f} seconds.")
    logger.info(f"Found {len(collected_segments_list)} segments in this run.")
    logger.info(f"Successfully processed {len(processed_indices)} points out of {actual_sample_size} sampled.")

    # --- Process and Save Results ---
    if collected_segments_list:
        new_segments_gdf = segments_to_gdf(collected_segments_list)
        logger.info(f"Converted {len(new_segments_gdf)} new segments to GeoDataFrame.")

        # Load existing segments and append new ones
        existing_segments_gdf = load_geopackage(collected_segments_path)
        if existing_segments_gdf is None:
            combined_segments_gdf = new_segments_gdf
        else:
            # Ensure columns match before concatenating
            common_columns = list(set(existing_segments_gdf.columns) & set(new_segments_gdf.columns))
            id_col = settings.input_data.segment_id_field
            if 'geometry' not in common_columns: common_columns.append('geometry')
            if id_col not in common_columns: common_columns.append(id_col)

            # Ensure CRS match before concat
            existing_segments_gdf = reproject_gdf(existing_segments_gdf, new_segments_gdf.crs)

            combined_segments_gdf = pd.concat(
                [existing_segments_gdf[common_columns], new_segments_gdf[common_columns]],
                ignore_index=True
            )
            # Drop duplicates based on segment ID
            if id_col in combined_segments_gdf.columns:
                combined_segments_gdf = combined_segments_gdf.drop_duplicates(subset=[id_col], keep='last')
            else:
                 logger.warning(f"Segment ID column '{id_col}' not found for deduplication.")

        save_vector_data(combined_segments_gdf, collected_segments_path)
    else:
        logger.info("No new segments found in this run.")

    # --- Update Remaining Points ---
    if processed_indices:
        # Get the unique IDs of the points that were successfully processed
        processed_point_ids = sampled_points_gdf.iloc[list(processed_indices)]['point_id'].tolist()

        logging.info(f"Removing {len(processed_point_ids)} processed points from the remaining list...")
        remaining_points_updated_gdf = remaining_points_gdf[
            ~remaining_points_gdf['point_id'].isin(processed_point_ids)
        ]
        save_vector_data(remaining_points_updated_gdf, remaining_points_path)
        logging.info(f"{len(remaining_points_updated_gdf)} points remaining for future runs.")
    else:
        logging.warning("No points were successfully processed in this run. Remaining points file not updated.")

    logger.info("--- Segment Collection from Simple Roads Run Finished ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
