import logging
import time
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString # Import MultiLineString
import numpy as np
import sys

from src.config import settings
from src.collect.segments import (
    explore_segments_with_multiple_bounds,
    segments_to_gdf,
    # create_bounds_around_point, # May not be needed directly, but good to have context
)
# from src.utils import setup_logging # Assuming a utility for logging setup exists

# Setup logging
# setup_logging() # Use your project's logging setup if available
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_geopackage(path: Path) -> gpd.GeoDataFrame | None:
    """Safely loads a GeoPackage file."""
    if path.exists() and path.stat().st_size > 0:
        try:
            gdf = gpd.read_file(path)
            logging.info(f"Successfully loaded GeoPackage: {path}")
            return gdf
        except Exception as e:
            logging.error(f"Error loading GeoPackage {path}: {e}", exc_info=True)
            return None
    else:
        logging.warning(f"GeoPackage not found or empty: {path}")
        return None

def save_geopackage(gdf: gpd.GeoDataFrame, path: Path):
    """Saves a GeoDataFrame to GeoPackage, creating parent directories."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(path, driver="GPKG")
        logging.info(f"Successfully saved GeoDataFrame to: {path} ({len(gdf)} features)")
    except Exception as e:
        logging.error(f"Error saving GeoDataFrame to {path}: {e}", exc_info=True)

def convert_lines_to_points(gdf: gpd.GeoDataFrame, segment_length: float = 50.0) -> gpd.GeoDataFrame:
    """
    Converts LineString geometries in a GeoDataFrame to points at regular intervals.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with LineString geometries.
        segment_length (float): Distance between points along the line (in the GDF's CRS units).

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Point geometries.
    """
    points_list = []
    original_crs = gdf.crs

    if not original_crs:
        logging.warning("Input GeoDataFrame for line-to-point conversion has no CRS set. Assuming project default.")
        original_crs = f"EPSG:{settings.processing.output_crs_epsg}" # Fallback

    for index, row in gdf.iterrows():
        geometry = row.geometry
        lines_to_process = []

        if isinstance(geometry, LineString):
            lines_to_process.append(geometry)
        elif isinstance(geometry, MultiLineString):
            # Add all individual LineStrings from the MultiLineString
            lines_to_process.extend(list(geometry.geoms))
        else:
            logging.warning(f"Skipping geometry at index {index} - not LineString or MultiLineString (Type: {type(geometry)}).")
            continue # Skip to the next row

        for i, line in enumerate(lines_to_process):
             if not isinstance(line, LineString) or line.is_empty or line.length == 0:
                 logging.debug(f"Skipping empty or invalid sub-line {i} in feature {index}")
                 continue

             distances = np.arange(0, line.length, segment_length)
             # Include the endpoint if it's not already covered
             if distances.size == 0 or distances[-1] < line.length:
                 distances = np.append(distances, line.length)

             for dist in distances:
                 point = line.interpolate(dist)
                 # Create a unique ID for each point, including multi-line index if applicable
                 sub_line_id = f"_part_{i}" if isinstance(geometry, MultiLineString) else ""
                 point_id = f"line_{index}{sub_line_id}_dist_{dist:.0f}"
                 points_list.append({'point_id': point_id, 'original_feature_index': index, 'geometry': point})


    if not points_list:
        logging.warning("No points generated from LineString or MultiLineString geometries.")
        return gpd.GeoDataFrame([], columns=['point_id', 'original_feature_index', 'geometry'], crs=original_crs)

    points_gdf = gpd.GeoDataFrame(points_list, crs=original_crs)
    logging.info(f"Converted {len(gdf)} features (LineString/MultiLineString) to {len(points_gdf)} points.")
    return points_gdf

def main():
    """Main script logic."""
    logging.info("--- Starting Segment Collection from Difference Layer ---")

    # --- Configuration ---
    diff_layer_path = settings.paths.diff_layer_gpkg
    remaining_points_path = settings.paths.remaining_search_points_gpkg
    collected_segments_path = settings.paths.collected_segments_gpkg
    sample_size = settings.processing.segment_collection_sample_size
    search_area_km2 = [0.5, 1, 2] # Example search areas, adjust as needed

    # --- Load or Generate Search Points ---
    remaining_points_gdf = load_geopackage(remaining_points_path)

    if remaining_points_gdf is None or remaining_points_gdf.empty:
        logging.info(f"No remaining points found at {remaining_points_path}. Loading diff layer.")
        diff_gdf = load_geopackage(diff_layer_path)
        if diff_gdf is None or diff_gdf.empty:
            logging.error(f"Difference layer not found or empty at {diff_layer_path}. Exiting.")
            return

        logging.info("Converting difference layer lines to points...")
        # Ensure diff_gdf has the correct CRS before conversion
        if diff_gdf.crs is None:
             logging.warning(f"Diff layer {diff_layer_path} has no CRS. Assuming EPSG:{settings.processing.output_crs_epsg}")
             diff_gdf.crs = f"EPSG:{settings.processing.output_crs_epsg}"
        elif diff_gdf.crs.to_epsg() != settings.processing.output_crs_epsg:
             logging.info(f"Reprojecting diff layer from {diff_gdf.crs} to EPSG:{settings.processing.output_crs_epsg}")
             diff_gdf = diff_gdf.to_crs(epsg=settings.processing.output_crs_epsg)

        remaining_points_gdf = convert_lines_to_points(diff_gdf, segment_length=50) # Use 50m spacing

        if remaining_points_gdf.empty:
            logging.error("Failed to generate any points from the difference layer. Exiting.")
            return

        # Add a unique index if 'point_id' wasn't generated reliably
        if 'point_id' not in remaining_points_gdf.columns:
             remaining_points_gdf['point_id'] = remaining_points_gdf.index.astype(str)

        save_geopackage(remaining_points_gdf, remaining_points_path)
        logging.info(f"Initial search points saved to {remaining_points_path}")
    else:
        logging.info(f"Loaded {len(remaining_points_gdf)} remaining points from {remaining_points_path}")
        # Ensure point_id column exists and is suitable as unique identifier
        if 'point_id' not in remaining_points_gdf.columns:
             logging.warning("Existing remaining points file lacks 'point_id'. Using index as fallback.")
             remaining_points_gdf['point_id'] = remaining_points_gdf.index.astype(str)


    if remaining_points_gdf.empty:
        logging.info("No points remaining to search. Exiting.")
        return

    # --- Sample Points for Current Run ---
    actual_sample_size = min(sample_size, len(remaining_points_gdf))
    if actual_sample_size == 0:
        logging.info("Sample size is 0 or no points left. Exiting.")
        return

    logging.info(f"Sampling {actual_sample_size} points for this run...")
    sampled_points_gdf = remaining_points_gdf.sample(n=actual_sample_size, random_state=settings.processing.seed)

    # --- Explore Segments ---
    logging.info("Starting segment exploration around sampled points...")
    # Ensure points are in WGS84 (Lat/Lon) for the Strava API bounds calculation
    sampled_points_wgs84 = sampled_points_gdf.to_crs("EPSG:4326")

    start_time = time.time()
    collected_segments_list, processed_indices = explore_segments_with_multiple_bounds(
        sampled_points_wgs84, # Pass WGS84 version
        area_sizes=search_area_km2,
        point_col='geometry' # Explicitly state geometry column
    )
    end_time = time.time()
    logging.info(f"Segment exploration completed in {end_time - start_time:.2f} seconds.")
    logging.info(f"Found {len(collected_segments_list)} segments in this run.")
    logging.info(f"Successfully processed {len(processed_indices)} points out of {actual_sample_size} sampled.")

    # --- Process and Save Results ---
    if collected_segments_list:
        new_segments_gdf = segments_to_gdf(collected_segments_list)
        logging.info(f"Converted {len(new_segments_gdf)} new segments to GeoDataFrame.")

        # Load existing segments and append new ones
        existing_segments_gdf = load_geopackage(collected_segments_path)
        if existing_segments_gdf is None:
            # Use specific columns if creating anew, matching segments_to_gdf output
            combined_segments_gdf = new_segments_gdf
        else:
            # Ensure columns match before concatenating, handle potential schema differences
            # A simple approach: only keep columns present in both
            common_columns = list(set(existing_segments_gdf.columns) & set(new_segments_gdf.columns))
            # Make sure 'geometry' and 'id' (or your segment identifier) are included
            if 'geometry' not in common_columns: common_columns.append('geometry')
            if settings.input_data.segment_id_field not in common_columns:
                 common_columns.append(settings.input_data.segment_id_field)

            combined_segments_gdf = pd.concat(
                [existing_segments_gdf[common_columns], new_segments_gdf[common_columns]],
                ignore_index=True
            )
            # Drop duplicates based on segment ID
            id_col = settings.input_data.segment_id_field
            if id_col in combined_segments_gdf.columns:
                combined_segments_gdf = combined_segments_gdf.drop_duplicates(subset=[id_col], keep='last')
            else:
                 logging.warning(f"Segment ID column '{id_col}' not found for deduplication.")


        save_geopackage(combined_segments_gdf, collected_segments_path)
    else:
        logging.info("No new segments found in this run.")

    # --- Update Remaining Points ---
    if processed_indices:
        # Get the unique IDs of the points that were successfully processed from the *original* sampled GDF
        processed_point_ids = sampled_points_gdf.iloc[list(processed_indices)]['point_id'].tolist()

        logging.info(f"Removing {len(processed_point_ids)} processed points from the remaining list...")
        # Filter out the processed points from the main remaining points GDF
        remaining_points_updated_gdf = remaining_points_gdf[
            ~remaining_points_gdf['point_id'].isin(processed_point_ids)
        ]
        save_geopackage(remaining_points_updated_gdf, remaining_points_path)
        logging.info(f"{len(remaining_points_updated_gdf)} points remaining for future runs.")
    else:
        logging.warning("No points were successfully processed in this run. Remaining points file not updated.")


    logging.info("--- Segment Collection Run Finished ---")


if __name__ == "__main__":
    main()
