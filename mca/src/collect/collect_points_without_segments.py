import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString

from src.config import settings
from src.utils import load_vector_data, save_vector_data, reproject_gdf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

        # Filter for simple car roads
        filtered_roads = roads_gdf[roads_gdf[typeveg_field] == simple_road_value].copy()
        logger.info(f"Filtered down to {len(filtered_roads)} simple car roads ('{simple_road_value}').")

        if filtered_roads.empty:
            logger.warning("No simple car roads found in the N50 layer.")
            return None

        # Ensure target CRS
        filtered_roads = reproject_gdf(filtered_roads, target_crs)
        logger.info(f"Ensured CRS {target_crs} for simple roads.")
        return filtered_roads

    except Exception as e:
        logger.error(f"Error loading or filtering N50 roads from {gdb_path} layer {layer_name}: {e}", exc_info=True)
        return None

def load_and_combine_segments(target_crs: str) -> gpd.GeoDataFrame | None:
    """Loads and combines segments from GeoJSON and collected GeoPackage."""
    logger.info("Loading and combining all known Strava segments...")
    segment_id_field = settings.input_data.segment_id_field

    # Load Original GeoJSON
    geojson_path = settings.paths.strava_segments_geojson
    logger.info(f"Loading original segments from: {geojson_path}")
    geojson_gdf = load_vector_data(geojson_path)
    if geojson_gdf is not None:
        if segment_id_field not in geojson_gdf.columns:
            logger.warning(f"Segment ID field '{segment_id_field}' not found in {geojson_path}. Skipping.")
            geojson_gdf = None
        else:
            geojson_gdf = reproject_gdf(geojson_gdf, target_crs)
            logger.info(f"Loaded and ensured CRS {target_crs} for {len(geojson_gdf)} original segments.")
    else:
        logger.warning(f"Could not load original segments from {geojson_path}")

    # Load Newly Collected GeoPackage
    gpkg_path = settings.paths.collected_segments_gpkg
    logger.info(f"Loading newly collected segments from: {gpkg_path}")
    gpkg_gdf = load_vector_data(gpkg_path)
    if gpkg_gdf is not None:
        if segment_id_field not in gpkg_gdf.columns:
            logger.warning(f"Segment ID field '{segment_id_field}' not found in {gpkg_path}. Skipping.")
            gpkg_gdf = None
        else:
            gpkg_gdf = reproject_gdf(gpkg_gdf, target_crs)
            logger.info(f"Loaded and ensured CRS {target_crs} for {len(gpkg_gdf)} newly collected segments.")
    else:
        logger.warning(f"Could not load collected segments from {gpkg_path}")

    # Combine Data
    if geojson_gdf is not None and gpkg_gdf is not None:
        logger.info("Combining original and collected segments...")
        if not geojson_gdf.crs.equals(gpkg_gdf.crs):
            logger.error(f"CRS mismatch before combining: {geojson_gdf.crs} vs {gpkg_gdf.crs}. Aborting.")
            return None
        combined_gdf = pd.concat([geojson_gdf, gpkg_gdf], ignore_index=True, sort=False)
        combined_gdf = combined_gdf.drop_duplicates(subset=[segment_id_field], keep='last')
        logger.info(f"Combined data has {len(combined_gdf)} unique segments (CRS: {combined_gdf.crs}).")
    elif geojson_gdf is not None:
        logger.info("Using only original segments.")
        combined_gdf = geojson_gdf
    elif gpkg_gdf is not None:
        logger.info("Using only newly collected segments.")
        combined_gdf = gpkg_gdf
    else:
        logger.error("No segment data could be loaded.")
        return None

    # Final check for geometry and CRS
    if 'geometry' not in combined_gdf.columns or combined_gdf.geometry.isnull().all():
        logger.error("Combined segments lack valid geometry column.")
        return None
    if not combined_gdf.crs or not combined_gdf.crs.equals(target_crs):
        logger.error(f"Combined segments CRS is not set or does not match target {target_crs}.")
        return None

    # Keep only essential columns (geometry and ID)
    essential_cols = [segment_id_field, 'geometry']
    cols_to_keep = [col for col in essential_cols if col in combined_gdf.columns]
    return combined_gdf[cols_to_keep]


def convert_lines_to_points(gdf: gpd.GeoDataFrame, segment_length: float = 50.0) -> gpd.GeoDataFrame:
    """
    Converts LineString/MultiLineString geometries in a GeoDataFrame to points at regular intervals.
    (Adapted from collect_segments_from_diff.py)
    """
    points_list = []
    original_crs = gdf.crs

    if not original_crs:
        logger.warning("Input GeoDataFrame for line-to-point conversion has no CRS set. Assuming project default.")
        original_crs = f"EPSG:{settings.processing.output_crs_epsg}" # Fallback

    logger.info(f"Converting {len(gdf)} line features to points with interval {segment_length}m...")
    for index, row in gdf.iterrows():
        geometry = row.geometry
        lines_to_process = []

        if isinstance(geometry, LineString):
            lines_to_process.append(geometry)
        elif isinstance(geometry, MultiLineString):
            lines_to_process.extend(list(geometry.geoms))
        else:
            logger.warning(f"Skipping geometry at index {index} - not LineString or MultiLineString (Type: {type(geometry)}).")
            continue

        for i, line in enumerate(lines_to_process):
             if not isinstance(line, LineString) or line.is_empty or line.length == 0:
                 logger.debug(f"Skipping empty or invalid sub-line {i} in feature {index}")
                 continue

             distances = np.arange(0, line.length, segment_length)
             if distances.size == 0 or distances[-1] < line.length:
                 distances = np.append(distances, line.length)

             for dist in distances:
                 point = line.interpolate(dist)
                 sub_line_id = f"_part_{i}" if isinstance(geometry, MultiLineString) else ""
                 point_id = f"road_{index}{sub_line_id}_dist_{dist:.0f}"
                 # Include original road attributes if needed later
                 # point_attrs = row.drop('geometry').to_dict()
                 # points_list.append({'point_id': point_id, 'original_road_index': index, 'geometry': point, **point_attrs})
                 points_list.append({'point_id': point_id, 'original_road_index': index, 'geometry': point})


    if not points_list:
        logger.warning("No points generated from LineString or MultiLineString geometries.")
        return gpd.GeoDataFrame([], columns=['point_id', 'original_road_index', 'geometry'], crs=original_crs)

    points_gdf = gpd.GeoDataFrame(points_list, crs=original_crs)
    logger.info(f"Converted {len(gdf)} line features to {len(points_gdf)} points.")
    return points_gdf

def main():
    """Main script logic."""
    logger.info("--- Starting Collection: Points on Simple Roads without Segments ---")

    target_crs = f"EPSG:{settings.processing.output_crs_epsg}"
    output_path = settings.paths.points_without_segments_gpkg
    segment_buffer_distance = 20.0 # Meters - how close a point needs to be to a segment to be excluded
    point_interval = 50.0 # Meters - spacing of points along roads

    # --- 1. Load Simple Roads ---
    simple_roads_gdf = load_n50_simple_roads(target_crs)
    if simple_roads_gdf is None or simple_roads_gdf.empty:
        logger.error("Failed to load simple roads. Exiting.")
        return

    # --- 2. Load All Segments ---
    all_segments_gdf = load_and_combine_segments(target_crs)
    if all_segments_gdf is None or all_segments_gdf.empty:
        logger.error("Failed to load any segments. Exiting.")
        return

    # --- 3. Convert Roads to Points ---
    road_points_gdf = convert_lines_to_points(simple_roads_gdf, segment_length=point_interval)
    if road_points_gdf.empty:
        logger.error("Failed to generate points from roads. Exiting.")
        return

    # --- 4. Buffer Segments ---
    logger.info(f"Buffering {len(all_segments_gdf)} segments by {segment_buffer_distance}m...")
    try:
        # Ensure segments geometry is valid before buffering
        invalid_segments = ~all_segments_gdf.geometry.is_valid
        if invalid_segments.any():
            logger.warning(f"Found {invalid_segments.sum()} invalid segment geometries. Attempting buffer(0) fix...")
            all_segments_gdf.loc[invalid_segments, 'geometry'] = all_segments_gdf.loc[invalid_segments].geometry.buffer(0)
            # Remove any that became empty
            all_segments_gdf = all_segments_gdf[~all_segments_gdf.geometry.is_empty]
            logger.info(f"{len(all_segments_gdf)} segments remaining after validity fix.")

        segment_buffers = all_segments_gdf.geometry.buffer(segment_buffer_distance)
        segment_buffers_gdf = gpd.GeoDataFrame(geometry=segment_buffers, crs=target_crs)
        # Dissolve buffers into a single MultiPolygon for efficiency in spatial join
        dissolved_buffers = segment_buffers_gdf.unary_union
        dissolved_gdf = gpd.GeoDataFrame([1], geometry=[dissolved_buffers], crs=target_crs)
        logger.info("Segment buffering and dissolving complete.")
    except Exception as e:
        logger.error(f"Error buffering segments: {e}", exc_info=True)
        return

    # --- 5. Identify Points Outside Segment Buffers ---
    logger.info(f"Identifying which of the {len(road_points_gdf)} road points fall outside segment buffers...")
    try:
        # Perform spatial join: find points WITHIN the dissolved buffer
        points_within = gpd.sjoin(road_points_gdf, dissolved_gdf, how='inner', predicate='within')
        points_within_ids = set(points_within['point_id']) # Use the unique point ID

        # Filter the original points to keep only those NOT in the 'within' set
        points_outside_gdf = road_points_gdf[~road_points_gdf['point_id'].isin(points_within_ids)].copy()

        logger.info(f"Found {len(points_outside_gdf)} points on simple roads that are not within {segment_buffer_distance}m of any known segment.")

    except Exception as e:
        logger.error(f"Error during spatial join to find points outside buffers: {e}", exc_info=True)
        return

    # --- 6. Save Results ---
    if not points_outside_gdf.empty:
        logger.info(f"Saving {len(points_outside_gdf)} points without nearby segments to: {output_path}")
        save_vector_data(points_outside_gdf, output_path)
    else:
        logger.info("No points found outside segment buffers. No output file created.")

    logger.info("--- Collection Script Finished ---")


if __name__ == "__main__":
    main()
