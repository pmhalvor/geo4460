import time
import logging
import numpy as np
import requests  # Import requests to catch HTTPError
import pandas as pd
import geopandas as gpd

from src.strava.explore import explore_segments, store_segments
from src.strava.locations import locations
from src.traffic.stations import get_oslo_stations
from src.config import settings # Import settings for configuration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_km_to_degree_at_latitude(km, latitude):
    """Convert kilometers to degrees at a given latitude"""
    # 1 degree of latitude is approximately 111.32 km
    latitude_degree = km / 111.32

    # 1 degree of longitude varies with latitude
    longitude_degree = km / (111.32 * np.cos(np.radians(latitude)))

    return float(latitude_degree), float(longitude_degree)


def create_bounds_around_point(point, areas_km2=[1, 2, 5]):
    """Create bounds of different sizes around a point"""
    lon, lat = point.x, point.y
    bounds_list = []

    for area in areas_km2:
        side_length = np.sqrt(area)

        lat_offset, lon_offset = calculate_km_to_degree_at_latitude(
            side_length / 2, lat
        )

        # Create bounds [min_lat, min_lng, max_lat, max_lng]
        bounds = [
            lat - lat_offset,  # min_lat
            lon - lon_offset,  # min_lng
            lat + lat_offset,  # max_lat
            lon + lon_offset,  # max_lng
        ]
        bounds_list.append(bounds)

    return bounds_list


def explore_segments_with_multiple_bounds(gdf, area_sizes=[1, 2, 5], point_col='geometry'):
    """
    Explore segments around points with multiple bound sizes, handling rate limits.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing points to search around.
        area_sizes (list): List of area sizes (in km²) to create search bounds.
        point_col (str): Name of the geometry column in gdf containing the points.

    Returns:
        list: A list of unique segment dictionaries found.
    """
    all_segments_data = []
    unique_segment_ids = set()
    processed_point_indices = set() # Keep track of points processed in this run

    max_retries = settings.processing.segment_collection_max_retries
    retry_delay = settings.processing.segment_collection_retry_delay
    request_delay = settings.processing.strava_api_request_delay

    total_points = len(gdf)
    for idx, row in gdf.reset_index().iterrows():
        point = row[point_col]
        point_id = row.get('point_id', idx) # Use a unique ID if available, else index
        logging.info(f"Processing point {idx + 1}/{total_points} (ID: {point_id})...")

        bounds_list = create_bounds_around_point(point, area_sizes)
        point_processed_successfully = False

        for i, bounds in enumerate(bounds_list):
            logging.debug(f"  Exploring {area_sizes[i]} km² area around point {point_id} (Bounds: {bounds})")
            retries = 0
            success = False
            while retries <= max_retries:
                try:
                    # Make the API call
                    response_data = explore_segments(bounds) # Assuming this function makes the actual request
                    result = response_data.get("segments", [])
                    success = True
                    logging.debug(f"    API call successful for bounds {i+1}/{len(bounds_list)}.")
                    break # Exit retry loop on success

                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429 or response_data==429:
                        retries += 1
                        logging.warning(
                            f"    Rate limit hit (429) for point {point_id}, bounds {i+1}. "
                            f"Retry {retries}/{max_retries} after {retry_delay} seconds."
                        )
                        time.sleep(retry_delay)
                    else:
                        # Come to think of it, I don't think this will ever be triggered
                        # bc explore_segments never raises any error
                        # Handle other HTTP errors (e.g., 401 Unauthorized, 500 Server Error)
                        logging.error(f"    HTTP Error {e.response.status_code} for point {point_id}, bounds {i+1}: {e}")
                        # Decide whether to retry or give up based on the error
                        # For now, we'll stop retrying for non-429 errors
                        break # exit while
                except AttributeError as e:
                    logging.error(f"    Limit rate reached for row {idx}/{total_points}, bounds {i+1}")
                    break # exit while
                except Exception as e:
                    # Catch any other unexpected errors during the API call
                    logging.error(f"    Unexpected error during API call for point {point_id}, bounds {i+1}: {e}")
                    break # exit while

            if success:
                new_segments_count = 0
                for segment in result:
                    segment_id = segment.get("id")
                    if segment_id and segment_id not in unique_segment_ids:
                        unique_segment_ids.add(segment_id)
                        all_segments_data.append(segment)
                        new_segments_count += 1
                logging.debug(f"    Found {new_segments_count} new unique segments.")
                # Add a small delay between successful requests to be polite
                time.sleep(request_delay)
                point_processed_successfully = True # Mark point as processed if at least one bound succeeded
            else:
                logging.error(
                    f"  Failed to retrieve segments for point {point_id}, bounds {i+1} "
                    f"after {max_retries} retries."
                )
                # If even one bound fails after retries, we might consider the point not fully processed
                point_processed_successfully = False
                break # Stop trying other bounds for this point if one fails definitively
        if point_processed_successfully:
             processed_point_indices.add(idx) # Add original gdf index
        else:
            break 

    logging.info(f"Finished exploring. Found {len(all_segments_data)} total unique segments.")
    return all_segments_data, processed_point_indices


def segments_to_gdf(segments_list):
    """Converts a list of segment dictionaries (with polyline) to a GeoDataFrame."""
    if not segments_list:
        return gpd.GeoDataFrame([], columns=['id', 'name', 'polyline', 'geometry'], crs=settings.processing.output_crs_epsg) # Ensure CRS

    df = pd.DataFrame(segments_list)

    # TODO: Add polyline decoding here if explore_segments returns encoded polylines
    # Example placeholder: assumes 'polyline' column exists and needs decoding
    # from polyline import decode
    # df['decoded_polyline'] = df['polyline'].apply(lambda x: decode(x) if x else [])
    # df['geometry'] = df['decoded_polyline'].apply(lambda coords: LineString([(lon, lat) for lat, lon in coords]) if coords else None)

    # Placeholder: If 'start_latlng' and 'end_latlng' are available directly
    from shapely.geometry import Point, LineString
    # Check if 'points' (decoded polyline) or 'start_latlng'/'end_latlng' exist
    if 'points' in df.columns and isinstance(df['points'].iloc[0], str): # Assuming 'points' is the encoded polyline string
         # Need polyline decoding logic here!
         logging.warning("Polyline decoding not implemented in segments_to_gdf. Geometry might be incorrect.")
         # As a fallback, maybe use start/end points if available?
         if 'start_latlng' in df.columns and 'end_latlng' in df.columns:
              df['geometry'] = df.apply(lambda row: LineString([Point(row['start_latlng'][1], row['start_latlng'][0]), Point(row['end_latlng'][1], row['end_latlng'][0])]) if row['start_latlng'] and row['end_latlng'] else None, axis=1)
         else:
              df['geometry'] = None # Cannot create geometry
    elif 'start_latlng' in df.columns and 'end_latlng' in df.columns:
         # Create simple LineString from start/end if polyline isn't available/decoded
         df['geometry'] = df.apply(lambda row: LineString([Point(row['start_latlng'][1], row['start_latlng'][0]), Point(row['end_latlng'][1], row['end_latlng'][0])]) if row['start_latlng'] and row['end_latlng'] else None, axis=1)
    else:
         logging.error("Could not find suitable columns ('points', or 'start_latlng'/'end_latlng') to create geometry.")
         df['geometry'] = None


    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326") # Assume Strava returns WGS84
    gdf = gdf.to_crs(epsg=settings.processing.output_crs_epsg) # Reproject to target CRS
    gdf = gdf.dropna(subset=['geometry']) # Drop segments where geometry couldn't be created
    return gdf


if __name__ == "__main__":
    from shapely.geometry import Point
    # The main script logic will be in the new file in src/collect/
    logging.info("Running segments.py directly for testing purposes.")

    # Example: Create a dummy GeoDataFrame with one point in Oslo
    oslo_center_lon, oslo_center_lat = 10.7522, 59.9139
    dummy_gdf = gpd.GeoDataFrame(
        [{'id': 1, 'geometry': Point(oslo_center_lon, oslo_center_lat)}],
        crs="EPSG:4326"
    ).to_crs(epsg=settings.processing.output_crs_epsg) # Convert to project CRS for bounds calculation

    logging.info(f"Dummy GDF created with CRS: {dummy_gdf.crs}")
    logging.info(f"Point coordinates: {dummy_gdf.geometry.iloc[0]}")


    # Test bounds calculation
    test_bounds = create_bounds_around_point(dummy_gdf.geometry.iloc[0], areas_km2=[1])
    logging.info(f"Test bounds (1 km²): {test_bounds}") # Should be in Lat/Lon (EPSG:4326) format for Strava API

    # Test exploring (will make actual API calls if token is valid)
    # Note: This might hit rate limits quickly if run repeatedly.
    try:
        # Need to convert point back to Lat/Lon for create_bounds_around_point
        dummy_gdf_wgs84 = dummy_gdf.to_crs("EPSG:4326")
        found_segments, processed_indices = explore_segments_with_multiple_bounds(
            dummy_gdf_wgs84, area_sizes=[0.5, 1], point_col='geometry' # Use smaller areas for testing
        )
        logging.info(f"Found {len(found_segments)} segments near Oslo center.")
        logging.info(f"Processed point indices: {processed_indices}")

        if found_segments:
            segments_gdf = segments_to_gdf(found_segments)
            logging.info(f"Converted {len(segments_gdf)} segments to GeoDataFrame.")
            # Optionally save for inspection
            # test_output_path = settings.paths.output_dir / "test_collected_segments.gpkg"
            # segments_gdf.to_file(test_output_path, driver="GPKG")
            # logging.info(f"Saved test segments to {test_output_path}")

    except Exception as e:
        logging.error(f"Error during testing: {e}", exc_info=True)

    # Original example code (commented out as it's superseded by the new script's logic)
    # oslo_stations = get_oslo_stations()
    # oslo_bounds = locations["oslo"]["bounds"]
    # oslo_segments = explore_segments(oslo_bounds).get("segments", None)
    # segments = explore_segments_with_multiple_bounds(
    #     oslo_stations, area_sizes=[3, 5, 10]
    # )
    # store_segments(segments) # store_segments might need updates or replacement
    # print(f"Nice exploring! 🔎" f" Found {len(segments)} new segments!")
#                 segment_id = segment.get("id")
#                 if segment_id not in unique_segment_ids:
#                     unique_segment_ids.add(segment_id)
#                     new_segments.append(segment)

#             print(f"Found {len(new_segments)} new segments")
#             all_segments.extend(new_segments)

#     return all_segments


# if __name__ == "__main__":
#     oslo_stations = get_oslo_stations()

#     oslo_bounds = locations["oslo"]["bounds"]
#     oslo_segments = explore_segments(oslo_bounds).get("segments", None)

#     segments = explore_segments_with_multiple_bounds(
#         oslo_stations, area_sizes=[3, 5, 10]
#     )
#     store_segments(segments)
#     print(f"Nice exploring! 🔎" f" Found {len(segments)} new segments!")
