import time
import logging
import numpy as np
import requests  # Import requests to catch HTTPError
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString # Moved import here
from polyline import decode # Import polyline decode globally

from src.strava.explore import explore_segments, update_geodata
from src.strava.locations import locations
from src.traffic.stations import get_oslo_stations
from src.config import settings # Import settings for configuration

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
                # except keyboard interpupt:
                except KeyboardInterrupt:
                    logging.info("    Process interrupted by user.")
                    break
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
                try:
                    time.sleep(request_delay)
                except KeyboardInterrupt:
                    logging.info("    Process interrupted by user.")
                    break
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

    logging.info(f"Example segment: {segments_list[0]}")
    df = pd.DataFrame(segments_list)

    # Decode the polyline string from the 'points' column
    # Handle potential errors during decoding or if 'points' is missing/None
    def safe_decode(polyline_str):
        if pd.isna(polyline_str) or not polyline_str:
            return None
        try:
            # Polyline library expects (lat, lon) pairs
            return decode(polyline_str)
        except Exception as e:
            logging.warning(f"Could not decode polyline '{polyline_str[:20]}...': {e}")
            return None

    df['decoded_coords'] = df['points'].apply(safe_decode)

    # Create LineString geometry from decoded coordinates
    # Polyline decode gives (lat, lon), Shapely LineString expects (lon, lat)
    def create_linestring(coords):
        if coords and len(coords) >= 2:
            try:
                # Swap lat/lon for Shapely
                return LineString([(lon, lat) for lat, lon in coords])
            except Exception as e:
                logging.warning(f"Could not create LineString from coords: {e}")
                return None
        elif coords and len(coords) == 1:
             # Handle single point case if necessary, maybe return a Point or None
             logging.warning(f"Segment only has one point, cannot create LineString: {coords}")
             # return Point(coords[0][1], coords[0][0]) # Option: return Point
             return None
        return None

    df['geometry'] = df['decoded_coords'].apply(create_linestring)

    # Drop rows where geometry creation failed
    df = df.dropna(subset=['geometry'])

    if df.empty:
        logging.warning("No valid geometries could be created from the segments list.")
        return gpd.GeoDataFrame([], columns=df.columns.tolist() + ['geometry'], crs=settings.processing.output_crs_epsg)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326") # Assume Strava returns WGS84 (lat/lon)

    # Reproject to the target CRS specified in settings
    gdf = gdf.to_crs(epsg=settings.processing.output_crs_epsg)

    # Select and potentially reorder columns for the final output GDF
    # Keep essential segment info + geometry
    cols_to_keep = ['id', 'name', 'distance', 'avg_grade', 'climb_category', 'start_latlng', 'end_latlng', 'points', 'geometry']
    # Filter columns that actually exist in the DataFrame
    final_cols = [col for col in cols_to_keep if col in gdf.columns]
    gdf = gdf[final_cols]

    return gdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    oslo_stations = get_oslo_stations()

    for place in locations.keys():
        try:
            oslo_bounds = locations[place]["bounds"]
            oslo_segments = explore_segments(oslo_bounds).get("segments", None)

            oslo_segments_gdf = segments_to_gdf(oslo_segments).to_crs("EPSG:4326")

            update_geodata(oslo_segments_gdf)
            print(
                f"Nice exploring! 🔎" \
                f" Found {len(oslo_segments_gdf)} new segments in {place}!"
            )
        except AttributeError as e:
            logging.error(f"Limit rate reached for {place}: {e}")
            break