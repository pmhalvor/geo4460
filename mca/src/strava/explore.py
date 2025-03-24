import requests
import geopandas as gpd
import os
import pandas as pd
import polyline

from shapely.geometry import LineString
from authorize import get_token


# ensure valid token is present
get_token()


SEGMENT_SHAPEFILE_PATH = "mca/data/segments/segments_oslo_4258.shp"
SEGMENT_METADATA_PATH = "mca/data/segments/segments_oslo.geojson"


def explore_segments(
        bounds=[ # default all of Oslo
            59.8181886681663,   # sw lat first
            10.42043828050879,  # sw lon second
            60.0142603407657,   # ne lat third
            11.007603658932084, # ne lon fourth
        ],
    ):
    """
    Explore segments using the Strava API.
    https://developers.strava.com/docs/reference/#api-Segments-exploreSegments 
    
    bounds* (array[number]):
    The latitude and longitude for two points describing a rectangular boundary
     for the search: [
        southwest corner latitutde, 
        southwest corner longitude, 
        northeast corner latitude, 
        northeast corner longitude
    ]

    activity_type (string):
    Desired activity type. options: running or riding

    min_cat (integer):
    The minimum climbing category.

    max_cat (integer):
    The maximum climbing category.

    """
    url = "https://www.strava.com/api/v3/segments/explore"
    token = get_token()

    params = {
        "bounds": bounds,
        "activity_type": "riding",
        "access_token": token,
    }

    # manually parse params as query str (for strava api specific formatting)
    params = "&".join([f"{k}={v}" for k, v in params.items()])
    url = f"{url}?{params}".replace(" ", "").replace("[", "").replace("]", "")
    print(f"Searching: {url} ... \n")

    response = requests.get(url) # no need for header

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to explore segments: {response.status_code} {response.text}")
        return None
    

def parse_segments_points(segments):
    """
    Convert encoded polyline points to LineString objects.
    """
    all_points = []
    for segment in segments:
        # print(f"Segment: {segment['name']}", segment["points"])
        decoded_points = polyline.decode(segment["points"], geojson=True)
        all_points.append(LineString(decoded_points))

    return all_points


def update_geodata(
        gdf, 
        segment_shapefile_path=SEGMENT_SHAPEFILE_PATH,
        segment_metadata_path=SEGMENT_METADATA_PATH,
    ):
    """
    Add new data to previously stored geo data.
    """
    print(f"{len(gdf.id.values)} new ids to store")

    if os.path.exists(segment_metadata_path):
        previous = gpd.read_file(segment_metadata_path)
        print(f"{len(previous.id.values)} previous ids loaded")
        
        combined_gdf = gpd.GeoDataFrame(pd.concat([previous, gdf], ignore_index=True), crs="EPSG:4326")
        combined_gdf.drop_duplicates(subset="id", inplace=True)
    else:
        combined_gdf = gdf

    print(f"{len(combined_gdf.id.values)} updated ids")
    
    combined_gdf.to_file(segment_metadata_path, driver="GeoJSON")
    print(f"Segment data file {segment_metadata_path} updated.")

    # isolate only the LineString and ids
    points_gdf = combined_gdf[["id", "geometry"]]
    points_gdf_4258 = points_gdf.to_crs(epsg=4258)
    points_gdf_4258.to_file(segment_shapefile_path, driver="ESRI Shapefile")


def store_segments(segments, linestring_points):
    """
    Store the segments in a geodata file.
    """
    gdf = gpd.GeoDataFrame(data=segments, geometry=linestring_points, crs="EPSG:4326")

    update_geodata(gdf)
    print("Segments stored in geodata file.")


if __name__ == "__main__":
    from locations import locations

    token = get_token()
    example = locations["e18"]

    segments = explore_segments(example["bounds"]).get("segments", None)

    if segments:
        linestring_points = parse_segments_points(segments)
        store_segments(segments, linestring_points)

