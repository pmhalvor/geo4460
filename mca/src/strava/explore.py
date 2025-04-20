import requests
import geopandas as gpd
import os
import pandas as pd
import polyline
import json  # Added import

from shapely.geometry import LineString
import folium
import webbrowser

# Try different import paths to work regardless of current directory
try:
    from src.strava.authorize import get_token
except ImportError:
    try:
        from authorize import get_token
    except ImportError:
        from strava.authorize import get_token

# ensure valid token is present
get_token()

# Get the base directory (project root) to make paths absolute
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Define paths with os.path.join to be platform-independent
SEGMENT_SHAPEFILE_PATH = os.path.join(
    BASE_DIR, "data", "segments", "segments_oslo_4258.shp"
)
SEGMENT_METADATA_PATH = os.path.join(
    BASE_DIR, "data", "segments", "segments_oslo.geojson"
)

# Define path for activities
ACTIVITY_DATA_PATH = os.path.join(BASE_DIR, "data", "activities")

# Create directories if they don't exist
os.makedirs(os.path.dirname(SEGMENT_SHAPEFILE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SEGMENT_METADATA_PATH), exist_ok=True)
os.makedirs(ACTIVITY_DATA_PATH, exist_ok=True)  # Added directory creation


def explore_segments(
    bounds=[  # default all of Oslo
        59.8181886681663,  # sw lat first
        10.42043828050879,  # sw lon second
        60.0142603407657,  # ne lat third
        11.007603658932084,  # ne lon fourth
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

    response = requests.get(url)  # no need for header

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to explore segments: {response.status_code} {response.text}")
        return None


def get_athlete_activities(per_page=100):
    """
    Fetch athlete activities from the Strava API.
    https://developers.strava.com/docs/reference/#api-Activities-getLoggedInAthleteActivities
    Fetches all activities page by page.
    """
    url = "https://www.strava.com/api/v3/athlete/activities"
    token = get_token()
    all_activities = []
    page = 1

    print("Fetching athlete activities...")

    while True:
        params = {
            "access_token": token,
            "page": page,
            "per_page": per_page,
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            activities = response.json()
            if not activities:  # No more activities found
                print("No more activities found.")
                break

            print(f"Fetched page {page} with {len(activities)} activities.")
            all_activities.extend(activities)
            page += 1

            # Small check to prevent infinite loops in case API behaves unexpectedly
            if len(activities) < per_page:
                print("Last page reached.")
                break
        elif response.status_code == 429:
            print("Rate limit exceeded. Please wait before making more requests.")
            print(
                f"NOTE! On rerun, make sure to start collection at page {page} to avoid duplicates."
            )
            # Consider adding a wait mechanism here if needed
            break
        else:
            print(f"Failed to fetch activities: {response.status_code} {response.text}")
            break  # Stop fetching on error

    print(f"Total activities fetched: {len(all_activities)}")
    return all_activities


def store_activities(activities, activity_type_filter="Ride"):
    """
    Store fetched activities as individual JSON files, filtering by type.
    """
    if not activities:
        print("No activities to store.")
        return

    stored_count = 0
    for activity in activities:
        if activity.get("type") == activity_type_filter:
            activity_id = activity.get("id")
            if not activity_id:
                print("Skipping activity with no ID.")
                continue

            file_path = os.path.join(ACTIVITY_DATA_PATH, f"activity_{activity_id}.json")
            try:
                with open(file_path, "w") as f:
                    json.dump(activity, f, indent=4)
                stored_count += 1
            except IOError as e:
                print(f"Error writing file {file_path}: {e}")

    print(
        f"Stored {stored_count} activities of type '{activity_type_filter}' in {ACTIVITY_DATA_PATH}"
    )


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

        combined_gdf = gpd.GeoDataFrame(
            pd.concat([previous, gdf], ignore_index=True), crs="EPSG:4326"
        )
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


def store_segments(segments):
    """
    Store the segments in a geodata file.
    """
    linestring_points = parse_segments_points(segments)

    gdf = gpd.GeoDataFrame(data=segments, geometry=linestring_points, crs="EPSG:4326")

    update_geodata(gdf)
    print("Segments stored in geodata file.")


def show_segments(segment_metadata_path=SEGMENT_METADATA_PATH):
    """
    Show the segments on an interactive map.
    """
    if os.path.exists(segment_metadata_path):
        segments = gpd.read_file(segment_metadata_path)
        print(f"Loaded {len(segments)} segments from {segment_metadata_path}")

        # Create a folium map centered around Oslo
        m = folium.Map(location=[59.91, 10.75], zoom_start=12)

        # Add each segment to the map
        for _, segment in segments.iterrows():
            if segment.geometry and segment.geometry.geom_type == "LineString":
                folium.PolyLine(
                    locations=[
                        (coord[1], coord[0]) for coord in segment.geometry.coords
                    ],
                    color="blue",
                    weight=2,
                    opacity=0.7,
                    tooltip=segment.get("name", "Segment"),
                ).add_to(m)

        # Save and display the map
        map_path = "segments_map.html"
        m.save(map_path)
        print(
            f"Map saved to {os.path.abspath(map_path)}. "
            "Opening the map in your default browser..."
        )
        webbrowser.open("file://" + os.path.abspath(map_path))
    else:
        print(f"No segment data found at {segment_metadata_path}.")


def get_segment_details(segment):
    """
    Get segment details using the Strava API.
    https://developers.strava.com/docs/reference/#api-Segments-getSegmentById
    """
    url = f"https://www.strava.com/api/v3/segments/{segment['id']}"
    token = get_token()

    params = {
        "access_token": token,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        print("Rate limit exceeded. Please wait before making more requests.")
        return response.status_code
    else:
        print(f"Failed to get segment details: {response.status_code} {response.text}")
        return None


if __name__ == "__main__":
    # --- Example usage for fetching and storing activities ---
    # fetched_activities = get_athlete_activities()
    # if fetched_activities:
    #    store_activities(fetched_activities, activity_type_filter="Ride")
    # --------------------------------------------------------

    # --- Original example usage for segments ---
    # from locations import locations

    # example = locations["frognerparken"]

    # segments = explore_segments(example["bounds"]).get("segments", None)

    # if segments:
    #     store_segments(segments)

    # show_segments()

    if os.path.exists(SEGMENT_METADATA_PATH):
        segments = gpd.read_file(SEGMENT_METADATA_PATH)

    for i, segment in enumerate(segments.iloc[:5].iterrows()):
        print(f"{i+1}. {segment[1].name}")
        print(segment)
        segment_details = get_segment_details(segment[1])
        print(segment_details)
        print(segment_details.keys())
        print("\n")
