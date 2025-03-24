import requests
import geopandas as gpd
import polyline

from shapely.geometry import LineString
from authorize import get_token


# ensure valid token is present
get_token()


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
    

def parse_segment_points(segment):
    """
    Parse a segment from the Strava API response.
    """
    
    decoded_points = polyline.decode(segment["points"])

    return LineString(decoded_points)

if __name__ == "__main__":
    from locations import locations

    token = get_token()
    example = locations["ring2"]

    all_segments = explore_segments(example["bounds"])
    all_points = []

    if all_segments:
        print(f"Found segments around {example['name']}:\n")
        for i, segment in enumerate(all_segments['segments']):
            print(f"Segment {i+1}:")
            for key, value in segment.items():
                print(f"{key}: {value}")
            print("-"*50)

            segment_points = parse_segment_points(segment)
            all_points.append(segment_points)

    else:
        print("Failed to explore segments.")

    # save all points to a GeoDataFrame
    gdf = gpd.GeoDataFrame(data=all_segments, geometry=all_points)
    # append to the existing file
    gdf.to_file("segments.geojson", driver="GeoJSON")
