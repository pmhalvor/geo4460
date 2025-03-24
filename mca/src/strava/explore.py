import requests
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
    print("Searching:", url)

    response = requests.get(url) # no need for header

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to explore segments: {response.status_code} {response.text}")
        return None
    

if __name__ == "__main__":
    from locations import filipstad

    token = get_token()

    segments = explore_segments(filipstad)
    
    if segments:
        print("Found segments:")
        print(segments['segments'])
        for i, segment in enumerate(segments['segments']):
            print(f"Segment {i+1}:")
            for key, value in segment.items():
                print(f"{key}: {value}")
            print("-"*50)
    else:
        print("Failed to explore segments.")
