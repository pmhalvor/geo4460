import requests


def explore_segments(token):
    """
    Explore segments using the Strava API.
    
    bounds* array[number]
    The latitude and longitude for two points describing a rectangular boundary for the search: [southwest corner latitutde, southwest corner longitude, northeast corner latitude, northeast corner longitude]

    activity_type string (query)
    Desired activity type. options: running or riding

    min_cat integer (query)
    The minimum climbing category.

    max_cat integer (query)
    The maximum climbing category.

    """
    url = f"https://www.strava.com/api/v3/segments/explore?bounds=36.372975,-94.220234,36.415949,-94.183670&access_token={token}"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    params = {
        "bounds": [
            10.42043828050879,
            59.8181886681663,
            11.007603658932084,
            60.0142603407657,
        ],
        # "activity_type": "riding",
        "activity_type": "running",
        "min_cat": 0,
        "max_cat": 10
    }

    # mnually add params as queyr string
    params = "&".join([f"{k}={v}" for k, v in params.items()])
    # url = f"{url}?{params}".replace(" ", "")
    print("Searching:", url)

    response = requests.get(url, headers=headers)# , params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to explore segments: {response.status_code} {response.text}")
        return None
    

if __name__ == "__main__":
    from credentials import ACCESS_TOKEN
    segments = explore_segments(ACCESS_TOKEN)
    if segments:
        print("Found segments:")
        print(segments['segments'])
        for segment in segments['segments']:
            print(segment['name'], segment.keys())
    else:
        print("Failed to explore segments.")
