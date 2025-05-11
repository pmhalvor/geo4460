import subprocess
import webbrowser
import time
import json
import os
import sys
import requests
import datetime

from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

REDIRECT_URI = "http://localhost:3333/callback"


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TOKEN_PATH = os.path.join(BASE_DIR, "data", "token.json")
print(f"Token path: {TOKEN_PATH}")


def get_authorization_url():
    """Generate the Strava authorization URL."""
    auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&approval_prompt=force"
        f"&scope=read,activity:read_all"
    )
    return auth_url


def authorize():
    print("Starting flask server...")

    flask_app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "src",
        "strava",
        "flask_app.py",
    )

    token_dir = os.path.dirname(os.path.abspath(TOKEN_PATH))
    os.makedirs(token_dir, exist_ok=True)

    print(f"Running {flask_app_path}  with {sys.executable}...")
    flask_process = subprocess.Popen([sys.executable, flask_app_path])

    time.sleep(1)  # waits for server to be fully loaded

    auth_url = get_authorization_url()
    webbrowser.open(auth_url)

    print("Waiting for user to authorize...")

    while True:
        time.sleep(1)

        try:
            response = requests.get("http://localhost:3333/code")
            if response.status_code == 200:
                auth_code = response.text
                break
        except requests.exceptions.ConnectionError:
            pass

        print("Waiting for user to authorize...")

    flask_process.kill()

    token_data = exchange_code_for_token(auth_code)

    # Use absolute path for token file
    token_file_path = os.path.abspath(TOKEN_PATH)
    with open(token_file_path, "w") as f:
        json.dump(token_data, f, indent=4)

    print(f"Authorization successful! Token saved to {token_file_path}")


def exchange_code_for_token(auth_code):
    """Exchange the authorization code for an access token."""
    token_url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": auth_code,
        "grant_type": "authorization_code",
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to exchange token: {response.status_code}, {response.text}"
        )


def get_token():
    if not os.path.exists(TOKEN_PATH):
        authorize()

    with open(TOKEN_PATH) as f:
        token_data = json.load(f)

    if token_data.get("error", None):  # TODO do errors ever get stored?
        print("Error in token data:", token_data)
        return None

    if token_data.get("expires_at") < time.time():  # past expiration
        token_data = refresh_token(token_data)
        if not token_data:
            return None

    return token_data["access_token"]


def refresh_token(token_data):
    """Refresh the Strava access token using the refresh token."""

    token_url = "https://www.strava.com/oauth/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": token_data["refresh_token"],
    }

    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        refresh_token_data = response.json()

        refresh_token_data["athlete"] = token_data["athlete"]
        refresh_token_data["athlete"]["updated_at"] = datetime.datetime.now(
            datetime.UTC
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        with open(TOKEN_PATH, "w") as f:
            json.dump(refresh_token_data, f, indent=4)

        print("Access token refreshed successfully.")
        return refresh_token_data
    else:
        print(f"Failed to refresh token: {response.status_code} {response.text}")
        print("Trying to authorize again...")
        authorize()
        print("Need to get token again, this time reading the new token.json.")
        return None


if __name__ == "__main__":

    authorize()

    print("Getting token...")
    token = get_token()

    print("Token:", token)
