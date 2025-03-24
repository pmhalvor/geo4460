import subprocess
import webbrowser
import time
import json
from credentials import CLIENT_ID, ACCESS_TOKEN, REFRESH_TOKEN
import requests
from credentials import CLIENT_SECRET


REDIRECT_URI = 'http://localhost:3333/callback'
TOKEN_PATH = 'token.json'

def get_authorization_url():
    """Generate the Strava authorization URL."""
    # http://www.strava.com/oauth/authorize
    # ?client_id=[REPLACE_WITH_YOUR_CLIENT_ID]
    # &response_type=code
    # &redirect_uri=http://localhost/exchange_token
    # &approval_prompt=force&scope=read
    auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&approval_prompt=force"
        f"&scope=read"
    )
    return auth_url

def get_token():
    with open(TOKEN_PATH) as f:
        token_data = json.load(f)
    
    if token_data.get("error", None):
        print("Error in token data:", token_data)
    
    if token_data.get("expires_at") > time.time():
        refresh_token(token_data)

def refresh_token(token_data):
    # """Refresh the Strava access token using the refresh token."""

    # token_url = "https://www.strava.com/oauth/token"
    # payload = {
    #     'client_id': CLIENT_ID,
    #     'client_secret': CLIENT_SECRET,
    #     'grant_type': 'refresh_token',
    #     'refresh_token': REFRESH_TOKEN
    # }

    # response = requests.post(token_url, data=payload)
    # if response.status_code == 200:
    #     token_data = response.json()
    #     new_access_token = token_data['access_token']
    #     new_refresh_token = token_data['refresh_token']

    #     # Save the new tokens to credentials or a secure location
    #     with open('token.json', 'w') as f:
    #         f.write(response.text)

    #     print("Access token refreshed successfully.")
    #     return new_access_token
    # else:
    #     print(f"Failed to refresh token: {response.status_code} {response.text}")
    #     return None


if __name__ == "__main__":

    flask_process = subprocess.Popen(["python", "mca/src/strava_flask_app.py"])
    time.sleep(3)

    auth_url = get_authorization_url()
    webbrowser.open(auth_url)

    print("Waiting for user to authorize...")

    while True:
        time.sleep(1)
        try:
            with open('token.json') as f:
                break
        except FileNotFoundError:
            pass

    flask_process.kill()
    print("Authorization successful! Token found at token.json.")