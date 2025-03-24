import subprocess
import webbrowser
import time
import requests
from flask import Flask, request, redirect, jsonify
from credentials import CLIENT_ID, CLIENT_SECRET

app = Flask(__name__)


@app.route('/callback')
def callback():
    """Callback route to handle the redirect from Strava."""
    auth_code = request.args.get('code')
    if not auth_code:
        return "Authorization code not found in the request.", 400
    try:
        token_response = exchange_code_for_token(auth_code)
        with open('token.json', 'w') as f:
            import json
            json.dump(token_response, f)
        return "Authorization successful! You can close this tab."
    except Exception as e:
        return f"Error: {e}", 500


def exchange_code_for_token(auth_code):
    """Exchange the authorization code for an access token."""
    token_url = "https://www.strava.com/oauth/token"
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': auth_code,
        'grant_type': 'authorization_code'
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to exchange token: {response.status_code}, {response.text}")



if __name__ == "__main__":
    app.run(port=3333)