import os

from flask import Flask, request
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")


@app.route("/callback")
def callback():
    """Callback route to handle the redirect from Strava."""
    auth_code = request.args.get("code")
    if not auth_code:
        return "Authorization code not found in the request.", 400

    with open(".auth_code", "w") as f:
        f.write(auth_code)
    return "Authorization code saved! You can close this tab."


@app.route("/code")
def code():
    try:
        with open(".auth_code") as f:
            auth_code = f.read()
    except FileNotFoundError:
        return "Authorization code not found.", 404

    # rm .auth_code
    os.remove(".auth_code")
    return auth_code, 200


if __name__ == "__main__":
    app.run(port=3333)
