#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to collect detailed information for previously fetched Strava activities.
"""

import os
import sys
import json
import time
import re

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.strava.explore import get_activity_details, ACTIVITY_DATA_PATH, BASE_DIR
except ImportError:
    # Fallback if running directly from src directory
    from strava.explore import get_activity_details, ACTIVITY_DATA_PATH, BASE_DIR

# Define path for detailed activity data
ACTIVITY_DETAILS_PATH = os.path.join(BASE_DIR, "data", "activity_details")
os.makedirs(ACTIVITY_DETAILS_PATH, exist_ok=True)  # Ensure directory exists

# Define a delay between API calls (in seconds) to respect rate limits
API_CALL_DELAY = 2  # Adjust as needed based on Strava's limits


def extract_id_from_filename(filename):
    """Extracts the activity ID from a filename like 'activity_12345.json'."""
    match = re.search(r"activity_(\d+)\.json", filename)
    if match:
        return int(match.group(1))
    return None


def save_activity_details(activity_id, details):
    """Saves the activity details JSON file to the details directory."""
    file_path = os.path.join(ACTIVITY_DETAILS_PATH, f"activity_{activity_id}.json")
    try:
        with open(file_path, "w") as f:
            json.dump(details, f, indent=4)
        print(f"Successfully saved details to {file_path}.")
        return True
    except IOError as e:
        print(f"Error writing file {file_path}: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error reading existing JSON from {file_path}: {e}")
        return False


def main():
    """
    Main function to fetch and store details for activities.
    """
    print(f"Looking for activity files in: {ACTIVITY_DATA_PATH}")
    if not os.path.isdir(ACTIVITY_DATA_PATH):
        print("Error: Activity data directory does not exist.")
        return

    activity_files = [
        f
        for f in os.listdir(ACTIVITY_DATA_PATH)
        if f.startswith("activity_") and f.endswith(".json")
    ]

    if not activity_files:
        print("No activity files found to process.")
        return

    print(f"Found {len(activity_files)} activity files.")
    updated_count = 0
    error_count = 0
    rate_limit_hit = False

    for i, filename in enumerate(activity_files):
        activity_id = extract_id_from_filename(filename)
        if not activity_id:
            print(f"Could not extract ID from {filename}. Skipping.")
            error_count += 1
            continue

        print(
            f"\nProcessing file {i+1}/{len(activity_files)}: {filename} (ID: {activity_id})"
        )

        # Check if file already contains splits_metric to avoid refetching (optional)
        # try:
        #     with open(os.path.join(ACTIVITY_DATA_PATH, filename), 'r') as f:
        #         data = json.load(f)
        #     if 'splits_metric' in data:
        #         print(f"Skipping {filename}, already contains splits_metric.")
        #         continue
        # except (FileNotFoundError, json.JSONDecodeError):
        #     pass # Proceed to fetch if file missing or invalid

        activity_details = get_activity_details(activity_id)

        if activity_details == 429:  # Rate limit hit
            print("Stopping due to rate limit.")
            rate_limit_hit = True
            break
        elif activity_details:
            # Check if splits_metric is present before saving
            if "splits_metric" in activity_details:
                if save_activity_details(activity_id, activity_details):
                    updated_count += 1
                else:
                    error_count += 1
            else:
                print(
                    f"Activity {activity_id} details fetched, but 'splits_metric' not found. Details file not saved."
                )
                # Optionally save anyway if other details are useful
                # save_activity_details(activity_id, activity_details)
                error_count += 1  # Count as error if splits are required
        else:
            print(f"Failed to get details for activity {activity_id}. Skipping update.")
            error_count += 1

        # Pause between requests
        print(f"Waiting {API_CALL_DELAY} seconds before next request...")
        time.sleep(API_CALL_DELAY)

    print("\n--- Summary ---")
    print(f"Total files processed: {i + 1 if not rate_limit_hit else i}")
    print(f"Successfully updated files: {updated_count}")
    print(f"Files with errors or missing splits: {error_count}")
    if rate_limit_hit:
        print("Processing stopped early due to rate limits.")


if __name__ == "__main__":
    main()
