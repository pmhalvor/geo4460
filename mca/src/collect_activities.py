#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to collect athlete's activities from Strava API.
"""

import os
import sys

# Add the project root directory to the Python path
# This allows importing modules from src, strava, etc. regardless of current directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.strava.explore import get_athlete_activities, store_activities
except ImportError:
    # Fallback if running directly from src directory
    from strava.explore import get_athlete_activities, store_activities


def main():
    """
    Main function to fetch and store Strava activities.
    """
    print("Starting activity collection...")
    fetched_activities = get_athlete_activities()

    if fetched_activities:
        # Store only activities of type "Ride"
        store_activities(fetched_activities, activity_type_filter="Ride")
        print(f"Finished collecting and storing ride activities.")
    else:
        print("No activities were fetched.")


if __name__ == "__main__":
    main()
