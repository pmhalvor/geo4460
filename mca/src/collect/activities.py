#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to collect athlete's activities from Strava API.
"""
from src.strava.explore import get_athlete_activities, store_activities


def main():
    """
    Main function to fetch and store Strava activities.
    """
    print("Starting activity collection...")
    fetched_activities = get_athlete_activities()

    if fetched_activities:
        # Store only activities of type "Ride"
        store_activities(fetched_activities, activity_type_filter="Ride")
        print("Finished collecting and storing ride activities.")
    else:
        print("No activities were fetched.")


if __name__ == "__main__":
    main()
