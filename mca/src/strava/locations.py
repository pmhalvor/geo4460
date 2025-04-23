"""
This file contains typical cycling locations around Oslo,
formatted as 'bounds' for the Strava API.

Long/lat coordinates easily found via https://geojson.io/

"""

# all locations
locations = {
    "filipstad": {
        "name": "Filipstad",
        "bounds": [
            59.9050,
            10.7086,
            59.9111,
            10.7225,
        ],
    },
    "grunerlokka": {
        "name": "Grünerløkka",
        "bounds": [
            59.917,
            10.745,
            59.93,
            10.77,
        ],
    },
    "sthanshaugen": {
        "name": "St. Hanshaugen",
        "bounds": [
            59.922,
            10.733,
            59.932,
            10.748,
        ],
    },
    "fagerborg": {
        "name": "Fagerborg",
        "bounds": [
            59.922,
            10.722,
            59.933,
            10.735,
        ],
    },
    "bygdøy": {
        "name": "Bygdøy",
        "bounds": [
            59.8944,
            10.6663,
            59.9197,
            10.6927,
        ],
    },
    "holmenkollen": {
        "name": "Holmenkollen",
        "bounds": [
            59.958,
            10.636,
            59.983,
            10.666,
        ],
    },
    "ekeberg": {
        "name": "Ekeberg",
        "bounds": [
            59.8819,
            10.7545,
            59.9008,
            10.7899,
        ],
    },
    "groruddalen": {
        "name": "Groruddalen",
        "bounds": [
            59.927,
            10.800,
            59.963,
            10.9,
        ],
    },
    "e18": {
        "name": "E18",
        "bounds": [
            59.840,
            10.761,
            59.899,
            10.779,
        ],
    },
    "barcode": {
        "name": "Barcode",
        "bounds": [
            59.9069,
            10.7485,
            59.9072,
            10.7634,
        ],
    },
    "ring2": {
        "name": "Ring 2",
        "bounds": [
            59.9056,
            10.6992,
            59.9376,
            10.7825,
        ],
    },
    "griffenfeldtsgate": {
        "name": "Griffenfeldtsgate",
        "bounds": [
            59.9253556102245,
            10.733905295904663,
            59.93671012314178,
            10.776545253217193,
        ],
    },
    "majorstuen": {
        "name": "Majorstuen",
        "bounds": [
            59.937,
            10.690,
            59.919,
            10.734,
        ],
    },
    "frognerparken": {
        "name": "Frognerparken",
        "bounds": [
            59.92266606115345,
            10.70386777617938,
            59.929499268779125,
            10.716781298222372,
        ],
    },
    "ballerud": {
        "name": "Ballerud",
        "bounds": [
            59.89046990187995,
            10.554981936406875,
            59.92010744816895,
            10.594535728332943,
        ],
    },
    "oslo": {  # whole oslo city
        "name": "Oslo",
        "bounds": [
            59.8181886681663,
            10.42043828050879,
            60.0142603407657,
            11.007603658932084,
        ],
    },
}
