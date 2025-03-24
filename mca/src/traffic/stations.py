import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import Point

DAY_TRAFFIC_PATH = "mca/data/traffic/all-oslo-bikes-929776236_day_20240101T0000_20250101T0000.csv"
HOUR_TRAFFIC_PATH = "mca/data/traffic/all-oslo-bikes-929776236_hour_20240401T0000_20240501T0000.csv"
OSLO_STATION_SHAPEFILE_PATH = "mca/data/traffic/oslo_stations.shp"
ROAD_CATEGORY_IDS = [
    "e", # European
    "r", # National
    "f", # County
    "k", # Municipal
    "p", # Private
]
STATIONS_PATH_TEMPLATE = "mca/data/traffic/stations_{id}_road.json"


def get_stations_gdf():
    df = None

    for road_id in ROAD_CATEGORY_IDS:
        with open(STATIONS_PATH_TEMPLATE.format(id=road_id)) as f:
            stations = json.load(f)

            assert list(stations.keys()) == ["data"]
            assert list(stations["data"].keys()) == ["trafficRegistrationPoints"]
            assert (
                list(stations["data"]["trafficRegistrationPoints"][0].keys()) 
                == 
                ['id', 'name', 'location']
            )

        if df is None:
            df = pd.DataFrame(stations["data"]["trafficRegistrationPoints"])
        else:
            df = pd.concat(
                [
                    df, 
                    pd.DataFrame(stations["data"]["trafficRegistrationPoints"])
                ]
            )

    def loc_to_point(loc):
        latLon = loc['coordinates']['latLon']
        return Point(latLon['lon'], latLon['lat'])
    
    df['points'] = df['location'].apply(loc_to_point)

    return gpd.GeoDataFrame(df, geometry='points', crs="EPSG:4326")


def get_traffic_df(interval:str = "day"):
    if interval == "day":
        # df = pd.read_csv(DAY_TRAFFIC_PATH)
        datapath = DAY_TRAFFIC_PATH
    elif interval == "hour":
        # df = pd.read_csv(HOUR_TRAFFIC_PATH)
        datapath = HOUR_TRAFFIC_PATH
    else:
        raise ValueError(f"Invalid interval: {interval}")
    
    df = pd.read_csv(datapath, encoding="ISO-8859-1", header=0, sep=";", dtype='unicode')

    return df


if __name__ == "__main__":
    stations = get_stations_gdf()
    print(stations.head())

    day_traffic_df = get_traffic_df(interval="day")
    print(day_traffic_df.head())
    hour_traffic_df = get_traffic_df(interval="hour")
    print(hour_traffic_df.head())

    # station 48146B618224 missing from hourly data (taken only for April 2024)
    oslo_station_ids = day_traffic_df.Trafikkregistreringspunkt.unique()
    print(f"Number Oslo station IDs: {len(oslo_station_ids)}")

    oslo_stations = stations[stations.id.isin(oslo_station_ids)]
    print(f"Number Oslo stations: {len(oslo_stations)}")
    print(oslo_stations.head())

    oslo_stations_4258 = oslo_stations.to_crs(epsg=4258)
    oslo_stations_4258.to_file(OSLO_STATION_SHAPEFILE_PATH, driver="ESRI Shapefile")
    print(f"Oslo stations saved to {OSLO_STATION_SHAPEFILE_PATH}")
