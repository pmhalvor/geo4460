import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import Point

ALL_STATIONS_PATH = "mca/data/traffic/all_stations.json"
DAY_TRAFFIC_PATH = "mca/data/traffic/all-oslo-bikes-929776236_day_20240101T0000_20250101T0000.csv"
HOUR_TRAFFIC_PATH = "mca/data/traffic/all-oslo-bikes-929776236_hour_20240401T0000_20240501T0000.csv"
OSLO_STATION_SHAPEFILE_PATH = "mca/data/traffic/oslo_stations.shp"

def get_stations_gdf():
    with open(ALL_STATIONS_PATH) as f:
        all_stations = json.load(f)

    assert list(all_stations.keys()) == ["data"]
    assert list(all_stations["data"].keys()) == ["trafficRegistrationPoints"]
    assert (
        list(all_stations["data"]["trafficRegistrationPoints"][0].keys()) 
        == 
        ['id', 'name', 'location']
    )

    # Create a DataFrame
    df = pd.DataFrame(all_stations["data"]["trafficRegistrationPoints"])

    # Convert location to geometry
    def loc_to_point(loc):
        latLon = loc['coordinates']['latLon']
        return Point(latLon['lon'], latLon['lat'])
    
    df['geometry'] = df['location'].apply(loc_to_point)

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    return gdf


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
    # Convert to GeoDataFrame
    stations = get_stations_gdf()
    print(stations.head())

    # Load traffic data
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
