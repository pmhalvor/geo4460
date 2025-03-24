import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import Point

ALL_STATIONS_PATH = "mca/data/traffic/all_stations.json"

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


if __name__ == "__main__":
    # Convert to GeoDataFrame
    stations = get_stations_gdf()
    print(stations.head())
