"""
1. Load data: poly lines and points 
    1. Segments: gdf, get_metric(metric, id=“all”), get(id), len, …
    2. Heatmap: gdf, get_activity(id), len, …
    3. Traffic: gdf, get_metric(metric, id="all”, vehicle=["all”, “bike”, “car”]), get(id), len, … 
    4. Lanes: gdf, get_classification(id), get(id), len, …
    5. AQI: gdf, get_metric(metric, id="all”), get(id), len, …
    6. Elevation: contour lines, get_metric(metric, id="all”), get(id), len, …
    7. Roads: gdf, get_classification()
"""

from pathlib import Path


class Segments:
    def __init__(self, segments_df_path: Path):
        """
        Initialize the Segments class with a path to a CSV file containing segments data.

        Args:
            segments_df_path (Path): Path to the CSV file containing segments data.
        """
        self.segments_df_path = segments_df_path


class Heatmap:
    def __init__(self, heatmap_df_path: Path):
        """
        Initialize the Heatmap class with a path to a CSV file containing heatmap data.

        Args:
            heatmap_df_path (Path): Path to the CSV file containing heatmap data.
        """
        self.heatmap_df_path = heatmap_df_path


class Traffic:
    def __init__(self, traffic_df_path: Path):
        """
        Initialize the Traffic class with a path to a CSV file containing traffic data.

        Args:
            traffic_df_path (Path): Path to the CSV file containing traffic data.
        """
        self.traffic_df_path = traffic_df_path


class Lanes:
    def __init__(self, lanes_df_path: Path):
        """
        Initialize the Lanes class with a path to a CSV file containing lanes data.
        Args:
            lanes_df_path (Path): Path to the CSV file containing lanes data.
        """
        self.lanes_df_path = lanes_df_path


class Roads:
    def __init__(self, roads_df_path: Path):
        """
        Initialize the Roads class with a path to a CSV file containing roads data.
        Args:
            roads_df_path (Path): Path to the CSV file containing roads data.
        """
        self.roads_df_path = roads_df_path


class AQI:
    def __init__(self, aqi_df_path: Path):
        """
        Initialize the AQI class with a path to a CSV file containing AQI data.
        Args:
            aqi_df_path (Path): Path to the CSV file containing AQI data.
        """
        self.aqi_df_path = aqi_df_path
