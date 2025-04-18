"""
1. Load data: poly lines and points 
    1. Segments: gdf, get_metric(metric, id=“all”), get(id), len, …
    2. Heatmap: gdf, get_activity(id), len, …
    3. Traffic: gdf, get_metric(metric, id="all”, vehicle=["all”, “bike”, “car”]), get(id), len, … 
    4. Lanes: gdf, get_classification(id), get(id), len, …
    5. Elevation: contour lines, get_metric(metric, id="all”), get(id), len, …
    6. Roads: gdf, get_classification()

2. Generate rasters:
    1. Traffic buffers (for better segment intersection)
        1. Bike
        2. Car
    2. Lanes raster/lines
    3. Average speed raster/lines (from personal heat map)
    4. Segment popularity rasters/lines
        1. Athletes/age
        2. Stars/age
        3. Stars/athletes
        4. …
    5. Generate roads polygon (cost distance)
    6. Create Oslo mask (from random CDW)
    7. Elevation DEM
    8. Slope Raster 
    9. PASR: Predicted average speed raster (Co-Kriging or similar using slope as external variable)

"""

import dask
import time

from abc import ABC
from abc import abstractmethod
from pathlib import Path

from dask.distributed import Client


class FeatureBase(ABC):
    def __init__(self, data_path):
        self.data_path = data_path
        self.gdf = None

    @abstractmethod
    def interpolate(self, points):
        pass

    @dask.delayed
    def reclassify(self, layer, category, feature_name):
        """
        Returns raster of reclassified data.
        """
        time.sleep(2)
        print(f"Reclassifying {layer} with {category} data in {feature_name}")
        return f"Reclassified {layer} with {category}"

    @dask.delayed
    def buffer(self, layer, feature_name):
        time.sleep(2)
        print(f"Buffering {layer} in {feature_name}")
        return f"Buffered {layer}"

    def load_data(self):
        # Simulate loading data wbt
        time.sleep(1)
        print(f"Loading data from {self.data_path}")
        self.gdf = f"Loaded dataframe from {self.data_path}"
        self.base_layer = f"Converted data to base layer from {self.data_path}"
        return f"Loaded data from {self.data_path}"


class Segments(FeatureBase):
    def __init__(self, data_path: Path, metric="athletes/age"):
        super().__init__(data_path)
        self.load_data()
        self.metric = metric
        self.popularity_raster_task = self.build_popularity_raster(metric)

    def set_metric(self, metric):
        if metric in self.allowed_metrics:  # TODO define allowed metrics in config
            self.metric = metric
        else:
            raise ValueError(
                f"Invalid metric: {metric}. Allowed metrics are: {self.allowed_metrics}"
            )

        print(f"Metric set to {self.metric}")

    def preprocess(self):
        print(f"Preprocessing segments data from {self.data_path}")
        if self.gdf is None:
            raise ValueError("Data not loaded. Please load data before preprocessing.")
        time.sleep(2)
        print(f"Preprocessed segments data from {self.data_path} in Segments")
        return f"Preprocessed segments data from {self.data_path}"

    def segments_to_popularity_polyline_layer(self, metric):
        print(
            f"Converting segments data to a popularity polyline from {self.data_path}"
        )
        if self.gdf is None:
            raise ValueError(
                "Data not loaded. Please load data before converting to polyline."
            )
        time.sleep(2)
        print(f"Converted segments data to a popularity polyline from {self.data_path}")
        return f"Converted segments data to a popularity polyline from {self.data_path}"

    def interpolate(self, points):
        time.sleep(5)
        print(f"Interpolating segments data with {points}")
        return f"Interpolated segments data with {points}"

    @dask.delayed
    def build_popularity_raster(self, metric):
        self.popularity_polyline = self.segments_to_popularity_polyline_layer(metric)
        self.popularity_points = self.polyline_to_points(self.popularity_polyline)
        return self.interpolate(self.popularity_points)


class Heatmap(FeatureBase):
    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.load_data()
        self.average_speed_raster_task = self.build_average_speed_raster("ride")

    def activities_to_speed_polyline_layer(self, activity_type):
        print(f"Converting activities data to a speed polyline from {self.data_path}")
        if self.gdf is None:
            raise ValueError(
                "Data not loaded. Please load data before converting to polyline."
            )
        time.sleep(2)

        # Ensure activity type
        gdf = self.gdf[self.gdf["activity_type"] == activity_type]
        if gdf.empty:
            raise ValueError(f"No data found for activity type: {activity_type}")

        polylines = gdf["polyline"].tolist()

        # some join logic on speed over polyline

        print(f"Converted activities data to a speed polyline: {polylines}")
        return f"Converted activities data to a speed polyline: {polylines}"

    def interpolate(self, points):
        print(f"Interpolating heatmap data {points}")
        # Use Co-Kriging if elevation data is available, otherwise just Kriging
        time.sleep(5)
        return f"Interpolated heatmap data {points}"

    @dask.delayed
    def build_average_speed_raster(self, activity_type):
        self.speed_polyline = self.activities_to_speed_polyline_layer(activity_type)
        self.speed_points = self.polyline_to_points(self.speed_polyline)
        return self.interpolate(self.speed_points)


class Roads(FeatureBase):
    def __init__(self, data_path: Path):
        """
        Initialize the Roads class with a path to a CSV file containing roads data.
        Args:
            data_path (Path): Path to the CSV file containing roads data.
        """
        super().__init__(data_path)
        self.load_data()
        self.roads_layer_task = self.build_layer("roads")
        self.bike_lanes_layer_task = self.build_layer("bike_lanes")

    def interpolate(self, metric):
        # Necessary due to abstract parent
        raise NotImplementedError("No interpolation defined for Roads")

    @dask.delayed
    def build_layer(self, layer_name):
        if self.base_layer is None:
            raise ValueError(
                "Data not loaded. Please load data before getting roads layer."
            )

        print(f"Reclassifying and buffering {layer_name} data in Roads")
        # can decide later whether to persist or drop intermediate layers
        self.reclassified_roads_layer = self.reclassify(
            self.base_layer, "classification", layer_name
        )
        self.buffered_roads = self.buffer(self.reclassified_roads_layer, layer_name)

        time.sleep(2)
        return self.buffered_roads


class Traffic(FeatureBase):
    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.load_data()
        self.road_traffic_task = self.build_layer("roads")
        self.lanes_traffic_task = self.build_layer("lanes")

    @dask.delayed
    def build_layer(self, layer_name):
        if self.base_layer is None:
            raise ValueError(
                "Data not loaded. Please load data before getting roads layer."
            )

        print(f"Reclassifying and buffering {layer_name} data in Roads")
        reclassified_layer = self.reclassify(
            self.base_layer, "classification", layer_name
        )
        buffered_layer = self.buffer(reclassified_layer, layer_name)

        # is buffer necessary or go straight to interpolation?

        points = self.polyline_to_points(buffered_layer)

        time.sleep(2)
        return self.interpolate(points)

    def interpolate(self, points):
        # IDW or some form of interpolation to generate expected traffic raster
        print(f"Interpolating traffic data {points}")
        time.sleep(5)
        return f"Interpolated traffic data {points}"


class Elevation(FeatureBase):
    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.load_data(data_path)
        self.dem_task = self.build_dem()

    def interpolate(self, points):
        print(f"Interpolating elevation data {points}")
        time.sleep(5)
        return f"Interpolated elevation data {points}"

    @dask.delayed
    def build_dem(self):
        # get contour data
        # convert contours to points
        # interpolate points to DEM (TIN should be fine)
        # calculate slope
        time.sleep(5)
        self.dem = "DEM raster"
        self.slope = "Slope raster"
        print(f"Built DEM and slope raster from {self.data_path}")
        return "Completed DEM and slope raster"


def build_features_task(data_path: Path):
    """
    Build features task to load and process data.

    Args:
        data_path (Path): Path to the CSV file containing data.

    Returns:
        str: Message indicating the completion of the task.
    """
    print("--- Start Feature Building Task ---")

    # Initialize Dask client
    client = Client()

    # Load data
    segments = Segments(data_path)
    heatmap = Heatmap(data_path)
    traffic = Traffic(data_path)
    roads = Roads(data_path)
    elevation = Elevation(data_path)

    tasks = [
        segments.popularity_raster_task,
        heatmap.average_speed_raster_task,
        traffic.road_traffic_task,
        traffic.lanes_traffic_task,
        roads.roads_layer_task,
        roads.bike_lanes_layer_task,
        elevation.dem_task,
    ]

    results = dask.compute(*tasks)
    print("All tasks completed.")

    client.close()

    for result in results:
        print(result)

    print("--- Feature Building Task Completed ---")
    return "Feature building task completed."
