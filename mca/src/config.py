from pathlib import Path
from pydantic import BaseModel, Field, DirectoryPath, FilePath
from typing import Optional, List
from datetime import datetime

# Determine the base directory relative to this config file
# config.py is in src/, so BASE_DIR is the parent of src/ which is mca/
BASE_DIR = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    """Configuration for base directories and main input/output paths."""

    base_dir: Path = BASE_DIR
    data_dir: DirectoryPath = Field(default_factory=lambda: BASE_DIR / "data")
    output_dir: Path = Field(
        default_factory=lambda: BASE_DIR
        / "output"
        / f"mca_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    # Specific input data paths (relative to data_dir or base_dir)
    strava_segments_geojson: FilePath = Field(
        # Use BASE_DIR directly in lambda to avoid potential recursion
        default_factory=lambda: BASE_DIR
        / "data"
        / "segments"
        / "segments_oslo.geojson"
    )
    strava_activities_dir: Optional[DirectoryPath] = Field(
        default=None, description="Directory containing Strava activity GPX/TCX files"
    )  # TODO change based on gathered data
    traffic_bikes_csv: FilePath = Field(
        # Use BASE_DIR directly in lambda
        default_factory=lambda: BASE_DIR
        / "data"
        / "traffic"
        / "all-oslo-bikes-day_20240101T0000_20250101T0000.csv"
    )
    traffic_stations_dir: DirectoryPath = Field(
        # Use BASE_DIR directly in lambda
        default_factory=lambda: BASE_DIR / "data" / "traffic",
        description="Directory containing traffic station JSON files (e_road, f_road etc.)",
    )
    n50_gdb_path: Optional[FilePath] = Field(
        default=None, description="Path to the N50 Geodatabase (if used)"
    )  # TODO should not be optional, but for now it is
    # Cache file for segment details fetched from API
    segment_details_cache_csv: Path = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "segments"
        / "segment_details_cache.csv"
    )

    # Ensure directories exist or create them if needed
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure output directory exists after initialization
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure cache directory exists
        self.segment_details_cache_csv.parent.mkdir(parents=True, exist_ok=True)

        # Validate essential input directories/files exist
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Data directory not found: {self.data_dir}")
        if not self.strava_segments_geojson.is_file():
            raise FileNotFoundError(
                f"Strava segments file not found: {self.strava_segments_geojson}"
            )
        if not self.traffic_bikes_csv.is_file():
            raise FileNotFoundError(
                f"Traffic bikes CSV file not found: {self.traffic_bikes_csv}"
            )
        if not self.traffic_stations_dir.is_dir():
            raise NotADirectoryError(
                f"Traffic stations directory not found: {self.traffic_stations_dir}"
            )
        # Optional validations
        if self.strava_activities_dir and not self.strava_activities_dir.is_dir():
            raise NotADirectoryError(
                f"Strava activities directory not found: {self.strava_activities_dir}"
            )
        if self.n50_gdb_path and not self.n50_gdb_path.is_dir():  # GDB is a directory
            raise NotADirectoryError(
                f"N50 Geodatabase directory not found: {self.n50_gdb_path}"
            )


class InputDataConfig(BaseModel):
    """Configuration for input data specifics, like layer names or field names."""

    # N50 Layer Names
    n50_roads_layer: str = "veg_veglenke"  # TODO update w/ correct field name
    n50_bike_lanes_layer: str = "veg_sykkelveg"  # TODO update w/ correct field name
    n50_contour_layer: str = "terreng_N50_kontur"  # TODO update w/ correct field name

    # Strava Segments Fields
    segment_id_field: str = "id"
    segment_polyline_field: str = "polyline"
    segment_athlete_count_field: str = "athlete_count"
    segment_effort_count_field: str = "effort_count"
    segment_star_count_field: str = "star_count"
    segment_created_at_field: str = "created_at"
    segment_distance_field: str = "distance"
    segment_elevation_diff_field: str = "total_elevation_gain"  # Or similar

    # Strava Activities Fields (if processing GPX/TCX)
    activity_speed_field: str = "speed"  # Field name within parsed activity data
    activity_time_field: str = "time"
    activity_elevation_field: str = "elevation"

    # Traffic Data Fields
    traffic_station_id_field: str = "id"
    traffic_volume_field: str = "total_volume"  # TODO update w/ correct field name
    traffic_bike_volume_field: str = (
        "volume_bicycle"  # TODO update w/ correct field name
    )
    traffic_timestamp_field: str = "timestamp"  # TODO update w/ correct field name


class OutputFilesConfig(BaseModel):
    """Relative filenames for output files within the output directory."""

    # Prepared Data (GeoDataFrames saved, e.g., as GeoPackage)
    prepared_segments_gpkg: str = "prepared_segments.gpkg"
    prepared_activities_gpkg: str = "prepared_activities.gpkg"
    prepared_traffic_points_gpkg: str = "prepared_traffic_points.gpkg"
    prepared_roads_gpkg: str = "prepared_roads.gpkg"  # From N50
    prepared_bike_lanes_gpkg: str = "prepared_bike_lanes.gpkg"  # From N50
    prepared_roads_no_lanes_gpkg: str = "prepared_roads_no_lanes.gpkg"
    prepared_contours_gpkg: str = "prepared_contours.gpkg"  # From N50

    # Feature Rasters
    segment_popularity_raster_prefix: str = (
        "segment_popularity"  # e.g., segment_popularity_athletes_per_age.tif
    )
    average_speed_raster: str = "average_speed.tif"
    traffic_density_raster: str = "traffic_density.tif"
    elevation_dem_raster: str = "elevation_dem.tif"
    slope_raster: str = "slope.tif"
    cost_function_raster: str = "cost_function.tif"

    # Intermediate Files (optional, for debugging)
    buffered_traffic_stations_gpkg: str = "buffered_traffic_stations.gpkg"
    reclassified_roads_gpkg: str = "reclassified_roads.gpkg"
    reclassified_bike_lanes_gpkg: str = "reclassified_bike_lanes.gpkg"

    # Combined Features / Overlays (Vector or Raster)
    overlay_a_gpkg: str = "overlay_A_popular_no_lanes.gpkg"
    overlay_b_gpkg: str = "overlay_B_popular_no_lanes_high_speed.gpkg"
    overlay_c_gpkg: str = "overlay_C_popular_no_lanes_high_speed_traffic.gpkg"
    overlay_d_gpkg: str = "overlay_D_final_recommendations.gpkg"  # Final vector output

    # Evaluation Outputs
    evaluation_stats_csv: str = "evaluation_segment_comparison.csv"
    recommended_segments_gpkg: str = "recommended_segments.gpkg"
    evaluation_plots_dir: str = "evaluation_plots"  # Directory for plots

    # Method to get full path
    def get_full_path(self, filename_attr: str, output_dir: Path) -> Path:
        """Returns the full path for a given output filename attribute."""
        filename = getattr(self, filename_attr)
        return output_dir / filename


class ProcessingConfig(BaseModel):
    """Parameters controlling the processing steps."""

    # General Raster Settings
    output_crs_epsg: int = 25833  # UTM Zone 33N for Oslo
    output_cell_size: float = 10.0  # Meters, adjust as needed
    interpolation_method_points: str = "idw"  # e.g., idw, kriging, natural_neighbor
    interpolation_method_polylines: str = "linear"  # Method for rasterizing lines

    # WhiteboxTools Settings
    wbt_verbose: bool = False

    # Feature Generation Settings
    segment_popularity_metrics: List[str] = Field(
        default=[
            "efforts_per_age",
            "athletes_per_age",
            "stars_per_age",
            "stars_per_athlete",
        ],
        description="Metrics to calculate for segment popularity.",
    )
    segment_popularity_idw_min_points: int = 5
    segment_popularity_idw_power: float = 3.0
    segment_popularity_idw_radius: float = 500.0  # Meters
    segment_age_calculation_method: str = (
        "days"  # 'days', 'years' - determines denominator for _per_age metrics
    )
    strava_api_request_delay: float = (
        0.25  # Seconds delay between API calls to avoid rate limits
    )

    traffic_buffer_distance: float = 500.0  # Meters, for buffering traffic stations
    traffic_interpolation_power: float = 2.0  # For IDW interpolation
    road_buffer_distance: Optional[float] = Field(
        default=5.0, description="Optional buffer for road/lane matching (meters)"
    )
    slope_units: str = "degrees"  # 'degrees' or 'percent'

    # Kriging specific parameters (if used for popularity)
    kriging_model: str = "spherical"  # e.g., spherical, exponential, gaussian
    kriging_range: Optional[float] = None  # Search radius (if None, WBT might estimate)
    kriging_sill: Optional[float] = None
    kriging_nugget: Optional[float] = None

    # Cost Function Settings
    cost_slope_weight: float = 1.0
    cost_speed_weight: float = -0.5  # Negative weight = reward
    cost_road_restriction_value: Optional[float] = (
        None  # Value for non-road areas if restricting
    )

    # Overlay Settings
    overlay_speed_threshold: Optional[float] = None  # Avg speed threshold for Overlay B
    overlay_traffic_threshold: Optional[float] = (
        None  # Traffic density threshold for Overlay C
    )
    overlay_cost_threshold: Optional[float] = None  # Cost threshold for Overlay D

    # Evaluation Settings
    top_n_recommendations: int = 20  # Number of top segments to highlight


class AppConfig(BaseModel):
    """Main application configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    input_data: InputDataConfig = Field(default_factory=InputDataConfig)
    output_files: OutputFilesConfig = Field(default_factory=OutputFilesConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)


# Instantiate the main config object for easy import
settings = AppConfig()

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("--- MCA Configuration Loaded ---")
    print(f"Base Directory: {settings.paths.base_dir}")
    print(f"Data Directory: {settings.paths.data_dir}")
    print(f"Output Directory: {settings.paths.output_dir}")
    print(f"Strava Segments: {settings.paths.strava_segments_geojson}")
    print(
        f"Segment Cache: {settings.paths.segment_details_cache_csv}"
    )  # Updated cache path print
    print(f"Output CRS EPSG: {settings.processing.output_crs_epsg}")
    print(f"Output Cell Size: {settings.processing.output_cell_size}")
    print(
        f"Segment Popularity Metrics: {settings.processing.segment_popularity_metrics}"
    )
    # Example of getting a full output path
    pop_raster_path = settings.output_files.get_full_path(
        "segment_popularity_raster_prefix", settings.paths.output_dir
    )
    print(f"Example Popularity Raster Prefix Path: {pop_raster_path}_METRIC.tif")
    print(f"Output directory exists: {settings.paths.output_dir.exists()}")
    print("--- Configuration Test End ---")
