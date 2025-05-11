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
    segment_details_cache_csv: Path = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "segments"
        / "segment_details_cache.csv"
    )
    
    traffic_bikes_csv: FilePath = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "traffic"
        / "all-oslo-bikes-day_20240101T0000_20250101T0000.csv"
    )
    traffic_stations_dir: DirectoryPath = Field(
        default_factory=lambda: BASE_DIR / "data" / "traffic",
        description="Directory containing traffic station JSON files (e_road, f_road etc.)",
    )
    
    n50_gdb_path: DirectoryPath = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "Basisdata_03_Oslo_25833_N50Kartdata_FGDB.gdb",
        description="Path to the N50 Geodatabase",
    )
    oslo_sykkelfelt_kml_path: FilePath = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "bike_lanes"
        / "Sykkelruter i Oslo.kml"
    )
    
    # Cache file for segment details fetched from API
    activity_details_dir: Path = Field(
        default_factory=lambda: BASE_DIR / "data" / "activity_details",
        description="Directory activity details stored at after collection",
    )

    # Input for the non-bike-laned roads to segment collection script
    diff_layer_gpkg: Path = Field(
        # Example path, user should verify or update this default if needed
        default_factory=lambda: BASE_DIR
        / "output"
        / "mca_20250423_1257_roads" # TODO: Make this dynamic or user-configurable?
        / "prepared_roads_all_diff_lanes.gpkg", 
        # / "prepared_roads_simple_diff_lanes.gpkg",  # Alternative
        description="Path to the difference layer (e.g., roads minus bike lanes) used for sampling points.",
    )
    # Intermediate/Output files for the new segment collection script
    remaining_search_points_gpkg: Path = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "segments"
        / "road_diff_remaining_search_points.gpkg",
        description="GeoPackage storing points sampled from the diff layer that still need to be searched.",
    )
    collected_segments_gpkg: Path = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "segments"
        / "collected_segments_from_diff.gpkg",
        description="GeoPackage storing segments collected via API calls based on the diff layer points.",
    )
    # Files for collecting segments based on simple roads lacking segments
    remaining_simple_road_points_gpkg: Path = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "segments"
        / "simple_road_remaining_search_points.gpkg",
        description="GeoPackage storing points sampled from simple roads (lacking segments) that still need searching.",
    )
    collected_segments_from_simple_roads_gpkg: Path = Field(
        default_factory=lambda: BASE_DIR
        / "data"
        / "segments"
        / "collected_segments_from_simple_roads.gpkg",
        description="GeoPackage storing segments collected via API calls based on simple road points.",
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
        # if not self.strava_segments_geojson.is_file():
        #     raise FileNotFoundError(
        #         f"Strava segments file not found: {self.strava_segments_geojson}"
        #     )
        if not self.traffic_bikes_csv.is_file():
            raise FileNotFoundError(
                f"Traffic bikes CSV file not found: {self.traffic_bikes_csv}"
            )
        if not self.traffic_stations_dir.is_dir():
            raise NotADirectoryError(
                f"Traffic stations directory not found: {self.traffic_stations_dir}"
            )
        if self.n50_gdb_path and not self.n50_gdb_path.is_dir():
            raise NotADirectoryError(
                f"N50 Geodatabase directory not found: {self.n50_gdb_path}"
            )


class InputDataConfig(BaseModel):
    """Configuration for input data specifics, like layer names or field names."""

    # N50
    n50_land_cover_layer: str = "N50_Arealdekke_omrade"  # omrade or grense works
    n50_samferdsel_layer: str = "N50_Samferdsel_senterlinje"
    n50_samferdsel_typeveg_field: str = "typeveg" # Field name for road type
    n50_typeveg_bike_lane: str = "gangOgSykkelveg" # Value for bike lanes
    n50_typeveg_road_simple: str = "enkelBilveg" # Value for simple car roads
    n50_contour_layer: str = "N50_Høyde_senterlinje"
    n50_contour_elevation_field: str = "hoyde"

    # Strava Segments Fields
    segment_id_field: str = "id"
    segment_polyline_field: str = "polyline"
    segment_athlete_count_field: str = "athlete_count"
    segment_effort_count_field: str = "effort_count"
    segment_star_count_field: str = "star_count"
    segment_created_at_field: str = "created_at"
    segment_distance_field: str = "distance"
    segment_elevation_diff_field: str = "total_elevation_gain"  # Or similar

    # Strava Activities Fields
    activity_speed_field: str = "speed"
    activity_time_field: str = "time"
    activity_elevation_field: str = "elevation"

    # Traffic Data Fields
    traffic_station_id_field: str = "id"
    traffic_volume_field: str = "total_volume"  # TODO update w/ correct field name
    traffic_bike_volume_field: str = (
        "volume_bicycle"  # TODO update w/ correct field name
    )
    traffic_timestamp_field: str = "timestamp"  # TODO update w/ correct field name

    # Roads/Bike Lane Fields
    oslo_bike_path_layers: List[str] = [
        "Gang- og sykkelveier - vis hensyn (separate shared-use paths)",
        "Sykkelfelt (Bicycle lane)",
        "Sykkelvei med fortau (Separate bicycle path)",
    ]


class OutputFilesConfig(BaseModel):
    """Relative filenames for output files within the output directory."""

    # Prepared Data (GeoDataFrames saved, e.g., as GeoPackage)
    prepared_segments_gpkg: str = "prepared_segments.gpkg"
    prepared_activities_gpkg: str = "prepared_activities.gpkg"  # not used TODO remove
    prepared_activity_splits_gpkg: str = "prepared_activity_splits.gpkg"
    prepared_traffic_points_gpkg: str = "prepared_traffic_points.gpkg"
    prepared_roads_gpkg: str = "prepared_roads_all_samferdsel.gpkg"  # From N50 Samferdsel
    prepared_bike_lanes_gpkg: str = "prepared_bike_lanes_filtered.gpkg" # Filtered from Samferdsel
    prepared_roads_simple_filtered_gpkg: str = "prepared_roads_simple_filtered.gpkg" # Filtered from Samferdsel
    prepared_roads_simple_diff_lanes_gpkg: str = "prepared_roads_simple_diff_lanes.gpkg" # Difference: simple roads - bike lanes
    prepared_roads_all_diff_lanes_gpkg: str = "prepared_roads_all_diff_lanes.gpkg" # Difference: all roads - bike lanes
    prepared_contours_gpkg: str = "prepared_contours.gpkg"  # From N50

    # Feature Rasters
    segment_popularity_raster_prefix: str = (
        "segment_popularity"  # e.g., segment_popularity_athletes_per_age.tif
    )
    average_speed_raster: str = "average_speed.tif"
    # traffic_density_raster: str = "traffic_density.tif" # Old single file
    traffic_density_raster_morning: str = "traffic_density_morning.tif" # 00:00-08:00
    traffic_density_raster_daytime: str = "traffic_density_daytime.tif" # 08:00-16:00
    traffic_density_raster_evening: str = "traffic_density_evening.tif" # 16:00-24:00
    elevation_dem_raster: str = "elevation_dem.tif"
    slope_raster: str = "slope.tif"
    cost_function_raster: str = "cost_function.tif"
    normalized_cost_layer: str = "normalized_cost.tif"  # Normalized cost distance
    rasterized_roads_mask: str = "rasterized_roads_mask.tif"  # Mask for roads
    aligned_speed_raster: str = "aligned_speed.tif"  # Aligned speed raster for overlay
    prepared_kml_bike_lanes_gpkg: str = "prepared_kml_bike_lanes.gpkg"


    # visualizations
    heatmap_visualization_html: str = "heatmap_visualization.html"  
    activity_segments_viz: str = "activity_segments_visualization.html"  
    heatmap_train_points_viz: str = "heatmap_train_points_visualization.html"
    heatmap_test_points_viz: str = "heatmap_test_points_visualization.html"
    cost_layer_visualization_html: str = "cost_layer_visualization.html"

    # Feature Vectors (alternative to rasters)
    segment_popularity_vector_prefix: str = (
        "segment_popularity_vector"  # e.g., segment_popularity_vector_athletes_per_age.gpkg
    )
    average_speed_vector: str = "average_speed_vector.gpkg"

    # Intermediate Files (optional, for debugging)
    buffered_traffic_stations_gpkg: str = "buffered_traffic_stations.gpkg"

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

    # Compute Settings
    wbt_verbose: bool = False
    dask_workers: int = 4  # Number of Dask workers for parallel processing

    # General Raster Settings
    interpolation_crs_epsg: int = 25833  # UTM Zone 33N for Oslo
    map_crs_epsg: int = 4326  # WGS 84 for map display
    output_crs_epsg: int = 25833  # UTM Zone 33N for wbt  # TODO remove from collect/*.py
    output_cell_size: float = 10.0  # Meters, adjust as needed
    seed: int = 42  # Random seed for reproducibility
    train_test_split_fraction: float = 0.8  # Fraction of data for training
    
    # Interpolation Settings
    interpolation_method_points: str = "idw"  # e.g., idw, kriging, natural_neighbor
    interpolation_method_polylines: str = (
        "tin"
        # Method for interpolating segment points: 'idw', 'nn', 'tin', 'kriging' (broke)
    )
    interpolation_method_dem: str = "tin"  # e.g. tin, idw, natural_neighbor

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
    segment_popularity_nn_max_dist: float = 1000.0  # Meters
    segment_popularity_tin_max_triangle_edge_length: float = 100.0  # Meters
    segment_popularity_buffer_distance: float = 5.0  # Meters, for vector output
    segment_age_calculation_method: str = "days"  # 'days', 'years'
    strava_api_request_delay: float = 0.05 # Base delay between successful requests
    segment_collection_sample_size: int = Field(
        default=500,
        description="Number of points to sample and search in each run of the collection script.",
    )
    segment_collection_retry_delay: float = Field(
        default=190.0, # Seconds (190 = 3 minutes + 10 seconds)
        description="Delay in seconds after hitting a Strava API rate limit (429 error).",
    )
    segment_collection_max_retries: int = Field(
        default=0,
        description="Maximum number of retries after hitting a rate limit before giving up on a point.",
    )
    segment_collection_simplify_tolerance_projected: float = 0.001  # Meters
    segment_details_max_api_calls: int = 5 # During dev, change to <=5

    traffic_buffer_distance: float = 500.0  # Meters, for buffering traffic stations
    traffic_interpolation_power: float = 2.0  # For IDW interpolation
    road_buffer_distance: Optional[float] = Field(
        default=5.0, description="Optional buffer for road/lane matching (meters)"
    )  # TODO remove if not used
    
    slope_units: str = "percent"  # 'degrees' or 'percent'
    
    bike_lane_buffer: float = Field(
        default=20.0, description="Buffer size for bike lanes (meters)"
    )

    # Heatmap (Average Speed) IDW Settings
    heatmap_idw_cell_size: float = 10.0  # Cell size for the output raster (meters)
    heatmap_idw_weight: float = 1.0  # Weight parameter for IDW
    heatmap_idw_radius: float = 500.0  # Search radius for IDW (meters)
    heatmap_idw_min_points: int = 150  # Minimum number of points required within radius
    heatmap_sample_fraction: float = (
        0.75  # Fraction of speed points to build raster from
    )

    # Elevation settings 
    dem_idw_weight: float = 1.0  # Weight parameter for IDW
    dem_idw_radius: float = 500.0  # Search radius for IDW (meters)
    dem_idw_min_points: int = 5  # Minimum number of points required within radius
    dem_tin_max_triangle_edge_length: float = 1000.0  # Meters


    # Kriging specific parameters (if used for popularity)
    kriging_model: str = "spherical"  # e.g., spherical, exponential, gaussian
    kriging_range: Optional[float] = None  # Search radius (if None, WBT might estimate)
    kriging_sill: Optional[float] = None
    kriging_nugget: Optional[float] = None

    # Cost Function Settings
    cost_slope_weight: float = Field(
        default=2.0,
        description="Weight factor for slope contribution to cost (higher value = higher cost for steep slopes)."
    )
    cost_speed_weight: float = Field(
        default=0.75, # Changed default to positive, logic in cost_distance uses (threshold - speed)
        description="Weight factor for speed contribution to cost. Applied as weight * (threshold - speed)."
    )
    cost_speed_threshold_ms: float = Field(
        default=6.0,
        description="Speed threshold in m/s. Speeds below this increase cost, speeds above decrease cost."
    )
    cost_road_buffer_meters: float = Field(
        default=7.5, # Increased default slightly
        description="Buffer distance in meters around roads to define traversable area for cost calculation."
    )
    # cost_road_restriction_value: Optional[float] = None # Kept commented out as masking is preferred

    # displays
    display_segments: bool = True  # Whether to display segments on the map

    # Overlay Settings
    overlay_popularity_threshold: Optional[float] = Field(
        default=0.5, # Example: Normalized popularity score (e.g., efforts_per_age > 0.5)
        description="Popularity threshold (normalized) for Overlay A and subsequent overlays."
    )
    overlay_speed_threshold: Optional[float] = Field(
        default=5.0, # Example: m/s (18 km/h)
        description="Average speed threshold (m/s) for Overlay B."
    )
    overlay_traffic_threshold: Optional[float] = Field(
        default=100, # Example: vehicles per hour/day (depending on traffic data aggregation)
        description="Traffic density threshold for Overlay C."
    )
    overlay_cost_threshold: Optional[float] = Field(
        default=0.3, # Example: Normalized cost (lower is better)
        description="Normalized cost distance threshold for Overlay D (e.g., keep segments with cost < threshold)."
    )

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
