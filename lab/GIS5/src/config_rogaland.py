from pathlib import Path
from pydantic import BaseModel, Field, DirectoryPath, FilePath
from typing import Optional

# Determine the base directory relative to this config file
# config.py is in src/, so BASE_DIR is the parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    """Configuration for base directories and main input paths."""

    base_dir: Path = BASE_DIR
    data_dir: DirectoryPath = Field(default_factory=lambda: BASE_DIR / "GIS5_datafiles")
    # Construct gdb_path relative to BASE_DIR and the known data subdir to avoid recursion
    gdb_path: FilePath = Field(
        default_factory=lambda: (BASE_DIR / "GIS5_datafiles") / "DEM_Rogaland.gdb"
    )
    output_dir: Path = Field(
        default_factory=lambda: BASE_DIR
        / "output_rogaland"  # Use a fixed name for pro, date name for dev
        # / f"output_py_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    # Path for the input TopoToRaster (ANUDEM) file, relative to data_dir
    toporaster_all_input_tif: FilePath = Field(
        default_factory=lambda: (BASE_DIR / "GIS5_datafiles")
        / "TopoRaster_Rogaland.tif"
    )
    # transect_input_shp removed, as we will create it based on coordinates
    grass_executable_path: Optional[str] = (
        "/Applications/GRASS-8.4.app/Contents/MacOS/Grass.sh"  # Optional path to GRASS GIS executable
    )

    # --- Paths specific to the raw data preparation task ---
    # Define the input GDB containing the raw/unprocessed data
    input_raw_gdb: Optional[FilePath] = Field(
        default=None,  # Default to None, requires explicit setting if task is run
        description="Path to the input Geodatabase with raw data for preparation. NOTE: Due to driver limitations (see prepare_raw_data.py), this task currently requires this path to point to a MANUALLY CONVERTED GeoPackage (.gpkg) or similar open format, NOT the original .gdb.",
    )
    # Define the output directory for the prepared data
    output_dir_prepared: Path = Field(
        default_factory=lambda: BASE_DIR / "output_prepared_data",
        description="Directory to save the prepared data (e.g., clipped layers).",
    )

    # Ensure directories exist or create them if needed (especially output)
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure output directories exist after initialization
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Main DEM output
        self.output_dir_prepared.mkdir(
            parents=True, exist_ok=True
        )  # Prepared data output

        # Validate data_dir and main gdb_path exist
        if not self.data_dir.is_dir():
            raise NotADirectoryError(
                f"Data directory not found or is not a directory: {self.data_dir}"
            )
        # A File Geodatabase (.gdb) is a directory, not a file
        if not self.gdb_path.is_dir():
            raise NotADirectoryError(
                f"Main Geodatabase directory not found or is not a directory: {self.gdb_path}"
            )
        # Validate input_raw_gdb *if* it's set (it's Optional now)
        if self.input_raw_gdb and not self.input_raw_gdb.is_dir():
            raise NotADirectoryError(
                f"Input Raw Geodatabase directory not found or is not a directory: {self.input_raw_gdb}"
            )
        # Removed validation for toporaster_all_input_tif to allow soft failure if missing


class InputLayersConfig(BaseModel):
    """Layer names within the input Geodatabase."""

    contour_layer: str = "contour_arc"
    river_layer: str = "rivers_arc"
    lake_layer: str = "lakes_polygon"
    points_layer: str = "elevation_point"
    # Field names (add flexibility)
    contour_elevation_field: str = "hoyde"
    point_elevation_field_candidates: list[str] = Field(
        default=[
            "RASTERVALU",
            "POINT_Z",
            "Elevation",
            "elevation",
            "Z_Value",
            "Value",
            "HOEYDE",
            "hoyde",
        ]
    )


class OutputFilesConfig(BaseModel):
    """Relative filenames for output files within the output directory."""

    # Intermediate Shapefiles
    contour_shp: str = "contours.shp"
    river_shp: str = "rivers.shp"
    lake_shp: str = "lakes.shp"
    points_shp: str = "points.shp"
    contour_points_shp: str = "contour_points.shp"  # Intermediate for interpolation
    contour_raster_temp: str = (
        "contour_raster_temp.tif"  # Intermediate for interpolation
    )
    contour_points_with_value_shp: str = (
        "contour_points_with_value.shp"  # Intermediate for interpolation/TIN
    )

    # DEMs
    # Contour-based
    dem_interpolated_contour_tif: str = (
        "dem_interpolated_contour.tif"  # Renamed from dem_interpolated_tif
    )
    dem_topo_contour_tif: str = "dem_topo_contour.tif"  # Renamed from dem_topo_tif
    # Elevation Point-based
    dem_interpolated_points_tif: str = "dem_interpolated_points.tif"  # New
    dem_topo_points_tif: str = "dem_topo_points.tif"  # New
    # Stream Burn (based on Contour TIN)
    dem_stream_burned_tif: str = (
        "dem_stream_burned.tif"  # Output from GRASS stream burning (based on Contour TIN)
    )

    # Hydrology outputs (potentially needed for long profile)
    d8_pointer_tif: str = "d8_pointer.tif"  # Assuming this is generated elsewhere
    streams_tif: str = "streams.tif"  # Assuming this is generated elsewhere

    # Analysis Outputs
    hillshade_interpolated_contour_tif: str = (
        "hillshade_interpolated_contour.tif"  # Renamed from hillshade_interpolated_tif
    )
    hillshade_topo_contour_tif: str = (
        "hillshade_topo_contour.tif"  # Renamed from hillshade_topo_tif
    )
    hillshade_interpolated_points_tif: str = "hillshade_interpolated_points.tif"  # New
    hillshade_topo_points_tif: str = "hillshade_topo_points.tif"  # New
    slope_interpolated_contour_tif: str = (
        "slope_interpolated_contour.tif"  # Renamed from slope_interpolated_tif
    )
    slope_topo_contour_tif: str = "slope_topo_contour.tif"  # Renamed
    slope_interpolated_points_tif: str = "slope_interpolated_points.tif"  # New
    slope_topo_points_tif: str = "slope_topo_points.tif"  # New
    contours_interpolated_contour_shp: str = (
        "contours_interpolated_contour.shp"  # Renamed
    )
    contours_topo_contour_shp: str = "contours_topo_contour.shp"  # Renamed
    contours_interpolated_points_shp: str = "contours_interpolated_points.shp"  # New
    contours_topo_points_shp: str = "contours_topo_points.shp"  # New
    contours_toporaster_all_shp: str = "contours_toporaster_all.shp"  # ANUDEM contours
    contours_stream_burned_shp: str = (
        "contours_stream_burned.shp"  # Stream Burn contours
    )
    hillshade_toporaster_all_tif: str = (
        "hillshade_toporaster_all.tif"  # ANUDEM hillshade
    )
    hillshade_stream_burned_tif: str = (
        "hillshade_stream_burned.tif"  # Stream Burn hillshade
    )
    slope_toporaster_all_tif: str = "slope_toporaster_all.tif"  # ANUDEM slope
    slope_stream_burned_tif: str = "slope_stream_burned.tif"  # Stream Burn slope
    # TODO: Add diffs for points-based DEMs if needed
    dem_diff_tif: str = (
        "dem_difference_contour.tif"  # Renamed (TopoContour - InterpContour)
    )
    # Profile Analysis Outputs
    profile_analysis_interp_contour_html: str = (
        "profile_interpolated_contour.html"  # Renamed
    )
    profile_analysis_topo_contour_html: str = "profile_topo_contour.html"  # Renamed
    profile_analysis_interp_points_html: str = "profile_interpolated_points.html"  # New
    profile_analysis_topo_points_html: str = "profile_topo_points.html"  # New
    profile_analysis_toporaster_all_html: str = "profile_toporaster_all.html"
    profile_analysis_stream_burned_html: str = "profile_stream_burned.html"

    # Transect file created by the script
    transect_created_shp: str = "transect_generated.shp"

    # Quality Assessment
    rmse_csv: str = "rmse_comparison.csv"
    points_extracted_shp: str = "points_with_dem_values.shp"

    # --- Shared Mappings ---
    # Map internal DEM keys to user-friendly names for reports and plots
    dem_type_map: dict[str, str] = Field(
        default={
            "interp_contour": "Natural Neighbor (Contour)",
            "topo_contour": "TIN Gridding (Contour)",
            "interp_points": "Natural Neighbor (Points)",
            "topo_points": "TIN Gridding (Points)",
            "stream_burn": "Stream Burn (Contour TIN based)",
            "toporaster_all": "ANUDEM (ArcGIS Pro)",
        }
    )

    # Method to get full path
    def get_full_path(self, filename_attr: str, output_dir: Path) -> Path:
        """Returns the full path for a given output filename attribute."""
        filename = getattr(self, filename_attr)
        return output_dir / filename


class ProcessingConfig(BaseModel):
    """Parameters controlling the processing steps."""

    output_cell_size: float = 50.0  # Meters
    contour_interval: float = 50.0  # Meters
    wbt_verbose: bool = False  # Control WhiteboxTools verbosity
    # Stream Burning Config
    enable_stream_burning: bool = True  # Set to True to run GRASS stream burning
    stream_burn_value: float = -10.0  # Value (in elevation units) to burn streams by
    stream_extract_threshold: int = 500  # Threshold for r.stream.extract (cells)
    # Transect is now generated dynamically based on data extent

    # --- Raw Data Preparation Task Config ---
    run_prepare_raw_data: bool = Field(
        default=False,
        description="Set to True to run the raw data prep task (prepare_raw_data.py)."
        " REQUIRES MANUAL DATA CONVERSION FIRST.",
    )
    # Note: prepare_raw_data.py cannot read .gdb directly due to driver limitations.
    # Input data must be manually converted to GPKG (or similar) and path set in PathsConfig.input_raw_gdb.
    target_crs_epsg: int = 25832  # Default CRS for data processing (can be overridden)


class AppConfig(BaseModel):
    """Main application configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    input_layers: InputLayersConfig = Field(default_factory=InputLayersConfig)
    output_files: OutputFilesConfig = Field(default_factory=OutputFilesConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)


# Instantiate the main config object for easy import
settings = AppConfig()

# Example usage (optional, for testing)
if __name__ == "__main__":
    print("Configuration loaded:")
    print(f"Base Directory: {settings.paths.base_dir}")
    print(f"Data Directory: {settings.paths.data_dir}")
    print(f"GDB Path: {settings.paths.gdb_path}")
    print(f"Output Directory: {settings.paths.output_dir}")
    print(f"Contour Layer: {settings.input_layers.contour_layer}")
    print(f"Output Cell Size: {settings.processing.output_cell_size}")
    print(
        f"Interpolated DEM Path: {settings.output_files.get_full_path('dem_interpolated_tif', settings.paths.output_dir)}"
    )
    # Check if output dir was created
    print(f"Output directory exists: {settings.paths.output_dir.exists()}")
