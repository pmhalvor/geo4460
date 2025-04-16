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
        default_factory=lambda: (BASE_DIR / "GIS5_datafiles") / "DEM_analysis_DATA.gdb"
    )
    output_dir: Path = Field(
        default_factory=lambda: BASE_DIR
        / "output_py"  # Use a fixed name for pro, date name for dev
        # / f"output_py_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )
    # Path for the input TopoToRaster (ANUDEM) file, relative to data_dir
    toporaster_all_input_tif: FilePath = Field(
        default_factory=lambda: (BASE_DIR / "GIS5_datafiles") / "TopoRaster_all.tif"
    )
    # transect_input_shp removed, as we will create it based on coordinates
    grass_executable_path: Optional[str] = (
        "/Applications/GRASS-8.4.app/Contents/MacOS/Grass.sh"  # Optional path to GRASS GIS executable
    )

    # Ensure directories exist or create them if needed (especially output)
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure output directory exists after initialization
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Validate data_dir and gdb_path exist
        if not self.data_dir.is_dir():
            raise NotADirectoryError(
                f"Data directory not found or is not a directory: {self.data_dir}"
            )
        # A File Geodatabase (.gdb) is a directory, not a file
        if not self.gdb_path.is_dir():
            raise NotADirectoryError(
                f"Geodatabase directory not found or is not a directory: {self.gdb_path}"
            )
        # Removed validation for toporaster_all_input_tif to allow soft failure if missing


class InputLayersConfig(BaseModel):
    """Layer names within the input Geodatabase."""

    contour_layer: str = "contour_arc"
    river_layer: str = "rivers_arc"
    lake_layer: str = "lakes_polygon"
    points_layer: str = "elevationp_point"  # TODO remove extra p (depend on input data)
    # Field names (add flexibility)
    contour_elevation_field: str = "HOEYDE"
    point_elevation_field_candidates: list[str] = Field(
        default=["RASTERVALU", "POINT_Z", "Elevation", "Z_Value", "Value", "HOEYDE"]
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
    dem_interpolated_tif: str = "dem_interpolated.tif"
    dem_topo_tif: str = (
        "dem_topo_to_raster.tif"  # Keep name consistent with original script
    )
    dem_stream_burned_tif: str = (
        "dem_stream_burned.tif"  # Output from GRASS stream burning
    )
    # Hydrology outputs (potentially needed for long profile)
    d8_pointer_tif: str = "d8_pointer.tif"  # Assuming this is generated elsewhere
    streams_tif: str = "streams.tif"  # Assuming this is generated elsewhere

    # Analysis Outputs
    hillshade_interpolated_tif: str = "hillshade_interpolated.tif"
    hillshade_topo_tif: str = "hillshade_topo.tif"
    slope_interpolated_tif: str = "slope_interpolated.tif"
    slope_topo_tif: str = "slope_topo.tif"
    contours_interpolated_shp: str = "contours_interpolated.shp"
    contours_topo_shp: str = "contours_topo.shp"
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
    dem_diff_tif: str = "dem_difference.tif"  # Original difference (Topo - Interp)
    # Profile Analysis Outputs
    profile_analysis_interp_html: str = "profile_interpolated.html"
    profile_analysis_topo_html: str = "profile_topo.html"
    profile_analysis_toporaster_all_html: str = "profile_toporaster_all.html"
    profile_analysis_stream_burned_html: str = "profile_stream_burned.html"

    # Transect file created by the script
    transect_created_shp: str = "transect_generated.shp"

    # Quality Assessment
    rmse_csv: str = "rmse_comparison.csv"
    points_extracted_shp: str = "points_with_dem_values.shp"

    # Method to get full path
    def get_full_path(self, filename_attr: str, output_dir: Path) -> Path:
        """Returns the full path for a given output filename attribute."""
        filename = getattr(self, filename_attr)
        return output_dir / filename


class ProcessingConfig(BaseModel):
    """Parameters controlling the processing steps."""

    output_cell_size: float = 50.0  # Meters
    contour_interval: float = 10.0  # Meters
    wbt_verbose: bool = False  # Control WhiteboxTools verbosity
    # Stream Burning Config
    enable_stream_burning: bool = True  # Set to True to run GRASS stream burning
    stream_burn_value: float = -10.0  # Value (in elevation units) to burn streams by
    stream_extract_threshold: int = 1  # Threshold for r.stream.extract (cells)
    # Transect Definition (CRS will be taken from input data)
    transect_start_coords: tuple[float, float] = (
        550000,
        6630000,
    )  # Example coordinates (X, Y)
    transect_end_coords: tuple[float, float] = (
        570000,
        6645000,
    )  # Example coordinates (X, Y)


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
