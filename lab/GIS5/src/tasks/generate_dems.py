import os
from pathlib import Path
from whitebox import WhiteboxTools

from src.config import AppConfig

def generate_dems(settings: AppConfig, wbt: WhiteboxTools, contour_shp_path: Path, contour_elev_field: str):
    """
    Generates DEM rasters using interpolation and TIN gridding methods.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        contour_shp_path: Path to the contour shapefile.
        contour_elev_field: Name of the elevation field in the contour shapefile.

    Raises:
        FileNotFoundError: If required intermediate files are missing or outputs are not created.
        Exception: If any WhiteboxTools command fails.
    """
    print("\n2. Generating DEMs...")
    output_dir = settings.paths.output_dir
    output_files = settings.output_files
    processing = settings.processing

    # Define output paths
    dem_interp_path = output_files.get_full_path('dem_interpolated_tif', output_dir)
    dem_topo_path = output_files.get_full_path('dem_topo_tif', output_dir)

    # Intermediate file paths needed for DEM generation
    contour_raster_path = output_files.get_full_path('contour_raster_temp', output_dir)
    contour_points_with_value_path = output_files.get_full_path('contour_points_with_value_shp', output_dir)

    # Set WBT verbosity
    wbt.set_verbose_mode(processing.wbt_verbose)

    # --- Method 1: Interpolation (Natural Neighbour) ---
    print("  - Method 1: Natural Neighbour Interpolation from Contours...")
    try:
        # Convert contour lines to raster, burning elevation attribute
        print(f"    - Converting contour lines to raster ({output_files.contour_raster_temp})...")
        wbt.vector_lines_to_raster(
            i=str(contour_shp_path),
            output=str(contour_raster_path),
            field=contour_elev_field,
            nodata=-9999.0,
            cell_size=processing.output_cell_size,
            base=None # Let WBT determine extent
        )
        print(f"    - Contour raster saved to: {contour_raster_path}")
        if not contour_raster_path.exists():
             raise FileNotFoundError(f"Intermediate contour raster not created: {contour_raster_path}")

        # Convert contour raster back to points
        print(f"    - Converting contour raster to points ({output_files.contour_points_with_value_shp})...")
        wbt.raster_to_vector_points(
            i=str(contour_raster_path),
            output=str(contour_points_with_value_path)
        )
        print(f"    - Points with elevation saved to: {contour_points_with_value_path}")
        if not contour_points_with_value_path.exists():
            raise FileNotFoundError(f"Points file with values not created: {contour_points_with_value_path}")

        # Interpolate using Natural Neighbour from the generated points
        print("    - Running Natural Neighbour Interpolation...")
        wbt.natural_neighbour_interpolation(
            i=str(contour_points_with_value_path),
            field='VALUE', # Default field name from raster_to_vector_points
            output=str(dem_interp_path),
            cell_size=processing.output_cell_size
            # extent=[minx, maxx, miny, maxy] # WBT should infer from input points
        )
        print(f"    - Interpolated (Natural Neighbour) DEM saved to: {dem_interp_path}")
        if not dem_interp_path.exists():
             raise FileNotFoundError(f"Interpolated DEM file not created: {dem_interp_path}")

    except Exception as e:
        print(f"Error during Interpolation DEM generation: {e}")
        raise # Re-raise to halt workflow if critical

    # --- Method 2: TIN Gridding ---
    print("\n  - Method 2: TIN Gridding from Contour Points with Values...")
    try:
        # Ensure contour points with values exist from previous step
        if not contour_points_with_value_path.exists():
             raise FileNotFoundError(f"Contour points file needed for TIN Gridding not found: {contour_points_with_value_path}")

        print("    - Running TIN Gridding...")
        wbt.tin_gridding(
            i=str(contour_points_with_value_path),
            field='VALUE',
            output=str(dem_topo_path),
            resolution=processing.output_cell_size
        )
        print(f"    - TIN Gridding DEM saved to: {dem_topo_path}")
        if not dem_topo_path.exists():
             raise FileNotFoundError(f"TIN Gridding DEM file not created: {dem_topo_path}")

    except Exception as e:
        print(f"Error during TIN Gridding DEM generation: {e}")
        raise # Re-raise

    print("--- DEM Generation Complete ---")
