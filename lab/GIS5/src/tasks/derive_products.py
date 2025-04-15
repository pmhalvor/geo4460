from pathlib import Path
from whitebox import WhiteboxTools

from pydantic import BaseModel


def generate_derived_products(settings: BaseModel, wbt: WhiteboxTools, dem_interp_path: Path, dem_topo_path: Path):
    """
    Generates derived raster and vector products from the DEMs (contours, hillshade, slope, difference).

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        dem_interp_path: Path to the interpolated DEM raster.
        dem_topo_path: Path to the TIN gridded DEM raster.

    Raises:
        FileNotFoundError: If input DEMs are missing.
        Exception: If any WhiteboxTools command fails.
    """
    print("\n4. Further Analysis (Contours, Hillshade, Difference, Slope)...")
    output_dir = settings.paths.output_dir
    output_files = settings.output_files
    processing = settings.processing

    # Define output paths
    contours_interp_path = output_files.get_full_path('contours_interpolated_shp', output_dir)
    contours_topo_path = output_files.get_full_path('contours_topo_shp', output_dir)
    hillshade_interp_path = output_files.get_full_path('hillshade_interpolated_tif', output_dir)
    hillshade_topo_path = output_files.get_full_path('hillshade_topo_tif', output_dir)
    dem_diff_path = output_files.get_full_path('dem_diff_tif', output_dir)
    slope_interp_path = output_files.get_full_path('slope_interpolated_tif', output_dir)
    slope_topo_path = output_files.get_full_path('slope_topo_tif', output_dir)

    # Set WBT verbosity
    wbt.set_verbose_mode(processing.wbt_verbose)

    # --- Input Checks ---
    if not dem_interp_path.exists():
        print(f"Error: Interpolated DEM not found ({dem_interp_path}). Skipping derived product generation.")
        return # Or raise error
    if not dem_topo_path.exists():
        print(f"Error: Topo DEM not found ({dem_topo_path}). Skipping derived product generation.")
        return

    try:
        # a) Generate Contours
        print(f"  - Generating contours (interval: {processing.contour_interval}m)...")
        wbt.contours_from_raster(
            i=str(dem_interp_path),
            output=str(contours_interp_path),
            interval=processing.contour_interval
        )
        print(f"    - Contours from Interpolated DEM saved to: {contours_interp_path}")
        wbt.contours_from_raster(
            i=str(dem_topo_path),
            output=str(contours_topo_path),
            interval=processing.contour_interval
        )
        print(f"    - Contours from Topo DEM saved to: {contours_topo_path}")

        # b) Generate Hillshades
        print("  - Generating hillshades...")
        wbt.hillshade(
            dem=str(dem_interp_path),
            output=str(hillshade_interp_path)
            # Default azimuth=315, altitude=30
        )
        print(f"    - Hillshade from Interpolated DEM saved to: {hillshade_interp_path}")
        wbt.hillshade(
            dem=str(dem_topo_path),
            output=str(hillshade_topo_path)
        )
        print(f"    - Hillshade from Topo DEM saved to: {hillshade_topo_path}")

        # c) Calculate DEM Difference
        print("  - Calculating DEM difference (Topo - Interpolated)...")
        wbt.subtract(
            input1=str(dem_topo_path),
            input2=str(dem_interp_path),
            output=str(dem_diff_path)
        )
        print(f"    - DEM difference map saved to: {dem_diff_path}")

        # d) Calculate Slope
        print("  - Calculating slope...")
        wbt.slope(
            dem=str(dem_interp_path),
            output=str(slope_interp_path)
            # Output units: degrees (default)
        )
        print(f"    - Slope from Interpolated DEM saved to: {slope_interp_path}")
        wbt.slope(
            dem=str(dem_topo_path),
            output=str(slope_topo_path)
        )
        print(f"    - Slope from Topo DEM saved to: {slope_topo_path}")

    except Exception as e:
        print(f"An error occurred during Further Analysis: {e}")
        raise # Re-raise

    print("--- Further Analysis Complete ---")
