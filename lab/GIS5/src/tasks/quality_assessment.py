import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from whitebox import WhiteboxTools
from math import sqrt
from sklearn.metrics import mean_squared_error
from typing import Optional

from src.config import AppConfig

def assess_dem_quality(settings: AppConfig, wbt: WhiteboxTools, points_shp_path: Path, dem_interp_path: Path, dem_topo_path: Path, point_elev_field: Optional[str]):
    """
    Performs quality assessment by calculating RMSE between DEMs and elevation points.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        points_shp_path: Path to the original elevation points shapefile.
        dem_interp_path: Path to the interpolated DEM raster.
        dem_topo_path: Path to the TIN gridded DEM raster.
        point_elev_field: The name of the elevation field in the points shapefile. If None, RMSE cannot be calculated.

    Raises:
        FileNotFoundError: If required input files (DEMs, points shapefile) are missing.
        ValueError: If the point elevation field is None or required columns are missing after extraction.
        Exception: If WhiteboxTools commands fail or other errors occur.
    """
    print("\n3. Quality Assessment (RMSE)...")
    output_dir = settings.paths.output_dir
    output_files = settings.output_files
    processing = settings.processing

    # Define input/output paths for this task
    points_extracted_path = output_files.get_full_path('points_extracted_shp', output_dir)
    rmse_csv_path = output_files.get_full_path('rmse_csv', output_dir)

    # Set WBT verbosity
    wbt.set_verbose_mode(processing.wbt_verbose)

    # --- Input Checks ---
    if not dem_interp_path.exists():
        print(f"Error: Interpolated DEM not found ({dem_interp_path}). Skipping RMSE calculation.")
        return # Or raise error if this is critical
    if not dem_topo_path.exists():
        print(f"Error: Topo DEM not found ({dem_topo_path}). Skipping RMSE calculation.")
        return
    if not points_shp_path.exists():
        print(f"Error: Points shapefile not found ({points_shp_path}). Skipping RMSE calculation.")
        return
    if point_elev_field is None:
         print(f"Error: Unknown elevation field in points layer. Skipping RMSE calculation.")
         return

    try:
        # --- Prepare Points File ---
        print(f"  - Copying original points shapefile to {points_extracted_path} for modification...")
        # Read and write with geopandas to handle all associated files (.dbf, .shx, etc.)
        points_orig_gdf = gpd.read_file(str(points_shp_path))
        points_orig_gdf.to_file(str(points_extracted_path), driver='ESRI Shapefile')
        print("    - Copy complete.")
        cols_before_any_extraction = set(points_orig_gdf.columns)


        # --- Extract Values from Interpolated DEM ---
        print(f"  - Extracting values from Interpolated DEM ({output_files.dem_interpolated_tif})...")
        wbt.extract_raster_values_at_points(
            inputs=str(dem_interp_path),
            points=str(points_extracted_path) # Tool modifies this file in place
        )

        # --- Robust Renaming for Interpolated DEM Column ---
        points_gdf_after_interp = gpd.read_file(str(points_extracted_path))
        cols_after_interp = set(points_gdf_after_interp.columns)
        new_cols_interp = list(cols_after_interp - cols_before_any_extraction)

        interp_col_name = 'DEM_Interp' # Target name
        if len(new_cols_interp) == 1:
            added_col_interp = new_cols_interp[0]
            points_gdf_after_interp.rename(columns={added_col_interp: interp_col_name}, inplace=True)
            print(f"    - Renamed extracted value column '{added_col_interp}' to '{interp_col_name}'")
        elif 'VALUE1' in cols_after_interp and interp_col_name not in cols_after_interp:
             print(f"    - Warning: Could not uniquely identify new column. Attempting to rename 'VALUE1' to '{interp_col_name}'.")
             points_gdf_after_interp.rename(columns={'VALUE1': interp_col_name}, inplace=True)
        elif interp_col_name not in cols_after_interp:
             print(f"    - Warning: Could not identify or rename column for Interpolated DEM. Found new: {new_cols_interp}. All: {list(cols_after_interp)}")
             # Decide how to handle: raise error, skip RMSE for this DEM, etc.
             # For now, we'll proceed, but calculation might fail.
             interp_col_name = None # Flag that renaming failed

        # Save potentially renamed file before next extraction
        if interp_col_name: # Only save if rename was successful or attempted
            points_gdf_after_interp.to_file(str(points_extracted_path), driver='ESRI Shapefile')
        cols_before_topo_extraction = set(points_gdf_after_interp.columns) # Update baseline columns


        # --- Extract Values from Topo DEM ---
        print(f"  - Extracting values from TIN Gridding DEM ({output_files.dem_topo_tif})...")
        wbt.extract_raster_values_at_points(
            inputs=str(dem_topo_path),
            points=str(points_extracted_path) # Modifies the same file again
        )

        # --- Robust Renaming for Topo DEM Column ---
        points_gdf_after_topo = gpd.read_file(str(points_extracted_path))
        cols_after_topo = set(points_gdf_after_topo.columns)
        new_cols_topo = list(cols_after_topo - cols_before_topo_extraction)

        topo_col_name = 'DEM_Topo' # Target name
        if len(new_cols_topo) == 1:
            added_col_topo = new_cols_topo[0]
            points_gdf_after_topo.rename(columns={added_col_topo: topo_col_name}, inplace=True)
            print(f"    - Renamed second extracted value column '{added_col_topo}' to '{topo_col_name}'")
        elif 'VALUE1' in new_cols_topo: # Check if 'VALUE1' is the *new* column this time
             print(f"    - Warning: Could not uniquely identify new column. Attempting to rename 'VALUE1' to '{topo_col_name}'.")
             points_gdf_after_topo.rename(columns={'VALUE1': topo_col_name}, inplace=True)
        elif 'VALUE1' in cols_after_topo and topo_col_name not in cols_after_topo: # Check if 'VALUE1' exists but wasn't the new one
             print(f"    - Warning: Fallback rename. Renaming existing 'VALUE1' to '{topo_col_name}'.")
             points_gdf_after_topo.rename(columns={'VALUE1': topo_col_name}, inplace=True)
        elif topo_col_name not in cols_after_topo:
             print(f"    - Warning: Could not identify or rename column for Topo DEM. Found new: {new_cols_topo}. All: {list(cols_after_topo)}")
             topo_col_name = None # Flag that renaming failed

        # Final GDF for calculations
        points_extracted_gdf = points_gdf_after_topo
        print(f"  - Extracted values saved to: {points_extracted_path}")


        # --- Calculate RMSE ---
        print("  - Calculating RMSE...")
        required_cols = [point_elev_field]
        if interp_col_name: required_cols.append(interp_col_name)
        if topo_col_name: required_cols.append(topo_col_name)

        if not all(col in points_extracted_gdf.columns for col in required_cols):
             missing = [col for col in required_cols if col not in points_extracted_gdf.columns]
             raise ValueError(f"Missing required columns for RMSE: {missing}. Available: {points_extracted_gdf.columns.tolist()}")

        # Clean data: Replace Inf with NaN, then drop rows with NaN in essential columns
        points_extracted_gdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        points_valid_gdf = points_extracted_gdf.dropna(subset=required_cols).copy() # Use copy to avoid SettingWithCopyWarning

        n_points = len(points_valid_gdf)
        if n_points == 0:
            print("    - Error: No valid points found after dropping NaN/Inf values. Cannot calculate RMSE.")
            rmse_results = {'DEM_Type': [], 'RMSE': [], 'N_Points': []}
        else:
            measured = points_valid_gdf[point_elev_field]
            rmse_interp = np.nan
            rmse_topo = np.nan

            if interp_col_name:
                interp_dem_values = points_valid_gdf[interp_col_name]
                rmse_interp = sqrt(mean_squared_error(measured, interp_dem_values))
                print(f"    - RMSE (Interpolated DEM vs Points): {rmse_interp:.3f} (using {n_points} points)")
            else:
                 print("    - Skipping RMSE calculation for Interpolated DEM due to column renaming issue.")

            if topo_col_name:
                topo_dem_values = points_valid_gdf[topo_col_name]
                rmse_topo = sqrt(mean_squared_error(measured, topo_dem_values))
                print(f"    - RMSE (Topo DEM vs Points): {rmse_topo:.3f} (using {n_points} points)")
            else:
                 print("    - Skipping RMSE calculation for Topo DEM due to column renaming issue.")

            # Prepare results for CSV
            rmse_results = {'DEM_Type': [], 'RMSE': [], 'N_Points': []}
            if interp_col_name:
                rmse_results['DEM_Type'].append('Interpolated (Natural Neighbour)')
                rmse_results['RMSE'].append(rmse_interp)
                rmse_results['N_Points'].append(n_points)
            if topo_col_name:
                rmse_results['DEM_Type'].append('TIN Gridding')
                rmse_results['RMSE'].append(rmse_topo)
                rmse_results['N_Points'].append(n_points)

        # Save results to CSV
        rmse_df = pd.DataFrame(rmse_results)
        rmse_df.to_csv(str(rmse_csv_path), index=False)
        print(f"    - RMSE results saved to: {rmse_csv_path}")

    except Exception as e:
        print(f"An error occurred during Quality Assessment: {e}")
        raise # Re-raise

    print("--- Quality Assessment Complete ---")
