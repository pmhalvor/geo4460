import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from whitebox import WhiteboxTools
from math import sqrt
from sklearn.metrics import mean_squared_error
from typing import Optional

from pydantic import BaseModel

def assess_dem_quality(settings: BaseModel, wbt: WhiteboxTools, points_shp_path: Path, dem_interp_path: Path, dem_topo_path: Path, dem_toporaster_all_path: Path, point_elev_field: Optional[str]):
    """
    Performs quality assessment by calculating RMSE between DEMs and elevation points.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        points_shp_path: Path to the original elevation points shapefile.
        dem_interp_path: Path to the interpolated DEM raster.
        dem_topo_path: Path to the TIN gridded DEM raster.
        dem_toporaster_all_path: Path to the TopoToRaster (ArcGIS Pro) DEM raster.
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
        # Continue to check other DEMs
    if not dem_toporaster_all_path.exists():
        print(f"Error: TopoRaster_all DEM not found ({dem_toporaster_all_path}). Skipping RMSE calculation for this DEM.")
        # Continue to check other DEMs
    if not points_shp_path.exists():
        print(f"Error: Points shapefile not found ({points_shp_path}). Skipping RMSE calculation.")
        return # Cannot proceed without points
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

        # Save potentially renamed file before next extraction
        if topo_col_name: # Only save if rename was successful or attempted
            points_gdf_after_topo.to_file(str(points_extracted_path), driver='ESRI Shapefile')
        cols_before_toporaster_all_extraction = set(points_gdf_after_topo.columns) # Update baseline columns


        # --- Extract Values from TopoRaster_all DEM ---
        toporaster_all_col_name = None # Initialize
        if dem_toporaster_all_path.exists():
            # Use the actual path variable in the print statement
            print(f"  - Extracting values from TopoRaster_all DEM ({dem_toporaster_all_path.name})...")
            wbt.extract_raster_values_at_points(
                inputs=str(dem_toporaster_all_path),
                points=str(points_extracted_path) # Modifies the same file again
            )

            # --- Robust Renaming for TopoRaster_all DEM Column ---
            points_gdf_after_toporaster_all = gpd.read_file(str(points_extracted_path))
            cols_after_toporaster_all = set(points_gdf_after_toporaster_all.columns)
            new_cols_toporaster_all = list(cols_after_toporaster_all - cols_before_toporaster_all_extraction)

            toporaster_all_col_name = 'DEM_TopoR_All' # Target name
            if len(new_cols_toporaster_all) == 1:
                added_col_toporaster_all = new_cols_toporaster_all[0]
                points_gdf_after_toporaster_all.rename(columns={added_col_toporaster_all: toporaster_all_col_name}, inplace=True)
                print(f"    - Renamed third extracted value column '{added_col_toporaster_all}' to '{toporaster_all_col_name}'")
            elif 'VALUE1' in new_cols_toporaster_all: # Check if 'VALUE1' is the *new* column this time
                 print(f"    - Warning: Could not uniquely identify new column. Attempting to rename 'VALUE1' to '{toporaster_all_col_name}'.")
                 points_gdf_after_toporaster_all.rename(columns={'VALUE1': toporaster_all_col_name}, inplace=True)
            elif 'VALUE1' in cols_after_toporaster_all and toporaster_all_col_name not in cols_after_toporaster_all: # Check if 'VALUE1' exists but wasn't the new one
                 print(f"    - Warning: Fallback rename. Renaming existing 'VALUE1' to '{toporaster_all_col_name}'.")
                 points_gdf_after_toporaster_all.rename(columns={'VALUE1': toporaster_all_col_name}, inplace=True)
            elif toporaster_all_col_name not in cols_after_toporaster_all:
                 print(f"    - Warning: Could not identify or rename column for TopoRaster_all DEM. Found new: {new_cols_toporaster_all}. All: {list(cols_after_toporaster_all)}")
                 toporaster_all_col_name = None # Flag that renaming failed

            # Final GDF for calculations
            points_extracted_gdf = points_gdf_after_toporaster_all
            print(f"  - Extracted values saved to: {points_extracted_path}")
        else:
            # If the TopoRaster_all DEM didn't exist, use the GDF from before this step
            points_extracted_gdf = points_gdf_after_topo
            print(f"  - Skipping extraction for non-existent TopoRaster_all DEM.")


        # --- Calculate RMSE ---
        print("  - Calculating RMSE...")
        required_cols = [point_elev_field]
        # Only add columns if they were successfully created and renamed
        if interp_col_name: required_cols.append(interp_col_name)
        if topo_col_name: required_cols.append(topo_col_name)
        if toporaster_all_col_name: required_cols.append(toporaster_all_col_name)

        # Check if *at least* the elevation field and one DEM column exist
        if len(required_cols) < 2 or not all(col in points_extracted_gdf.columns for col in required_cols):
             present_cols = points_extracted_gdf.columns.tolist()
             missing = [col for col in required_cols if col not in present_cols]
             if point_elev_field not in present_cols:
                 raise ValueError(f"Missing the original elevation point field '{point_elev_field}'. Cannot calculate RMSE.")
             elif len(required_cols) < 2:
                 print(f"    - Warning: No DEM columns were successfully extracted or renamed. Cannot calculate RMSE.")
                 # Set results to empty and skip calculation
                 rmse_results = {'DEM_Type': [], 'RMSE': [], 'N_Points': []}
                 n_points = 0 # Ensure n_points is defined
             else:
                 print(f"    - Warning: Missing some DEM columns for RMSE: {missing}. Available: {present_cols}. Proceeding with available columns.")
                 # We can proceed, but the missing columns won't be calculated.
                 # Remove missing columns from required_cols for the dropna step
                 required_cols = [col for col in required_cols if col in present_cols]

        # Clean data: Replace Inf with NaN, then drop rows with NaN in essential columns
        # Only proceed if we have at least one DEM column + the original elevation
        if len(required_cols) >= 2:
            points_extracted_gdf.replace([np.inf, -np.inf], np.nan, inplace=True)
            points_valid_gdf = points_extracted_gdf.dropna(subset=required_cols).copy() # Use copy to avoid SettingWithCopyWarning
            n_points = len(points_valid_gdf)
        else:
            # Handle case where no DEM columns were available from the start
             points_valid_gdf = pd.DataFrame() # Empty dataframe
             n_points = 0


        if n_points == 0:
            print("    - Error: No valid points found after dropping NaN/Inf values or no DEM columns available. Cannot calculate RMSE.")
            # Ensure rmse_results is initialized if it wasn't already
            if 'rmse_results' not in locals():
                rmse_results = {'DEM_Type': [], 'RMSE': [], 'N_Points': []}
        else:
            measured = points_valid_gdf[point_elev_field]
            rmse_interp = np.nan
            rmse_topo = np.nan
            rmse_toporaster_all = np.nan

            if interp_col_name and interp_col_name in points_valid_gdf.columns:
                interp_dem_values = points_valid_gdf[interp_col_name]
                rmse_interp = sqrt(mean_squared_error(measured, interp_dem_values))
                print(f"    - RMSE (Interpolated DEM vs Points): {rmse_interp:.3f} (using {n_points} points)")
            elif interp_col_name:
                 print("    - Skipping RMSE calculation for Interpolated DEM due to column missing after NaN drop.")
            else:
                 print("    - Skipping RMSE calculation for Interpolated DEM due to earlier column renaming issue.")

            if topo_col_name and topo_col_name in points_valid_gdf.columns:
                topo_dem_values = points_valid_gdf[topo_col_name]
                rmse_topo = sqrt(mean_squared_error(measured, topo_dem_values))
                print(f"    - RMSE (Topo DEM vs Points): {rmse_topo:.3f} (using {n_points} points)")
            elif topo_col_name:
                 print("    - Skipping RMSE calculation for Topo DEM due to column missing after NaN drop.")
            else:
                 print("    - Skipping RMSE calculation for Topo DEM due to earlier column renaming issue.")

            if toporaster_all_col_name and toporaster_all_col_name in points_valid_gdf.columns:
                toporaster_all_dem_values = points_valid_gdf[toporaster_all_col_name]
                rmse_toporaster_all = sqrt(mean_squared_error(measured, toporaster_all_dem_values))
                print(f"    - RMSE (TopoRaster_all DEM vs Points): {rmse_toporaster_all:.3f} (using {n_points} points)")
            elif toporaster_all_col_name:
                 print("    - Skipping RMSE calculation for TopoRaster_all DEM due to column missing after NaN drop.")
            else:
                 print("    - Skipping RMSE calculation for TopoRaster_all DEM due to earlier column renaming issue or file not found.")


            # Prepare results for CSV (Corrected names and inclusion of all DEMs)
            rmse_results = {'DEM_Type': [], 'RMSE': [], 'N_Points': []}
            if interp_col_name and interp_col_name in points_valid_gdf.columns:
                rmse_results['DEM_Type'].append('Natural Neighbour') # Renamed
                rmse_results['RMSE'].append(rmse_interp)
                rmse_results['N_Points'].append(n_points)
            if topo_col_name and topo_col_name in points_valid_gdf.columns:
                rmse_results['DEM_Type'].append('TIN Gridding (Contours)') # Renamed
                rmse_results['RMSE'].append(rmse_topo)
                rmse_results['N_Points'].append(n_points)
            if toporaster_all_col_name and toporaster_all_col_name in points_valid_gdf.columns:
                rmse_results['DEM_Type'].append('TopoToRaster (ArcGIS Pro)') # Kept name
                rmse_results['RMSE'].append(rmse_toporaster_all)
                rmse_results['N_Points'].append(n_points)

        # Save results to CSV (Moved the saving logic here, removed duplicated block below)
        rmse_df = pd.DataFrame(rmse_results)
        rmse_df.to_csv(str(rmse_csv_path), index=False)
        print(f"    - RMSE results saved to: {rmse_csv_path}")

    except Exception as e:
        print(f"An error occurred during Quality Assessment: {e}")
        raise # Re-raise

    print("--- Quality Assessment Complete ---")
