import geopandas as gpd
import numpy as np
import pandas as pd
from whitebox import WhiteboxTools  # Using WhiteboxTools for interpolation and analysis
import os
import shutil
from math import sqrt
from sklearn.metrics import mean_squared_error

print("Initializing WhiteboxTools...")
wbt = WhiteboxTools()
print(wbt.version())
# Set WhiteboxTools working directory (optional, defaults to system temp)
# wbt.work_dir = os.path.join(os.getcwd(), "wbt_temp") 
# print(f"WhiteboxTools working directory: {wbt.work_dir}")

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "GIS5_datafiles")
GDB_PATH = os.path.join(DATA_DIR, "DEM_analysis_DATA.gdb")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_py")
OUTPUT_CELL_SIZE = 50.0  # Meters, as specified in the lab description

# Layer names within the GDB
CONTOUR_LAYER = "contour_arc"
RIVER_LAYER = "rivers_arc"
LAKE_LAYER = "lakes_polygon"
POINTS_LAYER = "elevationp_point"

# Output filenames (relative to OUTPUT_DIR)
CONTOUR_SHP = "contours.shp"
RIVER_SHP = "rivers.shp"
LAKE_SHP = "lakes.shp"
POINTS_SHP = "points.shp"

DEM_INTERP_TIF = "dem_interpolated.tif" # Approximation of TIN-to-Raster
DEM_TOPO_TIF = "dem_topo_to_raster.tif"
HILLSHADE_INTERP_TIF = "hillshade_interpolated.tif"
HILLSHADE_TOPO_TIF = "hillshade_topo.tif"
SLOPE_INTERP_TIF = "slope_interpolated.tif"
SLOPE_TOPO_TIF = "slope_topo.tif"
CONTOURS_INTERP_SHP = "contours_interpolated.shp"
CONTOURS_TOPO_SHP = "contours_topo.shp"
DEM_DIFF_TIF = "dem_difference.tif"
RMSE_CSV = "rmse_comparison.csv"
POINTS_EXTRACTED_SHP = "points_with_dem_values.shp"


# --- Helper Functions ---
def setup_output_dir(directory):
    """Creates or clears the output directory."""
    if os.path.exists(directory):
        print(f"Removing existing output directory: {directory}")
        shutil.rmtree(directory)
    print(f"Creating output directory: {directory}")
    os.makedirs(directory)

def get_common_extent_and_crs(layers):
    """Calculates the total bounds and ensures consistent CRS."""
    if not layers:
        raise ValueError("Need at least one layer to determine extent and CRS.")
    
    crs = layers[0].crs
    if crs is None:
        print("Warning: Input layer CRS is None. Assuming a projected CRS for calculations.")
        # Attempt to guess or assign a common local CRS if known, otherwise raise error
        # For now, let's proceed cautiously, but this might need user input or metadata.
        # raise ValueError("Input layer CRS is missing. Cannot proceed.")
        
    total_bounds = layers[0].total_bounds
    minx, miny, maxx, maxy = total_bounds

    for layer in layers[1:]:
        if layer.crs != crs:
             print(f"Warning: CRS mismatch. Reprojecting layer to {crs}.")
             try:
                 layer = layer.to_crs(crs)
             except Exception as e:
                 raise ValueError(f"Failed to reproject layer to {crs}: {e}")
                 
        bounds = layer.total_bounds
        minx = min(minx, bounds[0])
        miny = min(miny, bounds[1])
        maxx = max(maxx, bounds[2])
        maxy = max(maxy, bounds[3])
        
    return minx, miny, maxx, maxy, crs

# --- Main Workflow ---
if __name__ == "__main__":
    print("\n--- Starting DEM Analysis Workflow ---")
    
    setup_output_dir(OUTPUT_DIR)
    
    # 1. Read Input Data from Geodatabase
    print(f"\n1. Reading data from: {GDB_PATH}")
    try:
        contours_gdf = gpd.read_file(GDB_PATH, layer=CONTOUR_LAYER)
        rivers_gdf = gpd.read_file(GDB_PATH, layer=RIVER_LAYER)
        lakes_gdf = gpd.read_file(GDB_PATH, layer=LAKE_LAYER)
        points_gdf = gpd.read_file(GDB_PATH, layer=POINTS_LAYER)
        
        print(f"  - Read {len(contours_gdf)} contours.")
        print(f"  - Read {len(rivers_gdf)} river segments.")
        print(f"  - Read {len(lakes_gdf)} lakes.")
        print(f"  - Read {len(points_gdf)} elevation points.")

        # --- Check CRS and Calculate Common Extent ---
        all_layers = [contours_gdf, rivers_gdf, lakes_gdf, points_gdf]
        minx, miny, maxx, maxy, common_crs = get_common_extent_and_crs(all_layers)
        print(f"  - Common CRS: {common_crs}")
        print(f"  - Combined Extent: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")

        # --- Save copies as Shapefiles (optional, good for debugging/compatibility) ---
        print("\n   Saving intermediate Shapefiles...")
        contours_gdf.to_file(os.path.join(OUTPUT_DIR, CONTOUR_SHP))
        rivers_gdf.to_file(os.path.join(OUTPUT_DIR, RIVER_SHP))
        lakes_gdf.to_file(os.path.join(OUTPUT_DIR, LAKE_SHP))
        points_gdf.to_file(os.path.join(OUTPUT_DIR, POINTS_SHP))
        print("   ...done.")

    except Exception as e:
        print(f"Error reading data: {e}")
        print("Ensure the GDAL FileGDB driver is correctly installed.")
        # Consider adding instructions or checks for the driver here
        exit()

    # --- Placeholder for next steps ---
    print("\n--- Data Reading Complete ---")
    print("Next steps: DEM Generation, Quality Assessment, Analysis.")

    # Example: Accessing contour elevation field (assuming 'HOEYDE' based on lab doc)
    if 'HOEYDE' in contours_gdf.columns:
        print(f"\nContour elevation field ('HOEYDE') found. Sample values: {contours_gdf['HOEYDE'].head().tolist()}")
        ELEV_FIELD = 'HOEYDE' # Field name from contours
    else:
        print("\nWarning: Contour elevation field 'HOEYDE' not found. Please check column names.")
        print(f"Available columns in contours: {contours_gdf.columns.tolist()}")
        # Attempt to find a likely elevation field or ask user
        potential_fields = [f for f in contours_gdf.columns if 'elev' in f.lower() or 'height' in f.lower() or 'z' in f.lower() or 'cont' in f.lower()]
        if potential_fields:
            ELEV_FIELD = potential_fields[0]
            print(f"Assuming '{ELEV_FIELD}' is the elevation field.")
        else:
            print("Error: Could not identify elevation field in contours. Exiting.")
            exit()
            
    # Example: Accessing point elevation field (assuming 'RASTERVALU' or similar based on lab doc context)
    # The lab doc mentions joining based on 'FID' and comparing DEM value to measured elevation.
    # Let's assume the measured elevation is in a field like 'RASTERVALU' or 'POINT_Z' or 'Elevation'
    potential_point_elev_fields = ['RASTERVALU', 'POINT_Z', 'Elevation', 'Z_Value', 'Value']
    POINT_ELEV_FIELD = None
    for field in potential_point_elev_fields:
        if field in points_gdf.columns:
            POINT_ELEV_FIELD = field
            print(f"\nElevation point field found: '{POINT_ELEV_FIELD}'. Sample values: {points_gdf[POINT_ELEV_FIELD].head().tolist()}")
            break
            
    if POINT_ELEV_FIELD is None:
        # Check specifically for 'HOEYDE' as it's likely correct based on output
        if 'HOEYDE' in points_gdf.columns:
             POINT_ELEV_FIELD = 'HOEYDE'
             print(f"\nElevation point field explicitly set to: '{POINT_ELEV_FIELD}'. Sample values: {points_gdf[POINT_ELEV_FIELD].head().tolist()}")
        else:
            print("\nWarning: Could not automatically identify elevation field in points layer (tried common names and 'HOEYDE').")
            print(f"Available columns in points: {points_gdf.columns.tolist()}")
            print("RMSE calculation will fail without the correct field name.")
            # Set a placeholder, but this needs correction
            POINT_ELEV_FIELD = "UNKNOWN_ELEV_FIELD" 


    # --- DEM Generation ---
    print("\n2. Generating DEMs...")

    # Define output paths relative to OUTPUT_DIR
    dem_interp_path = os.path.join(OUTPUT_DIR, DEM_INTERP_TIF)
    dem_topo_path = os.path.join(OUTPUT_DIR, DEM_TOPO_TIF)
    contour_points_path = os.path.join(OUTPUT_DIR, "contour_points.shp") # Intermediate file

    # Define input paths (using the shapefiles we saved earlier)
    contour_shp_path = os.path.join(OUTPUT_DIR, CONTOUR_SHP)
    river_shp_path = os.path.join(OUTPUT_DIR, RIVER_SHP)
    lake_shp_path = os.path.join(OUTPUT_DIR, LAKE_SHP)
    
    # --- Method 1: Interpolation (Approximation of TIN-to-Raster) ---
    print("  - Method 1: Natural Neighbour Interpolation from Contours...")
    # Intermediate files for creating points with elevation values
    contour_raster_path = os.path.join(OUTPUT_DIR, "contour_raster_temp.tif") 
    contour_points_with_value_path = os.path.join(OUTPUT_DIR, "contour_points_with_value.shp") 
    
    try:
        # Convert contour lines to raster, burning HOEYDE attribute
        print(f"    - Converting contour lines to raster ({os.path.basename(contour_raster_path)})...")
        wbt.vector_lines_to_raster(
            i=contour_shp_path,
            output=contour_raster_path,
            field=ELEV_FIELD, # Burn HOEYDE value
            nodata=-9999.0, # Assign a NoData value
            cell_size=OUTPUT_CELL_SIZE, # Use target cell size
            base=None # Let WBT determine extent from input lines
        )
        print(f"    - Contour raster saved to: {contour_raster_path}")

        # Convert contour raster back to points
        print(f"    - Converting contour raster to points ({os.path.basename(contour_points_with_value_path)})...")
        wbt.raster_to_vector_points(
            i=contour_raster_path,
            output=contour_points_with_value_path
        )
        print(f"    - Points with elevation saved to: {contour_points_with_value_path}")
        
        # Check if points file was created
        if not os.path.exists(contour_points_with_value_path):
            raise FileNotFoundError(f"Points file with values not created: {contour_points_with_value_path}")

        # Interpolate using Natural Neighbour from the generated points
        print("    - Running Natural Neighbour Interpolation...")
        # The elevation value is now likely in the 'VALUE' field of the points shapefile
        wbt.natural_neighbour_interpolation(
            i=contour_points_with_value_path, 
            field='VALUE', # Use the default field name from raster_to_vector_points
            output=dem_interp_path,
            cell_size=OUTPUT_CELL_SIZE
            # Use the calculated common extent
            # wd=OUTPUT_DIR # Removed wd argument as it's not accepted by this tool wrapper
            # extent=[minx, maxx, miny, maxy] # WBT might infer from input points
        )
        print(f"    - Interpolated (Natural Neighbour) DEM saved to: {dem_interp_path}")
        
        # Verify output exists
        if not os.path.exists(dem_interp_path):
             raise FileNotFoundError(f"Interpolated DEM file not created: {dem_interp_path}")

    except Exception as e:
        print(f"Error during Interpolation DEM generation: {e}")
        # Consider adding more specific error handling or logging
        # exit() # Optionally exit if this step fails

    # --- Method 2: TIN Gridding (Alternative to TopoToRaster/ANUDEM) ---
    # Note: TopoToRaster tool (ANUDEM equivalent) not found in this whitebox wrapper version. Using TIN Gridding instead.
    print("\n  - Method 2: TIN Gridding from Contour Points with Values...")
    try:
        # Ensure contour points with values exist from previous step
        if not os.path.exists(contour_points_with_value_path):
             raise FileNotFoundError(f"Contour points file needed for TIN Gridding not found: {contour_points_with_value_path}")

        print("    - Running TIN Gridding...")
        # Use 'i' for input and 'VALUE' for field based on previous step
        wbt.tin_gridding( 
            i=contour_points_with_value_path, # Changed 'points' to 'i'
            field='VALUE', # Use the 'VALUE' field from the generated points
            # streams=river_shp_path, # Not supported by tin_gridding
            # lakes=lake_shp_path,    # Not supported by tin_gridding
            output=dem_topo_path, # Still save as dem_topo_tif for consistency in subsequent steps
            resolution=OUTPUT_CELL_SIZE # Use 'resolution' parameter for cell size
            # base=dem_interp_path # Not supported by tin_gridding
        )
        print(f"    - TIN Gridding DEM saved to: {dem_topo_path}")
        
        # Verify output exists
        if not os.path.exists(dem_topo_path):
             raise FileNotFoundError(f"TIN Gridding DEM file not created: {dem_topo_path}")

    except Exception as e:
        print(f"Error during TIN Gridding DEM generation: {e}")
        # exit() # Optionally exit

    print("\n--- DEM Generation Complete ---")

    # --- Quality Assessment (RMSE) ---
    print("\n3. Quality Assessment (RMSE)...")
    
    # Define input paths
    points_shp_path = os.path.join(OUTPUT_DIR, POINTS_SHP)
    points_extracted_path = os.path.join(OUTPUT_DIR, POINTS_EXTRACTED_SHP)
    rmse_csv_path = os.path.join(OUTPUT_DIR, RMSE_CSV)

    # Check if required inputs exist
    if not os.path.exists(dem_interp_path):
        print(f"Error: Interpolated DEM not found ({dem_interp_path}). Skipping RMSE calculation.")
    elif not os.path.exists(dem_topo_path):
        print(f"Error: Topo DEM not found ({dem_topo_path}). Skipping RMSE calculation.")
    elif not os.path.exists(points_shp_path):
        print(f"Error: Points shapefile not found ({points_shp_path}). Skipping RMSE calculation.")
    elif POINT_ELEV_FIELD == "UNKNOWN_ELEV_FIELD":
         print(f"Error: Unknown elevation field in points layer. Skipping RMSE calculation.")
    else:
        try:
            # Copy the original points shapefile to the output path first
            print(f"  - Copying original points shapefile to {points_extracted_path}...")
            # Need to copy all parts of the shapefile (.shp, .dbf, .shx, .prj etc.)
            # Easiest way is often to read and write with geopandas
            points_orig_gdf = gpd.read_file(points_shp_path)
            points_orig_gdf.to_file(points_extracted_path, driver='ESRI Shapefile')
            print("    - Copy complete.")

            # Extract values from the Interpolated DEM (modifies points_extracted_path in place)
            print(f"  - Extracting values from Interpolated DEM ({DEM_INTERP_TIF})...")
            # Removed 'o' parameter - tool modifies the 'points' file directly
            wbt.extract_raster_values_at_points(
                inputs=dem_interp_path, 
                points=points_extracted_path # Use the copied file path
                # out_text=False # Default is shapefile output
            )
            
            # --- Robust column renaming logic ---
            # Get columns *after* first extraction
            points_extracted_gdf_after_1 = gpd.read_file(points_extracted_path)
            cols_after_1 = set(points_extracted_gdf_after_1.columns)
            # Find the new column added
            new_col_1 = list(cols_after_1 - set(points_orig_gdf.columns))
            
            if len(new_col_1) == 1:
                added_col_name_1 = new_col_1[0]
                points_extracted_gdf_after_1.rename(columns={added_col_name_1: 'DEM_Interp'}, inplace=True)
                print(f"    - Renamed extracted value column '{added_col_name_1}' to 'DEM_Interp'")
                # Save the renamed file before the next extraction
                points_extracted_gdf_after_1.to_file(points_extracted_path, driver='ESRI Shapefile')
            else:
                print(f"    - Warning: Could not uniquely identify column added by first extraction. Found: {new_col_1}")
                print(f"    - Available columns: {list(cols_after_1)}")
                # Attempt to find a likely candidate if logic failed (e.g., 'VALUE1')
                if 'VALUE1' in cols_after_1:
                     print("    - Attempting to rename 'VALUE1' to 'DEM_Interp'.")
                     points_extracted_gdf_after_1.rename(columns={'VALUE1': 'DEM_Interp'}, inplace=True)
                     points_extracted_gdf_after_1.to_file(points_extracted_path, driver='ESRI Shapefile')
                # Add more fallback logic if needed, or raise an error


            # Extract values from the TIN Gridding DEM (modifies points_extracted_path in place)
            print(f"  - Extracting values from TIN Gridding DEM ({DEM_TOPO_TIF})...")
            # Get columns *before* second extraction
            points_extracted_gdf_before_2 = gpd.read_file(points_extracted_path) # Reload after potential rename
            cols_before_2 = set(points_extracted_gdf_before_2.columns)

            # Removed 'o' parameter - tool modifies the 'points' file directly
            wbt.extract_raster_values_at_points(
                inputs=dem_topo_path, 
                points=points_extracted_path # Use the file already modified by the previous step
                # out_text=False
            )

            # --- Robust column renaming logic for second extraction ---
            points_extracted_gdf_after_2 = gpd.read_file(points_extracted_path)
            cols_after_2 = set(points_extracted_gdf_after_2.columns)
            new_col_2 = list(cols_after_2 - cols_before_2)

            if len(new_col_2) == 1:
                added_col_name_2 = new_col_2[0]
                points_extracted_gdf_after_2.rename(columns={added_col_name_2: 'DEM_Topo'}, inplace=True)
                print(f"    - Renamed second extracted value column '{added_col_name_2}' to 'DEM_Topo'")
            else:
                print(f"    - Warning: Could not uniquely identify column added by second extraction. Found: {new_col_2}")
                print(f"    - Available columns: {list(cols_after_2)}")
                 # Attempt to find a likely candidate if logic failed (e.g., 'VALUE1' if it reappeared)
                if 'VALUE1' in new_col_2: # Check if 'VALUE1' is the *new* column
                     print("    - Attempting to rename 'VALUE1' to 'DEM_Topo'.")
                     points_extracted_gdf_after_2.rename(columns={'VALUE1': 'DEM_Topo'}, inplace=True)
                elif 'VALUE1' in cols_after_2 and 'DEM_Topo' not in cols_after_2: # Check if 'VALUE1' exists but wasn't the new one
                     print("    - Fallback: Renaming existing 'VALUE1' to 'DEM_Topo'.")
                     points_extracted_gdf_after_2.rename(columns={'VALUE1': 'DEM_Topo'}, inplace=True)


            # Final reload for RMSE calculation
            points_extracted_gdf = points_extracted_gdf_after_2


            print(f"  - Extracted values saved to: {points_extracted_path}")

            # Calculate RMSE
            print("  - Calculating RMSE...")
            # Ensure columns exist before calculation
            required_cols = [POINT_ELEV_FIELD, 'DEM_Interp', 'DEM_Topo']
            if all(col in points_extracted_gdf.columns for col in required_cols):
                # Drop rows where DEM extraction might have failed (NoData values)
                points_extracted_gdf.replace([np.inf, -np.inf], np.nan, inplace=True)
                points_valid_gdf = points_extracted_gdf.dropna(subset=required_cols)
                
                n_points = len(points_valid_gdf)
                if n_points > 0:
                    measured = points_valid_gdf[POINT_ELEV_FIELD]
                    interp_dem = points_valid_gdf['DEM_Interp']
                    topo_dem = points_valid_gdf['DEM_Topo']

                    rmse_interp = sqrt(mean_squared_error(measured, interp_dem))
                    rmse_topo = sqrt(mean_squared_error(measured, topo_dem))

                    print(f"    - RMSE (Interpolated DEM vs Points): {rmse_interp:.3f} (using {n_points} points)")
                    print(f"    - RMSE (Topo DEM vs Points): {rmse_topo:.3f} (using {n_points} points)")

                    # Save results to CSV
                    rmse_data = {'DEM_Type': ['Interpolated (Natural Neighbour)', 'TIN Gridding'],
                                 'RMSE': [rmse_interp, rmse_topo],
                                 'N_Points': [n_points, n_points]}
                    rmse_df = pd.DataFrame(rmse_data)
                    rmse_df.to_csv(rmse_csv_path, index=False)
                    print(f"    - RMSE results saved to: {rmse_csv_path}")
                else:
                    print("    - Error: No valid points found after dropping NaN/Inf values. Cannot calculate RMSE.")

            else:
                print(f"    - Error: Missing one or more required columns for RMSE calculation: {required_cols}")
                print(f"    - Available columns: {points_extracted_gdf.columns.tolist()}")


        except Exception as e:
            print(f"Error during Quality Assessment: {e}")

    print("\n--- Quality Assessment Complete ---")

    # --- Further Analysis ---
    print("\n4. Further Analysis (Contours, Hillshade, Difference, Slope)...")

    # Define output paths
    contours_interp_path = os.path.join(OUTPUT_DIR, CONTOURS_INTERP_SHP)
    contours_topo_path = os.path.join(OUTPUT_DIR, CONTOURS_TOPO_SHP)
    hillshade_interp_path = os.path.join(OUTPUT_DIR, HILLSHADE_INTERP_TIF)
    hillshade_topo_path = os.path.join(OUTPUT_DIR, HILLSHADE_TOPO_TIF)
    dem_diff_path = os.path.join(OUTPUT_DIR, DEM_DIFF_TIF)
    slope_interp_path = os.path.join(OUTPUT_DIR, SLOPE_INTERP_TIF)
    slope_topo_path = os.path.join(OUTPUT_DIR, SLOPE_TOPO_TIF)
    
    # Set contour interval (e.g., 10m as mentioned in lab, or adjust as needed)
    CONTOUR_INTERVAL = 10.0 

    # Check if DEMs exist before proceeding
    if not os.path.exists(dem_interp_path) or not os.path.exists(dem_topo_path):
        print("Error: One or both DEM files not found. Skipping further analysis.")
    else:
        try:
            # a) Generate Contours
            print(f"  - Generating contours (interval: {CONTOUR_INTERVAL}m)...")
            wbt.contours_from_raster(
                i=dem_interp_path,
                output=contours_interp_path,
                interval=CONTOUR_INTERVAL
            )
            print(f"    - Contours from Interpolated DEM saved to: {contours_interp_path}")
            wbt.contours_from_raster(
                i=dem_topo_path,
                output=contours_topo_path,
                interval=CONTOUR_INTERVAL
            )
            print(f"    - Contours from Topo DEM saved to: {contours_topo_path}")

            # b) Generate Hillshades
            print("  - Generating hillshades...")
            wbt.hillshade(
                dem=dem_interp_path,
                output=hillshade_interp_path
                # Default azimuth=315, altitude=30
            )
            print(f"    - Hillshade from Interpolated DEM saved to: {hillshade_interp_path}")
            wbt.hillshade(
                dem=dem_topo_path,
                output=hillshade_topo_path
            )
            print(f"    - Hillshade from Topo DEM saved to: {hillshade_topo_path}")

            # c) Calculate DEM Difference
            print("  - Calculating DEM difference (Topo - Interpolated)...")
            wbt.subtract(
                input1=dem_topo_path,
                input2=dem_interp_path,
                output=dem_diff_path
            )
            print(f"    - DEM difference map saved to: {dem_diff_path}")

            # d) Calculate Slope
            print("  - Calculating slope...")
            wbt.slope(
                dem=dem_interp_path,
                output=slope_interp_path
                # Output units: degrees (default)
            )
            print(f"    - Slope from Interpolated DEM saved to: {slope_interp_path}")
            wbt.slope(
                dem=dem_topo_path,
                output=slope_topo_path
            )
            print(f"    - Slope from Topo DEM saved to: {slope_topo_path}")

        except Exception as e:
            print(f"Error during Further Analysis: {e}")

    print("\n--- Further Analysis Complete ---")
    print("\n--- Workflow Script Finished ---")
