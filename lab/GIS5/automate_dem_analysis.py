import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
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
        print("\nWarning: Could not automatically identify elevation field in points layer.")
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
    try:
        # Convert contour lines to points for interpolation
        print("    - Converting contours to points...")
        wbt.vector_lines_to_points(
            i=contour_shp_path, 
            output=contour_points_path, 
            densify=True # Add points along lines
        )
        print(f"    - Contour points saved to: {contour_points_path}")

        # Interpolate using Natural Neighbour
        print("    - Running Natural Neighbour Interpolation...")
        wbt.natural_neighbour_interpolation(
            i=contour_points_path,
            field=ELEV_FIELD,
            output=dem_interp_path,
            cell_size=OUTPUT_CELL_SIZE,
            # Use the calculated common extent
            wd=OUTPUT_DIR # Set working directory for relative paths if needed
            # extent=[minx, maxx, miny, maxy] # WBT might infer from input points
        )
        print(f"    - Interpolated DEM saved to: {dem_interp_path}")
        
        # Verify output exists
        if not os.path.exists(dem_interp_path):
             raise FileNotFoundError(f"Interpolated DEM file not created: {dem_interp_path}")

    except Exception as e:
        print(f"Error during Interpolation DEM generation: {e}")
        # Consider adding more specific error handling or logging
        # exit() # Optionally exit if this step fails

    # --- Method 2: Topo to Raster (ANUDEM) ---
    print("\n  - Method 2: Topo to Raster (ANUDEM)...")
    try:
        # Ensure contour points exist from previous step
        if not os.path.exists(contour_points_path):
             raise FileNotFoundError(f"Contour points file needed for TopoToRaster not found: {contour_points_path}")

        print("    - Running Topo to Raster...")
        wbt.topo_to_raster(
            points=contour_points_path,
            field=ELEV_FIELD,
            streams=river_shp_path, # Input rivers as streams
            lakes=lake_shp_path,    # Input lakes
            output=dem_topo_path,
            cell_size=OUTPUT_CELL_SIZE,
            # Ensure same extent as the first DEM for comparison
            # WBT TopoToRaster might not directly take extent, but uses input features.
            # We might need to clip/resample later if extents differ significantly.
            # For now, rely on WBT's extent handling based on inputs.
            # Alternatively, set a base raster:
            base=dem_interp_path, # Use the first DEM to define grid properties
            wd=OUTPUT_DIR
        )
        print(f"    - Topo-to-Raster DEM saved to: {dem_topo_path}")
        
        # Verify output exists
        if not os.path.exists(dem_topo_path):
             raise FileNotFoundError(f"Topo-to-Raster DEM file not created: {dem_topo_path}")

    except Exception as e:
        print(f"Error during Topo to Raster DEM generation: {e}")
        # exit() # Optionally exit

    print("\n--- DEM Generation Complete ---")

    # --- Quality Assessment (to be added) ---

    # --- Further Analysis (to be added) ---

    print("\n--- Workflow Script Updated (DEM Gen Added) ---")
