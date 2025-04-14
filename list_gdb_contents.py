import fiona
import sys
import os

gdb_path = os.path.abspath("lab/GIS5/GIS5_datafiles/DEM_analysis_DATA.gdb")

try:
    print(f"Attempting to list layers in: {gdb_path}")
    if not os.path.exists(gdb_path):
        print(f"Error: Geodatabase not found at {gdb_path}", file=sys.stderr)
        sys.exit(1)
        
    layers = fiona.listlayers(gdb_path)
    print("Available layers:")
    for layer in layers:
        print(f"- {layer}")
except Exception as e:
    print(f"Error accessing Geodatabase: {e}", file=sys.stderr)
    # Also print GDAL driver info if possible
    try:
        import geopandas
        print("\nAttempting to get Fiona/GDAL driver info:")
        supported_drivers = fiona.supported_drivers
        print(f"Supported Fiona drivers: {supported_drivers}")
        if 'OpenFileGDB' in supported_drivers:
             print("OpenFileGDB driver is available.")
        else:
             print("OpenFileGDB driver is NOT available.")
             print("Reading .gdb files might fail. Consider exporting layers to GeoPackage or Shapefile.")
    except ImportError:
        print("Could not import geopandas to check driver info.")
    except Exception as e_info:
        print(f"Error getting driver info: {e_info}")
    sys.exit(1)
