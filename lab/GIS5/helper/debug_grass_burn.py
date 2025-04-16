import os
from pathlib import Path
import sys
import tempfile
import traceback

os.environ["GRASSBIN"] = "/Applications/GRASS-8.4.app/Contents/MacOS/Grass.sh"

# Import GRASS libraries
from grass_session import Session
import grass.script as gs
import grass.script.setup as gsetup
import subprocess

# --- Configuration ---
# Assuming a previous run generated these in output_py
# Adjust if your output directory name is different
OUTPUT_DIR_NAME = "output_py"  # Or the specific timestamped dir if needed
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / OUTPUT_DIR_NAME

# Input files (relative to OUTPUT_DIR)
DEM_TOPO_FILENAME = "dem_topo_to_raster.tif"
RIVER_SHP_FILENAME = "rivers.shp"

# Output file (use a distinct name for debugging)
DEM_BURNED_FILENAME = "dem_stream_burned_debug.tif"

# GRASS Parameters
# GRASS_EXECUTABLE = "/Applications/GRASS-8.4.app/Contents/MacOS/Grass.sh" # No longer needed directly
BURN_VALUE = -10.0
# TIMEOUT_SECONDS = 600 # Timeout handled differently or less relevant

# --- Paths ---
dem_topo_path = OUTPUT_DIR / DEM_TOPO_FILENAME
river_shp_path = OUTPUT_DIR / RIVER_SHP_FILENAME
dem_burned_path = OUTPUT_DIR / DEM_BURNED_FILENAME

# --- GRASS Environment Setup ---
# Try to find GRASS GIS installation path automatically
# You might need to set GISBASE environment variable if this fails
gisbase = os.environ.get("GISBASE")
if not gisbase:
    # Attempt common macOS path
    potential_gisbase = (
        "/Applications/GRASS-8.4.app/Contents/Resources"  # Adjust version if needed
    )
    if Path(potential_gisbase).exists():
        gisbase = potential_gisbase
        print(f"Found potential GISBASE at: {gisbase}")
        os.environ["GISBASE"] = gisbase
    else:
        print("ERROR: GISBASE environment variable not set and common path not found.")
        print(
            "Please set GISBASE to your GRASS installation directory (e.g., /Applications/GRASS-8.4.app/Contents/Resources)"
        )
        sys.exit(1)

# Define GRASS Database, Location, Mapset (temporary)
gisdb = Path(tempfile.mkdtemp(prefix="grass_debug_db_"))
location_name = "debug_stream_burn_loc"
mapset_name = "PERMANENT"  # Use PERMANENT for session creation
grass_location_path = gisdb / location_name


def main():
    print("--- Starting GRASS Stream Burn Debug Script (using grass_session) ---")
    print(f"Using TIN DEM: {dem_topo_path}")
    print(f"Using River Shapefile: {river_shp_path}")
    print(f"Output Burned DEM: {dem_burned_path}")
    print(f"Burn Value: {BURN_VALUE}")
    print(f"Using temporary GISDBASE: {gisdb}")
    print(f"Target Location: {grass_location_path}")

    # --- Input Checks ---
    if not dem_topo_path.exists():
        print(f"ERROR: Input TIN DEM not found: {dem_topo_path}")
        print("Ensure the main workflow has run at least up to TIN gridding.")
        sys.exit(1)
    if not river_shp_path.exists():
        print(f"ERROR: Input river shapefile not found: {river_shp_path}")
        print("Ensure the main workflow has run the data loading step.")
        sys.exit(1)

    # --- GRASS Processing with Session ---
    try:
        # --- Workaround: Create Location Manually First ---
        grass_executable = os.environ.get("GRASSBIN")  # Get from env var set earlier
        if not grass_executable or not Path(grass_executable).exists():
            # Fallback or error if GRASSBIN wasn't set/found
            grass_executable = (
                "/Applications/GRASS-8.4.app/Contents/MacOS/Grass.sh"  # Default guess
            )
            if not Path(grass_executable).exists():
                print(
                    f"ERROR: GRASS executable not found via GRASSBIN or default path ({grass_executable})."
                )
                sys.exit(1)
            else:
                print(
                    f"Warning: GRASSBIN env var not found, using default: {grass_executable}"
                )

        print(
            f"\nManually creating temporary GRASS location '{location_name}' at: {grass_location_path}"
        )
        create_loc_cmd = [
            grass_executable,
            # "-text",        # Removed: Not needed/accepted with -c -e
            "-c",
            str(dem_topo_path),  # Create from this raster
            str(grass_location_path),  # Full path to the new location directory
            "-e",  # Exit after creation
        ]
        print(f"  Running: {' '.join(create_loc_cmd)}")
        create_result = subprocess.run(
            create_loc_cmd, check=True, capture_output=True, text=True, timeout=60
        )
        print(f"    GRASS Location Creation stdout:\n{create_result.stdout.strip()}")
        if create_result.stderr:
            print(
                f"    GRASS Location Creation stderr:\n{create_result.stderr.strip()}"
            )
        print("Location created successfully.")
        # --- End Workaround ---

        # Now start the session using the existing location
        print(f"\nStarting GRASS session in existing location '{location_name}'...")
        with Session(
            gisdb=str(gisdb), location=location_name, mapset=mapset_name
        ):  # No create_opts needed now
            print("GRASS session started successfully.")

            # Import the base DEM into the current mapset (if not already done by location creation)
            # Check if 'tin_dem' exists first to avoid unnecessary import/overwrite
            existing_rasters = gs.list_strings(type="raster", mapset=".")
            if "tin_dem" not in existing_rasters:
                print("\nImporting base DEM (tin_dem)...")
                gs.run_command(
                    "r.in.gdal",
                    input=str(dem_topo_path),
                    output="tin_dem",
                    flags="o",  # Add -o flag to override projection check
                    overwrite=True,  # Overwrite just in case, though check prevents it mostly
                    quiet=False,  # Show output
                    verbose=True,
                )
            else:
                print("\nBase DEM 'tin_dem' already exists in mapset.")

            # Set computational region based on imported DEM
            # print("\nImporting base DEM (tin_dem)...") # Removed redundant import
            # gs.run_command('r.in.gdal',
            #                input=str(dem_topo_path),
            #                output='tin_dem',
            #                overwrite=True,
            #                quiet=False, # Show output
            #                verbose=True)

            # Set computational region based on imported DEM
            print("\nSetting region...")
            gs.run_command("g.region", raster="tin_dem", flags="p", verbose=True)

            # Import Rivers
            print("\nImporting rivers...")
            gs.run_command(
                "v.in.ogr",
                input=str(river_shp_path),
                output="rivers",
                snap=0.0001,
                overwrite=True,
                flags="o",  # Override projection check (use location's)
                verbose=True,
            )

            # Rasterize rivers
            print("\nRasterizing rivers...")
            gs.run_command(
                "v.to.rast",
                input="rivers",
                output="rivers_rast",
                use="val",
                value=1,
                overwrite=True,
                verbose=True,
            )

            # --- Stream Extraction and Burning using r.stream.extract + r.mapcalc ---

            # 1. Extract streams using r.stream.extract
            print("\nExtracting streams using r.stream.extract...")
            stream_extract_threshold = 1  # User specified threshold
            print(f"Using threshold: {stream_extract_threshold}")
            gs.run_command(
                "r.stream.extract",
                elevation="tin_dem",
                threshold=stream_extract_threshold,
                stream_raster="streams_extracted",  # Output stream raster
                # accumulation='accum', # Optional: calculate accumulation first if needed
                # direction='flowdir', # Optional: output flow direction
                overwrite=True,
                verbose=True,
            )

            # 2. Burn streams into DEM using r.mapcalc
            print("\nBurning extracted streams into DEM using r.mapcalc...")
            mapcalc_expression = f"dem_burned = if(isnull(streams_extracted), tin_dem, tin_dem + ({BURN_VALUE}))"
            print(f"Mapcalc expression: {mapcalc_expression}")
            # Use run_command for r.mapcalc to ensure map creation
            gs.run_command(
                "r.mapcalc", expression=mapcalc_expression, overwrite=True, verbose=True
            )

            # Check if burned map exists
            print("\nChecking for dem_burned map using find_file...")
            try:
                # find_file checks for the existence of map elements
                map_info = gs.find_file(
                    name="dem_burned", element="cell"
                )  # Check for raster cell data
                print(f"Found dem_burned map info: {map_info}")
                # Check if the returned dictionary is meaningful (has a name)
                if not map_info or not map_info.get("name"):
                    raise RuntimeError(
                        "ERROR: dem_burned map not found by find_file after r.mapcalc (empty info)"
                    )
            except Exception as find_err:
                # Re-raise or create a specific error, including the original error message
                raise RuntimeError(
                    f"ERROR: dem_burned map not found by find_file after r.mapcalc. Error: {find_err}"
                )
            print("dem_burned map found.")

            # Export Burned DEM
            print("\nExporting burned DEM...")
            gs.run_command(
                "r.out.gdal",
                input="dem_burned",
                output=str(dem_burned_path),
                format="GTiff",
                type="Float32",  # Revert to Float32 as originally intended
                flags="f",  # Add -f flag to force export despite precision loss
                overwrite=True,
                verbose=True,
            )

            print("\n--- GRASS Processing within Session Finished Successfully ---")

        # Session context manager automatically cleans up the session/mapset
        print(f"\nGRASS session for {location_name} closed.")

        # --- Verification ---
        print(f"\nVerifying final output file: {dem_burned_path}")
        if not dem_burned_path.exists():
            # This shouldn't happen if r.out.gdal succeeded, but good practice
            raise FileNotFoundError(
                f"Stream-burned DEM file was NOT created despite apparent success: {dem_burned_path}"
            )
        else:
            print("SUCCESS: Stream-burned DEM file created successfully.")

    except Exception as e:
        print(f"\n--- An error occurred during GRASS processing ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        # Optional: Add specific GRASS error parsing if needed
        if hasattr(e, "stderr"):
            print(f"GRASS stderr:\n{e.stderr}")

    finally:
        # Clean up the entire temporary GISDBASE directory
        if gisdb.exists():
            import shutil

            print(f"\nCleaning up temporary GRASS GISDBASE: {gisdb}")
            try:
                shutil.rmtree(gisdb)
            except OSError as rm_err:
                print(f"Warning: Could not remove temporary GISDBASE {gisdb}: {rm_err}")

    print("\n--- GRASS Stream Burn Debug Script Finished ---")


if __name__ == "__main__":
    # Setup GRASS environment for the Python script process
    # This needs to happen *before* Session is initialized if GISBASE isn't already set globally
    # We already did this near the top, but double-checking doesn't hurt.
    if "GISBASE" not in os.environ:
        print("ERROR: GISBASE not set. Exiting.")
        sys.exit(1)

    # Add GRASS scripts and binaries to PATH if not already there
    # This helps grass.script find necessary components
    gisbase_bin = Path(gisbase) / "bin"
    gisbase_scripts = Path(gisbase) / "scripts"
    os.environ["PATH"] = f"{gisbase_bin}:{gisbase_scripts}:{os.environ.get('PATH', '')}"
    # For macOS, DYLD_LIBRARY_PATH might be needed for libraries
    gisbase_lib = Path(gisbase) / "lib"
    os.environ["DYLD_LIBRARY_PATH"] = (
        f"{gisbase_lib}:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
    )

    # Initialize the GRASS script environment
    # gsetup.init(gisbase) # Often not needed when using Session, but can resolve some path issues

    main()
