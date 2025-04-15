import os
import shutil
import tempfile
from pathlib import Path
from whitebox import WhiteboxTools
from typing import Optional

# Import config first to get GRASS path
from src.config import settings, AppConfig

# Ensure grass.sh is added to GRASSBIN env var
os.environ['GRASSBIN'] = "/Applications/GRASS-8.4.app/Contents/MacOS/Grass.sh"

# Now import GRASS libraries
from grass_session import Session
import grass.script as gs


def generate_dems(
    settings: AppConfig,
    wbt: WhiteboxTools,
    contour_shp_path: Path,
    contour_elev_field: str,
    river_shp_path: Optional[Path] = None, # Add river path
    stream_extract_threshold: Optional[int] = None # Add threshold for r.stream.extract
):
    """
    Generates DEM rasters using interpolation, TIN gridding, and optionally stream burning
    using GRASS r.stream.extract and r.mapcalc.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        contour_shp_path: Path to the contour shapefile.
        contour_elev_field: Name of the elevation field in the contour shapefile.
        river_shp_path: Optional path to the river shapefile.
        stream_extract_threshold: Optional threshold value for r.stream.extract.

    Raises:
        FileNotFoundError: If required intermediate files are missing or outputs are not created.
        RuntimeError: If GRASS executable is not found or stream burning fails.
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


    # --- Method 3: Stream Extraction and Burning (Optional, using GRASS) ---
    if settings.processing.enable_stream_burning:
        print("\n  - Method 3: Stream Extraction and Burning with GRASS...")
        # Note: river_shp_path is imported but not directly used for burning in this method
        # It might be useful for comparison or other steps later.
        # if not river_shp_path or not river_shp_path.exists():
        #     print(f"    - Warning: River shapefile not provided or not found at {river_shp_path}, but not strictly needed for r.stream.extract.")
            # Decide if this should be an error or just a warning

        if not stream_extract_threshold:
             raise ValueError("Stream burning enabled, but 'stream_extract_threshold' not provided in settings.")

        if not settings.paths.grass_executable_path or not Path(settings.paths.grass_executable_path).exists():
             raise RuntimeError(f"GRASS executable not found at '{settings.paths.grass_executable_path}'. Please check config.")

        if not dem_topo_path.exists():
            raise FileNotFoundError(f"Input DEM for stream burning not found: {dem_topo_path}")

        dem_burned_path = output_files.get_full_path('dem_stream_burned_tif', output_dir)
        burn_value = settings.processing.stream_burn_value # Get burn value from settings

        # Create a temporary directory for the GRASS database/location
        temp_grass_db_path = Path(tempfile.mkdtemp(prefix="grass_stream_extract_"))
        location_name = "temp_stream_burn_loc"
        mapset_name = "PERMANENT" # Standard mapset name

        # Calculate GISBASE directly from the executable path in settings
        grass_executable_path_obj = Path(settings.paths.grass_executable_path)
        gisbase = str(grass_executable_path_obj.parent.parent) # e.g., /Applications/GRASS-8.4.app/Contents
        print(f"    - Calculated GISBASE: {gisbase}")
        if not Path(gisbase).is_dir():
             raise RuntimeError(f"Calculated GISBASE directory does not exist: {gisbase}")

        # Add GRASS scripts and binaries to PATH/DYLD_LIBRARY_PATH for the session
        # This helps grass.script find necessary components, especially if not globally set
        print("    - Setting GRASS environment variables for session...")
        gisbase_bin = Path(gisbase) / 'bin'
        gisbase_scripts = Path(gisbase) / 'scripts'
        original_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{gisbase_bin}:{gisbase_scripts}:{original_path}"
        # For macOS, DYLD_LIBRARY_PATH might be needed for libraries
        gisbase_lib = Path(gisbase) / 'lib'
        original_dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
        os.environ['DYLD_LIBRARY_PATH'] = f"{gisbase_lib}:{original_dyld_path}"
        print(f"      - Updated DYLD_LIBRARY_PATH: {os.environ['DYLD_LIBRARY_PATH']}")

        # Set GISBASE environment variable for the session
        original_gisbase = os.environ.get('GISBASE') # Store original value if exists
        os.environ['GISBASE'] = gisbase
        print(f"      - Set GISBASE environment variable: {os.environ['GISBASE']}")

        # --- Manually Create GRASS Location ---
        grass_executable = os.environ.get("GRASSBIN") # Get from env var set at top
        if not grass_executable or not Path(grass_executable).exists():
             raise RuntimeError(f"GRASS executable not found via GRASSBIN env var: {grass_executable}")

        grass_location_path = temp_grass_db_path / location_name
        print(f"\n    - Manually creating temporary GRASS location '{location_name}' at: {grass_location_path}")
        create_loc_cmd = [
            grass_executable,
            "-c", str(dem_topo_path), # Create from this raster
            str(grass_location_path), # Full path to the new location directory
            "-e"              # Exit after creation
        ]
        print(f"      Running: {' '.join(create_loc_cmd)}")
        try:
            import subprocess
            create_result = subprocess.run(create_loc_cmd, check=True, capture_output=True, text=True, timeout=60)
            print(f"        GRASS Location Creation stdout:\n{create_result.stdout.strip()}")
            if create_result.stderr:
                # GRASS often prints info to stderr even on success
                print(f"        GRASS Location Creation stderr:\n{create_result.stderr.strip()}")
            print("      Location created successfully.")
        except subprocess.CalledProcessError as cpe:
             print(f"ERROR: Failed to create GRASS location manually.")
             print(f"  Command: {' '.join(cpe.cmd)}")
             print(f"  Return Code: {cpe.returncode}")
             print(f"  Stdout: {cpe.stdout}")
             print(f"  Stderr: {cpe.stderr}")
             raise RuntimeError("Manual GRASS location creation failed.") from cpe
        except Exception as e:
             raise RuntimeError(f"An unexpected error occurred during manual GRASS location creation: {e}") from e
        # --- End Manual Creation ---


        # Use a context manager for the GRASS session, connecting to existing location
        print(f"\n    - Starting GRASS session in existing location '{location_name}'...")
        try:
            # Connect to the manually created location
            with Session(gisdb=str(temp_grass_db_path),
                         location=location_name,
                         mapset=mapset_name): # No create_opts needed

                print("    - GRASS session started. Running commands...")

                # Explicitly import the base DEM into the current mapset
                # Use a consistent internal name like 'tin_dem'
                imported_raster_name = 'tin_dem'
                print(f"\n      - Importing base DEM as '{imported_raster_name}'...")
                gs.run_command('r.in.gdal',
                               input=str(dem_topo_path),
                               output=imported_raster_name,
                               flags='o', # Add -o flag to override projection check if needed
                               overwrite=True,
                               quiet=False,
                               verbose=True)

                # Set computational region based on the imported DEM
                print(f"\n      - Setting region based on '{imported_raster_name}'...")
                gs.run_command('g.region', raster=imported_raster_name, flags='p', verbose=True)

                # 1. Extract streams using r.stream.extract
                print(f"      - Extracting streams using r.stream.extract (threshold={stream_extract_threshold})...")
                gs.run_command('r.stream.extract',
                               elevation=imported_raster_name,
                               threshold=stream_extract_threshold,
                               stream_raster='streams_extracted', # Output stream raster
                               overwrite=True,
                               verbose=True)

                # 2. Burn streams into DEM using r.mapcalc
                print(f"      - Burning extracted streams into DEM using r.mapcalc (burn value={burn_value})...")
                mapcalc_expression = f"dem_burned = if(isnull(streams_extracted), {imported_raster_name}, {imported_raster_name} + ({burn_value}))"
                print(f"        - Mapcalc expression: {mapcalc_expression}")
                gs.run_command('r.mapcalc',
                               expression=mapcalc_expression,
                               overwrite=True,
                               verbose=True)

                # Check if burned map exists (optional but good practice)
                print("      - Checking for dem_burned map...")
                # map_info = gs.read_command('g.list', type='raster', mapset='.', pattern='dem_burned')
                try:
                    map_info = gs.find_file(name='dem_burned', element='cell')
                    if not map_info or not map_info.get('name'):
                        raise RuntimeError("Map 'dem_burned' not found by find_file after r.mapcalc.")
                    print("        - dem_burned map found.")
                except Exception as find_err:
                    raise RuntimeError(f"ERROR: dem_burned map not found by find_file after r.mapcalc. Error: {find_err}")


                # Export Burned DEM directly as Float32 with GeoTIFF profile
                print(f"      - Exporting burned DEM as Float32 to: {dem_burned_path}...")
                gs.run_command('r.out.gdal',
                               input='dem_burned',
                               output=str(dem_burned_path),
                               format='GTiff',
                               type='Float64', 
                               createopt="PROFILE=GeoTIFF,TFW=YES", # Use BASELINE profile and create World File
                               flags='f',      # Add -f flag to force export if precision issues arise
                               overwrite=True,
                               verbose=True)

                print(f"    - Stream-extracted and burned DEM saved to: {dem_burned_path}")

            # Session context manager handles cleanup of the session environment
            print("    - GRASS session closed.")

        except Exception as e:
             print(f"Error during GRASS session or command execution: {e}")
             # grass.script often raises CalledModuleError which includes details
             if hasattr(e, 'stderr'):
                 print(f"GRASS stderr:\n{e.stderr}")
             raise RuntimeError("GRASS stream extraction/burning failed within session.") from e
        finally:
            # Restore original environment variables
            os.environ['PATH'] = original_path
            os.environ['DYLD_LIBRARY_PATH'] = original_dyld_path
            # Restore original GISBASE if it existed, otherwise remove it
            if original_gisbase:
                os.environ['GISBASE'] = original_gisbase
            elif 'GISBASE' in os.environ:
                del os.environ['GISBASE']
            print("    - Restored original environment variables.")


        # Final check for the output file
        if not dem_burned_path.exists():
            raise FileNotFoundError(f"Stream-burned DEM file was not created by GRASS session: {dem_burned_path}")
        else:
             print(f"    - Verified output file exists: {dem_burned_path}")


    # Cleanup of the temporary GRASS Database directory itself
    if 'temp_grass_db_path' in locals() and temp_grass_db_path.exists(): # Check if var exists before cleanup
        print(f"    - Cleaning up temporary GRASS database directory: {temp_grass_db_path}")
        # Be cautious with rmtree; ensure it's the correct path
        try:
            shutil.rmtree(temp_grass_db_path)
        except OSError as e:
            print(f"Warning: Could not remove temporary GRASS directory {temp_grass_db_path}: {e}")


    print("--- DEM Generation and Stream Extraction/Burning (if enabled) Complete ---")




# Example usage for testing this module directly
if __name__ == "__main__":
    from src.config import settings # Import the instantiated settings
    from whitebox import WhiteboxTools

    print("--- Testing generate_dems.py ---")

    # Ensure output directory exists (config should handle this, but double-check)
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WhiteboxTools
    try:
        wbt = WhiteboxTools()
        wbt.set_working_dir(str(settings.paths.output_dir)) # Set WBT working dir
        print(f"WhiteboxTools initialized. Working directory: {wbt.get_working_dir()}")
    except Exception as e:
        print(f"Error initializing WhiteboxTools: {e}")
        exit()

    # Define required inputs (assuming load_data has run previously or files exist)
    # These paths are relative to the output directory where intermediate files are saved
    contour_shp_test_path = settings.output_files.get_full_path('contour_shp', settings.paths.output_dir)
    river_shp_test_path = settings.output_files.get_full_path('river_shp', settings.paths.output_dir)
    contour_elev_field_test = settings.input_layers.contour_elevation_field

    # Check if required input files exist for the test
    if not contour_shp_test_path.exists():
        print(f"ERROR: Contour shapefile for testing not found at {contour_shp_test_path}")
        print("Please run the 'load_data' step first (e.g., via the main workflow) to generate intermediate files.")
        exit()
    if settings.processing.enable_stream_burning and not river_shp_test_path.exists():
         print(f"ERROR: River shapefile for stream burning test not found at {river_shp_test_path}")
         print("Please run the 'load_data' step first (e.g., via the main workflow) to generate intermediate files.")
         exit()


    print(f"Using Contour Shapefile: {contour_shp_test_path}")
    print(f"Using Contour Elevation Field: {contour_elev_field_test}")
    if settings.processing.enable_stream_burning:
        print(f"Using River Shapefile: {river_shp_test_path}")
        print(f"Stream Burning Enabled: True (Value: {settings.processing.stream_burn_value})")
        print(f"GRASS Executable: {settings.paths.grass_executable_path}")
    else:
        print("Stream Burning Enabled: False")


    try:
        generate_dems(
            settings=settings,
            wbt=wbt,
            contour_shp_path=contour_shp_test_path,
            contour_elev_field=contour_elev_field_test,
            river_shp_path=river_shp_test_path, # Pass river path regardless (might be needed elsewhere)
            stream_extract_threshold=settings.processing.stream_extract_threshold if settings.processing.enable_stream_burning else None
        )
        print("\n--- generate_dems.py test completed successfully ---")
    except Exception as e:
        print(f"\n--- generate_dems.py test failed: {e} ---")
        import traceback
        traceback.print_exc()
