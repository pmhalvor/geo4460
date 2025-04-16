import sys
from whitebox import WhiteboxTools
import time

# Ensure the src directory is in the Python path if running as a script
# This might not be needed if run as a module or with `python -m lab.GIS5.src.workflow`
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.config import settings
    from src.utils import setup_output_dir
    from src.tasks.load_data import load_and_prepare_data
    from src.tasks.generate_dems import generate_dems
    from src.tasks.quality_assessment import assess_dem_quality
    from src.tasks.derive_products import generate_derived_products
except ImportError:
    # Fallback for running the script directly from the src directory
    from config import settings
    from utils import setup_output_dir
    from tasks.load_data import load_and_prepare_data
    from tasks.generate_dems import generate_dems
    from tasks.quality_assessment import assess_dem_quality
    from tasks.derive_products import generate_derived_products


def main():
    """Runs the complete DEM analysis workflow."""
    start_time = time.time()
    print("\n--- Starting DEM Analysis Workflow ---")

    # --- Initialization ---
    print("Initializing WhiteboxTools...")
    try:
        wbt = WhiteboxTools()
        print(f"  - WhiteboxTools version: {wbt.version()}")
        # Optional: Set working directory if needed, though outputs are explicitly pathed
        # wbt.work_dir = str(settings.paths.output_dir / "wbt_temp")
        # print(f"  - WhiteboxTools working directory set to: {wbt.work_dir}")
        wbt.set_verbose_mode(settings.processing.wbt_verbose)
    except Exception as e:
        print(f"Error initializing WhiteboxTools: {e}")
        print(
            "Ensure WhiteboxTools executable is correctly installed and in the system PATH."
        )
        sys.exit(1)

    # --- Setup Output Directory ---
    # Config handles creation, but setup_output_dir ensures it's clean
    setup_output_dir(settings.paths.output_dir)

    try:
        # --- Task 1: Load Data ---
        loaded_gdfs, common_extent, common_crs, contour_elev_field, point_elev_field = (
            load_and_prepare_data(settings)
        )

        # Get necessary paths for subsequent tasks
        output_dir = settings.paths.output_dir
        output_files = settings.output_files
        contour_shp_path = output_files.get_full_path("contour_shp", output_dir)
        points_shp_path = output_files.get_full_path("points_shp", output_dir)
        # Get the path to the river shapefile saved by load_data
        river_shp_path = output_files.get_full_path("river_shp", output_dir)

        # --- Task 2: Generate DEMs ---
        # Pass the river shapefile path for potential stream burning
        generate_dems(
            settings=settings,
            wbt=wbt,
            contour_shp_path=contour_shp_path,
            contour_elev_field=contour_elev_field,
            river_shp_path=river_shp_path,  # Pass the river path
            stream_extract_threshold=(
                settings.processing.stream_extract_threshold
                if settings.processing.enable_stream_burning
                else None
            ),
        )

        # Get DEM paths for next tasks
        dem_interp_path = output_files.get_full_path("dem_interpolated_tif", output_dir)
        dem_topo_path = output_files.get_full_path("dem_topo_tif", output_dir)
        # Get the path for the INPUT ArcGIS Pro TopoToRaster DEM from PathsConfig
        dem_toporaster_all_path = (
            settings.paths.toporaster_all_input_tif
        )  # Use the direct path from settings
        # Get the path for the generated stream burn DEM
        dem_stream_burn_path = output_files.get_full_path(
            "dem_stream_burned_tif", output_dir
        )

        # --- Task 3: Quality Assessment ---
        assess_dem_quality(
            settings=settings,
            wbt=wbt,
            points_shp_path=points_shp_path,
            dem_interp_path=dem_interp_path,
            dem_topo_path=dem_topo_path,
            dem_toporaster_all_path=dem_toporaster_all_path,
            dem_stream_burn_path=dem_stream_burn_path,  # Added stream burn path
            point_elev_field=point_elev_field,
        )

        # --- Task 4: Generate Derived Products ---
        # Note: Derived products are currently only generated for interp and topo DEMs.
        # If derived products are needed for TopoRaster_all, this section would also need modification.
        generate_derived_products(settings, wbt, dem_interp_path, dem_topo_path)

    except FileNotFoundError as fnf_error:
        print("\n--- Workflow Halted: Required file not found ---")
        print(f"Error: {fnf_error}")
        sys.exit(1)
    except ValueError as val_error:
        print("\n--- Workflow Halted: Data error ---")
        print(f"Error: {val_error}")
        sys.exit(1)
    except Exception as e:
        print("\n--- Workflow Halted: An unexpected error occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        # Consider adding traceback logging here for debugging
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

    end_time = time.time()
    print("\n--- Workflow Script Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
