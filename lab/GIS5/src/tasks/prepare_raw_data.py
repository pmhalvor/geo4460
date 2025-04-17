# --- IMPORTANT NOTE ---
# This script is designed to prepare raw geospatial data (e.g., clipping, selecting features)
# using open-source Python libraries (GeoPandas, Fiona, Shapely).
# It is expected to be moved to src/tasks/ in order for imports to work.
# Unfortunately, it relies on the underlying GDAL/Fiona installation being able to read the
# input data source specified in the configuration (config.paths.input_raw_gdb).
#
# LIMITATION ENCOUNTERED (April 2025):
# The common open-source distributions of GDAL and Fiona (e.g., via brew or pip)
# often lack the proprietary ESRI File Geodatabase (.gdb) driver. This prevents this script
# from directly reading data from a .gdb file.
#
# WORKAROUND:
# To use this script, the required input layers (e.g., contours, elevation, land cover)
# must first be manually exported from the source .gdb file into an open format
# like GeoPackage (.gpkg) using software like ArcGIS Pro or QGIS.
# The configuration (config.paths.input_raw_gdb or a similar variable for GPKG)
# should then be updated to point to the converted GeoPackage file.
# --- END NOTE ---

import geopandas as gpd
import fiona
import os
import random
import logging
from shapely.geometry import Polygon, Point
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_raw_data(config):
    """
    Prepares raw geodata (e.g., N50) using open-source libraries based on config.

    Steps:
    1. Define paths and layer names (requires verification, potentially from config).
    2. Read relevant layers (e.g., contours, elevation, land cover) from input GDB.
    3. Create a random Area of Interest (AOI) polygon within the data bounds.
    4. Clip N50 layers to the AOI.
    5. Select specific land cover types (lakes, rivers).
    6. Convert river polygons to polylines (using boundary approximation).
    7. Prepare final GeoDataFrames with standard names.
    8. Export final layers to a new GeoPackage.
    """
    logging.info("Starting raw data preparation using GeoPandas...")

    # Access paths and parameters from the main settings object
    try:
        input_gdb = Path(config.paths.input_raw_gdb)
        output_dir = Path(config.paths.output_dir_prepared)
        target_crs_epsg_code = config.processing.target_crs_epsg
    except AttributeError as e:
        logging.error(
            f"Configuration error: Missing expected attribute in settings object: {e}"
        )
        logging.error(
            "Ensure PathsConfig has 'input_raw_gdb', 'output_dir_prepared' and ProcessingConfig has 'target_crs_epsg'."
        )
        raise ValueError(
            "Invalid configuration object passed to prepare_raw_data."
        ) from e

    # Check if input GDB path is actually set
    if not input_gdb:
        logging.error(
            "Configuration error: 'input_raw_gdb' path is not set in the configuration."
        )
        raise ValueError(
            "'input_raw_gdb' path must be configured to run prepare_raw_data."
        )

    output_gpkg_name = "Prepared_Data.gpkg"  # Generic output name
    output_gpkg_path = output_dir / output_gpkg_name

    # Ensure output directory exists (config should handle this, but double-check)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured output directory exists: {output_dir}")

    # --- Step 1: Define Layer Names and Parameters (Needs verification) ---
    # These names are guesses based on common N50 naming conventions and the task description.
    # *** We MUST verify these layer names by inspecting the GDB contents. ***
    contour_layer_name = "N50_SentralLinje"  # Guess for contour lines
    elevation_layer_name = "N50_Hoyde"  # Guess for elevation points
    land_cover_layer_name = "N50_Arealdekke_Omrade"  # Guess for land cover polygons - TODO: Make configurable?
    land_cover_type_field = "OBJTYPE"  # Common field name for object type, needs verification - TODO: Make configurable?
    target_crs = f"EPSG:{target_crs_epsg_code}"  # Use the code from config

    logging.info(f"Using Input Raw GDB: {input_gdb}")
    logging.info(f"Target Output Prepared GPKG: {output_gpkg_path}")
    logging.info(
        f"Potential Layer Names (Verification Required): Contours='{contour_layer_name}', Elevation='{elevation_layer_name}', LandCover='{land_cover_layer_name}'"
    )
    logging.info(
        f"Land Cover Type Field (Verification Required): '{land_cover_type_field}'"
    )
    logging.info(f"Target CRS: {target_crs}")

    # --- Check GDB Read Capability ---
    try:
        gdb_layers = fiona.listlayers(input_gdb)
        logging.info(f"Successfully listed layers in GDB: {gdb_layers}")
        # Check if guessed layers exist
        if contour_layer_name not in gdb_layers:
            logging.warning(
                f"Contour layer '{contour_layer_name}' not found in GDB layers: {gdb_layers}. Please verify name."
            )
        if elevation_layer_name not in gdb_layers:
            logging.warning(
                f"Elevation layer '{elevation_layer_name}' not found in GDB layers: {gdb_layers}. Please verify name."
            )
        if land_cover_layer_name not in gdb_layers:
            logging.warning(
                f"Land cover layer '{land_cover_layer_name}' not found in GDB layers: {gdb_layers}. Please verify name."
            )

    except fiona.errors.DriverError as e:
        logging.error(f"Could not read layers from GDB: {input_gdb}. Error: {e}")
        logging.error(
            "Ensure GDAL is compiled with FileGDB driver support, or export GDB layers to GeoPackage/Shapefile first."
        )
        raise ValueError(
            f"Failed to read input raw GDB '{input_gdb}'. Check driver support or convert data."
        ) from e

    try:
        # --- Step 2: Read Layers ---
        logging.info("Reading layers from input raw GDB...")
        # TODO: Add error handling if layers don't exist or names are wrong
        gdf_contours = gpd.read_file(input_gdb, layer=contour_layer_name)
        gdf_elevation = gpd.read_file(input_gdb, layer=elevation_layer_name)
        gdf_land_cover = gpd.read_file(input_gdb, layer=land_cover_layer_name)
        logging.info(
            f"Read {len(gdf_contours)} contours, {len(gdf_elevation)} elevation points, {len(gdf_land_cover)} land cover polygons."
        )

        # Ensure consistent CRS
        logging.info("Ensuring consistent CRS...")
        gdf_contours = gdf_contours.to_crs(target_crs)
        gdf_elevation = gdf_elevation.to_crs(target_crs)
        gdf_land_cover = gdf_land_cover.to_crs(target_crs)
        logging.info(f"Data projected to {target_crs}.")

        # --- Step 3: Create AOI Polygon ---
        logging.info("Creating Area of Interest (AOI) polygon...")
        # Combine extents to define the overall area
        total_bounds = gdf_contours.total_bounds  # [minx, miny, maxx, maxy]
        # TODO: Refine bounds based on all layers if necessary
        minx, miny, maxx, maxy = total_bounds
        logging.info(f"Total bounds: {total_bounds}")

        # Generate random points within the bounds to define a rectangle
        # Ensure the rectangle is reasonably sized and placed
        aoi_width = (maxx - minx) * random.uniform(0.1, 0.3)  # Example: 10-30% of width
        aoi_height = (maxy - miny) * random.uniform(
            0.1, 0.3
        )  # Example: 10-30% of height
        aoi_minx = random.uniform(minx, maxx - aoi_width)
        aoi_miny = random.uniform(miny, maxy - aoi_height)
        aoi_maxx = aoi_minx + aoi_width
        aoi_maxy = aoi_miny + aoi_height

        aoi_poly = Polygon(
            [
                (aoi_minx, aoi_miny),
                (aoi_maxx, aoi_miny),
                (aoi_maxx, aoi_maxy),
                (aoi_minx, aoi_maxy),
            ]
        )
        gdf_aoi = gpd.GeoDataFrame([1], geometry=[aoi_poly], crs=target_crs)
        logging.info(
            f"Created AOI polygon with bounds: ({aoi_minx:.2f}, {aoi_miny:.2f}, {aoi_maxx:.2f}, {aoi_maxy:.2f})"
        )

        # --- Step 4: Clip Layers ---
        logging.info("Clipping layers to AOI...")
        gdf_contours_clipped = gpd.clip(gdf_contours, gdf_aoi)
        gdf_elevation_clipped = gpd.clip(gdf_elevation, gdf_aoi)
        gdf_land_cover_clipped = gpd.clip(gdf_land_cover, gdf_aoi)
        logging.info(
            f"Clipping complete. Contours: {len(gdf_contours_clipped)}, Elevation: {len(gdf_elevation_clipped)}, Land Cover: {len(gdf_land_cover_clipped)}"
        )

        # --- Step 5: Select Lakes and Rivers from Land Cover ---
        logging.info(
            f"Selecting lakes and rivers using field '{land_cover_type_field}'..."
        )
        # Verify the field exists
        if land_cover_type_field not in gdf_land_cover_clipped.columns:
            logging.error(
                f"Field '{land_cover_type_field}' not found in clipped land cover data. Columns: {gdf_land_cover_clipped.columns}"
            )
            raise KeyError(
                f"Field '{land_cover_type_field}' not found in clipped land cover data."
            )

        lake_types = ["innsjø", "innsjø regulert"]
        river_types = ["elv"]  # Assuming 'elv' means river polygon

        gdf_lakes_poly = gdf_land_cover_clipped[
            gdf_land_cover_clipped[land_cover_type_field].isin(lake_types)
        ].copy()
        gdf_rivers_poly = gdf_land_cover_clipped[
            gdf_land_cover_clipped[land_cover_type_field].isin(river_types)
        ].copy()
        logging.info(
            f"Selected {len(gdf_lakes_poly)} lake polygons and {len(gdf_rivers_poly)} river polygons."
        )

        # --- Step 6: Convert River Polygons to Polylines ---
        # This is non-trivial. A common approach is to get the boundary, but that might include shorelines.
        # Another is skeletonization (centerline), which is complex.
        # Simplest approximation: Use the polygon boundaries as lines.
        logging.info("Converting river polygons to polylines (using boundary)...")
        if not gdf_rivers_poly.empty:
            gdf_rivers_lines = gdf_rivers_poly.boundary.to_frame(name="geometry")
            # Ensure it's a GeoDataFrame with the correct CRS
            gdf_rivers_lines = gpd.GeoDataFrame(gdf_rivers_lines, crs=target_crs)
            # Explode MultiLineStrings into LineStrings if necessary
            gdf_rivers_lines = gdf_rivers_lines.explode(index_parts=True)
            logging.info(
                f"Converted river polygons to {len(gdf_rivers_lines)} line features."
            )
        else:
            # Create an empty GeoDataFrame with LineString type if no rivers found
            gdf_rivers_lines = gpd.GeoDataFrame(
                {"geometry": []}, geometry="geometry", crs=target_crs
            ).astype({"geometry": "geometry"})
            logging.info("No river polygons found to convert to lines.")

        # --- Step 7: Prepare Final Layers ---
        logging.info("Preparing final layers for export...")
        # Rename columns if necessary, select relevant columns, etc.
        # For now, we just use the clipped/processed data directly.
        final_layers = {
            "contour_arc": gdf_contours_clipped,
            "elevation_point": gdf_elevation_clipped,
            "lake_polygon": gdf_lakes_poly,
            "river_arc": gdf_rivers_lines,
        }

        # --- Step 8: Export Layers to GeoPackage ---
        logging.info(f"Exporting final layers to {output_gpkg_path}...")
        # Remove existing GPKG if it exists
        if output_gpkg_path.exists():
            output_gpkg_path.unlink()
            logging.info(f"Deleted existing output GPKG: {output_gpkg_path}")

        for layer_name, gdf in final_layers.items():
            if not gdf.empty:
                # Ensure geometry column is named 'geometry'
                if gdf.geometry.name != "geometry":
                    gdf = gdf.set_geometry(
                        gdf.geometry.name
                    )  # Find the geometry column and set it

                # Check for mixed geometry types before saving
                geom_types = gdf.geometry.geom_type.unique()
                if len(geom_types) > 1:
                    logging.warning(
                        f"Layer '{layer_name}' contains mixed geometry types: {geom_types}. Attempting to save anyway."
                    )
                elif len(geom_types) == 0:
                    logging.warning(
                        f"Layer '{layer_name}' has no geometry types (possibly empty after processing). Skipping save."
                    )
                    continue  # Skip saving empty or typeless layers

                try:
                    gdf.to_file(output_gpkg_path, layer=layer_name, driver="GPKG")
                    logging.info(f"Exported layer: {layer_name}")
                except Exception as e:
                    logging.error(f"Failed to export layer {layer_name}: {e}")
                    # Potentially add more specific error handling for Fiona/GDAL issues
            else:
                logging.info(f"Skipping empty layer: {layer_name}")

        logging.info("Raw data preparation completed successfully.")

    except KeyError as e:
        logging.error(
            f"Configuration key error or attribute error accessing settings: {e}."
        )
        raise
    except fiona.errors.DriverError as e:
        # Catch potential errors during read_file if initial check passed but read failed
        logging.error(f"Fiona/GDAL Driver Error during file read/write: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback

        logging.error(traceback.format_exc())  # Log full traceback for debugging
        raise


if __name__ == "__main__":
    # Example for running this script standalone
    # This requires loading the main config settings first
    try:
        # Attempt to import the main settings object
        from src.config import settings as main_settings

        # --- IMPORTANT: Configure necessary paths for standalone run ---
        # You MUST set the input_raw_gdb path here or ensure it's set in config.py
        # Example: Override if not set in config.py
        if not main_settings.paths.input_raw_gdb:
            main_settings.paths.input_raw_gdb = Path(
                "lab/GIS5/GIS5_datafiles/Basisdata_46_Vestland_25832_N50Kartdata_FGDB.gdb"
            )  # Example path
            print(
                f"INFO: 'input_raw_gdb' was not set, using default example: {main_settings.paths.input_raw_gdb}"
            )

        # Optionally override the run flag (not needed for direct call)
        # main_settings.processing.run_prepare_raw_data = True

        print("Running prepare_raw_data with settings from config.py...")
        prepare_raw_data(main_settings)  # Call the renamed function
        print(
            f"Script finished. Output potentially written to {main_settings.paths.output_dir_prepared}/Prepared_Data.gpkg"
        )

    except ImportError:
        print("Could not import main settings from src.config.")
        print(
            "Please run this script as part of the main workflow or ensure the 'src' directory is in the Python path."
        )
    except ValueError as ve:  # Catch config errors specifically
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")
        import traceback

        traceback.print_exc()
