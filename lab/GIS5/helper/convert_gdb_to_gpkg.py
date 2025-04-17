# --- IMPORTANT NOTE ---
# Similar to `prepare_raw_data.py`, this script was build with the intention to prepare new data.
# Unfortunately, due to proprietary software restrictions, it is not possible to convert outputs
# to the desired format after automated processing. So we instead leave this code here as a reference
# in case future projects allow slightly more flexible inputs.
# --- END NOTE ---

import sys
import os
import argparse
import logging
from pathlib import Path

# --- PyQGIS Initialization (Crucial and Environment-Dependent) ---
# This section is critical for standalone execution and often requires
# setting environment variables (QGIS_PREFIX_PATH, PATH, PYTHONPATH, LD_LIBRARY_PATH)
# *before* importing qgis.core or processing.
# Running this script via QGIS's own Python interpreter is the most reliable way.
try:
    # Attempt to import core QGIS modules
    from qgis.core import (
        QgsApplication,
        QgsVectorLayer,
        QgsProject,
        QgsCoordinateReferenceSystem,
    )

    # Import the processing module
    import processing
    from processing.core.Processing import Processing
except ImportError as e:
    print(f"Error: Failed to import PyQGIS modules: {e}")
    print(
        "This script likely needs to be run using the Python interpreter bundled with QGIS,"
    )
    print(
        "or have the environment correctly configured (QGIS_PREFIX_PATH, PYTHONPATH, etc.)."
    )
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def convert_gdb_layers_to_gpkg(
    gdb_path: Path,
    gpkg_path: Path,
    layers_to_convert: list[str],
    target_crs_authid: str = None,
):
    """
    Converts specified layers from an ESRI File Geodatabase (.gdb) to a GeoPackage (.gpkg)
    using PyQGIS processing algorithms.

    Args:
        gdb_path: Path to the input File Geodatabase directory.
        gpkg_path: Path to the output GeoPackage file. Will be overwritten if it exists.
        layers_to_convert: A list of layer names within the GDB to convert.
        target_crs_authid: Optional target CRS (e.g., "EPSG:25832"). If None, uses layer's original CRS.
    """
    logging.info(f"Starting GDB to GPKG conversion for {gdb_path}")
    logging.info(f"Output GPKG: {gpkg_path}")
    logging.info(f"Layers to convert: {layers_to_convert}")
    if target_crs_authid:
        logging.info(f"Target CRS: {target_crs_authid}")

    # --- Initialize QGIS Application (Required for processing) ---
    # QgsApplication.setPrefixPath('/path/to/qgis/installation', True) # Adjust path if needed
    # Find QGIS prefix path automatically if possible (might not work everywhere)
    qgis_prefix = os.environ.get("QGIS_PREFIX_PATH")
    if not qgis_prefix:
        # Common paths - adjust as needed for your system
        potential_paths = [
            "/Applications/QGIS.app/Contents/MacOS",  # macOS default
            "/usr",  # Linux common
            # Add Windows paths if relevant
        ]
        for p in potential_paths:
            if os.path.exists(os.path.join(p, "lib/qgis")):
                qgis_prefix = p
                break
    if qgis_prefix:
        QgsApplication.setPrefixPath(qgis_prefix, True)
        logging.info(f"QGIS Prefix Path set to: {qgis_prefix}")
    else:
        logging.warning(
            "Could not automatically determine QGIS_PREFIX_PATH. Initialization might fail."
        )

    qgs = QgsApplication([], False)  # Use False for no GUI
    qgs.initQgis()
    logging.info("QGIS Application initialized.")

    # --- Initialize Processing Framework ---
    try:
        Processing.initialize()
        logging.info("QGIS Processing Framework initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize QGIS Processing Framework: {e}")
        qgs.exitQgis()
        raise

    # --- Delete existing GPKG if it exists ---
    if gpkg_path.exists():
        try:
            gpkg_path.unlink()
            logging.info(f"Removed existing output file: {gpkg_path}")
        except OSError as e:
            logging.error(f"Could not remove existing GPKG file {gpkg_path}: {e}")
            qgs.exitQgis()
            raise

    # --- Conversion Loop ---
    conversion_successful = True
    for layer_name in layers_to_convert:
        input_layer_uri = f"{gdb_path}|layername={layer_name}"
        logging.info(f"Processing layer: {layer_name} from URI: {input_layer_uri}")

        # Define parameters for the 'native:package' algorithm (more robust for GPKG)
        # or 'gdal:convertformat'
        params = {
            "LAYERS": [input_layer_uri],  # Input layer(s)
            "OUTPUT": str(gpkg_path),  # Output GPKG path
            "OVERWRITE": False,  # We delete the file first, so don't overwrite layers within
            "SAVE_STYLES": False,
            "SAVE_METADATA": False,
            "SELECTED_FEATURES_ONLY": False,
            # Add CRS transformation if needed
        }
        if target_crs_authid:
            # Check if CRS is valid
            target_crs_obj = QgsCoordinateReferenceSystem()
            if not target_crs_obj.createFromUserInput(target_crs_authid):
                logging.error(f"Invalid Target CRS specified: {target_crs_authid}")
                conversion_successful = False
                break  # Stop processing further layers
            # Add reprojection parameter if using gdal:convertformat
            # params['TARGET_CRS'] = target_crs_obj
            # For native:package, reprojection happens implicitly if layer CRS differs? Check docs.
            # Alternatively, use 'native:reprojectlayer' first. Let's keep it simple for now.
            logging.info(
                f"Note: Target CRS specified, but reprojection within package algorithm might need verification."
            )

        try:
            # Use the 'native:package' algorithm for creating/adding to GPKG
            result = processing.run("native:package", params)
            logging.info(
                f"Successfully processed layer '{layer_name}' into {gpkg_path}"
            )
            # Check result if needed: result['OUTPUT'] should be the gpkg_path

        except Exception as e:
            logging.error(f"Error processing layer '{layer_name}': {e}")
            # Attempt to load the layer to see if it exists, maybe the URI is wrong
            vlayer = QgsVectorLayer(input_layer_uri, f"test_{layer_name}", "ogr")
            if not vlayer.isValid():
                logging.error(
                    f"  -> Failed to load layer '{layer_name}' from GDB. Check layer name and GDB path."
                )
            else:
                logging.error(
                    f"  -> Layer '{layer_name}' seems valid, but processing failed."
                )
            conversion_successful = False
            # Continue to try other layers? Or break? Let's break for now.
            break

    # --- Cleanup ---
    qgs.exitQgis()
    logging.info("QGIS Application exited.")

    if not conversion_successful:
        raise RuntimeError("One or more layers failed during conversion.")
    else:
        logging.info("GDB to GPKG conversion completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert layers from GDB to GPKG using PyQGIS."
    )
    parser.add_argument(
        "gdb_path",
        type=Path,
        help="Path to the input File Geodatabase (.gdb directory).",
    )
    parser.add_argument(
        "gpkg_path", type=Path, help="Path for the output GeoPackage (.gpkg file)."
    )
    parser.add_argument(
        "-l",
        "--layers",
        required=True,
        nargs="+",
        help="List of layer names within the GDB to convert.",
    )
    parser.add_argument(
        "-crs",
        "--target_crs",
        type=str,
        default=None,
        help="Optional target CRS Auth ID (e.g., 'EPSG:25832').",
    )

    args = parser.parse_args()

    # Basic validation
    if not args.gdb_path.is_dir():
        print(f"Error: Input GDB path is not a valid directory: {args.gdb_path}")
        sys.exit(1)

    # Ensure output directory exists
    args.gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        convert_gdb_layers_to_gpkg(
            args.gdb_path, args.gpkg_path, args.layers, args.target_crs
        )
        print("Script finished successfully.")
    except Exception as e:
        print(f"Script failed: {e}")
        sys.exit(1)
