import geopandas as gpd
import warnings
import logging
from typing import Tuple, Dict, Optional

from pydantic import BaseModel
from pyproj import CRS

try:
    from src.utils import get_common_extent_and_crs, find_elevation_field
except ImportError:
    from utils import get_common_extent_and_crs, find_elevation_field

# Get a logger for this module
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    settings: BaseModel,
) -> Tuple[
    Dict[str, gpd.GeoDataFrame],
    Tuple[float, float, float, float],
    Optional[CRS],
    str,
    Optional[str],
]:
    """
    Loads input vector data from the Geodatabase, calculates common extent/CRS,
    saves intermediate shapefiles, and identifies elevation fields.

    Args:
        settings: The application configuration object.

    Returns:
        A tuple containing:
        - A dictionary of loaded GeoDataFrames (keys: 'contours', 'rivers', 'lakes', 'points').
        - A tuple representing the common extent (minx, miny, maxx, maxy).
        - The common CRS object (or None).
        - The identified contour elevation field name.
        - The identified point elevation field name (or None if not found).

    Raises:
        FileNotFoundError: If input GDB or required layers are not found.
        ValueError: If CRS reprojection fails or elevation fields cannot be determined.
    """
    logger.info(f"Reading data from: {settings.paths.gdb_path}")
    gdb_path = str(settings.paths.gdb_path)  # Geopandas might prefer string path
    output_dir = settings.paths.output_dir
    input_layers = settings.input_layers
    output_files = settings.output_files

    try:
        contours_gdf = gpd.read_file(gdb_path, layer=input_layers.contour_layer)
        rivers_gdf = gpd.read_file(gdb_path, layer=input_layers.river_layer)
        lakes_gdf = gpd.read_file(gdb_path, layer=input_layers.lake_layer)
        points_gdf = gpd.read_file(gdb_path, layer=input_layers.points_layer)

        logger.info(f"  - Read {len(contours_gdf)} contours.")
        logger.info(f"  - Read {len(rivers_gdf)} river segments.")
        logger.info(f"  - Read {len(lakes_gdf)} lakes.")
        logger.info(f"  - Read {len(points_gdf)} elevation points.")

        loaded_gdfs = {
            "contours": contours_gdf,
            "rivers": rivers_gdf,
            "lakes": lakes_gdf,
            "points": points_gdf,
        }

        # --- Check CRS and Calculate Common Extent ---
        logger.info("Calculating common extent and CRS...")
        all_layers_list = list(loaded_gdfs.values())
        minx, miny, maxx, maxy, common_crs = get_common_extent_and_crs(all_layers_list)
        logger.info(f"  - Common CRS: {common_crs}")
        logger.info(
            f"  - Combined Extent: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})"
        )
        common_extent = (minx, miny, maxx, maxy)

        # --- Save copies as Shapefiles ---
        logger.info("Saving intermediate Shapefiles...")
        # Suppress specific warnings related to Shapefile limitations
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Column names longer than 10 characters.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module="pyogrio\\.raw",  # Target warnings specifically from pyogrio
            )
            contours_gdf.to_file(output_files.get_full_path("contour_shp", output_dir))
            rivers_gdf.to_file(output_files.get_full_path("river_shp", output_dir))
            lakes_gdf.to_file(output_files.get_full_path("lake_shp", output_dir))
            points_gdf.to_file(output_files.get_full_path("points_shp", output_dir))
        logger.info("...Shapefile saving done.")

        # --- Identify Elevation Fields ---
        logger.info("Identifying elevation fields...")
        # Contour Elevation Field
        contour_elev_field = find_elevation_field(
            contours_gdf, [input_layers.contour_elevation_field]
        )  # Prioritize configured field
        if contour_elev_field is None:
            # Try candidates if primary fails (though config should be specific)
            contour_elev_field = find_elevation_field(
                contours_gdf, input_layers.point_elevation_field_candidates
            )  # Reuse candidates list
            if contour_elev_field is None:
                raise ValueError(
                    f"Could not find contour elevation field (tried '{input_layers.contour_elevation_field}' and candidates) in {input_layers.contour_layer}."
                )
            else:
                logger.warning(
                    f"Configured contour field '{input_layers.contour_elevation_field}' not found, using '{contour_elev_field}' instead."
                )
        else:
            logger.info(f"  - Using contour elevation field: '{contour_elev_field}'")

        # Point Elevation Field
        point_elev_field = find_elevation_field(
            points_gdf, input_layers.point_elevation_field_candidates
        )
        if point_elev_field is None:
            logger.warning(
                "Could not identify elevation field in points layer using candidates."
            )
            # Allow workflow to continue but RMSE will fail later if needed
        else:
            logger.info(f"  - Using point elevation field: '{point_elev_field}'")

        logger.info("--- Data Loading and Preparation Complete ---")
        return (
            loaded_gdfs,
            common_extent,
            common_crs,
            contour_elev_field,
            point_elev_field,
        )

    except ImportError as ie:
        logger.error(f"Error reading data: {ie}")
        logger.error(
            "Ensure the necessary driver (e.g., GDAL FileGDB driver) is correctly installed and accessible."
        )
        raise  # Re-raise after providing context
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data loading: {e}", exc_info=True
        )
        raise  # Re-raise the exception
