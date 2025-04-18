import logging
import shutil
from pathlib import Path
from typing import Optional, Union, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from shapely.geometry import Point


logger = logging.getLogger(__name__)


def setup_output_dir(output_dir: Path):
    """
    Sets up the output directory.

    If the directory exists, it removes its contents. Then, it creates the directory.

    Args:
        output_dir (Path): The path to the output directory.
    """
    try:
        if output_dir.exists() and output_dir.is_dir():
            logger.warning(f"Output directory {output_dir} exists. Removing contents.")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory setup complete: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to set up output directory {output_dir}: {e}")
        raise


def load_vector_data(
    path: Path, layer: Optional[str] = None, **kwargs
) -> gpd.GeoDataFrame:
    """
    Loads vector data from various formats using GeoPandas.

    Args:
        path (Path): Path to the vector file (e.g., .shp, .geojson, .gpkg).
        layer (Optional[str]): Layer name if reading from a multi-layer source like GeoPackage.
        **kwargs: Additional keyword arguments passed to geopandas.read_file.

    Returns:
        gpd.GeoDataFrame: Loaded GeoDataFrame.

    Raises:
        FileNotFoundError: If the input file does not exist.
        Exception: For other GeoPandas loading errors.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input vector file not found: {path}")
    try:
        logger.info(f"Loading vector data from: {path} (Layer: {layer or 'Default'})")
        gdf = gpd.read_file(path, layer=layer, **kwargs)
        logger.info(f"Loaded {len(gdf)} features with CRS: {gdf.crs}")
        return gdf
    except Exception as e:
        logger.error(f"Error loading vector file {path}: {e}")
        raise


def save_vector_data(
    gdf: gpd.GeoDataFrame,
    path: Path,
    layer: Optional[str] = None,
    driver: Optional[str] = None,
    **kwargs,
):
    """
    Saves a GeoDataFrame to a vector file.

    Determines driver based on file extension if not provided.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to save.
        path (Path): Output file path.
        layer (Optional[str]): Layer name, required for some formats like GeoPackage.
        driver (Optional[str]): OGR driver to use (e.g., 'GPKG', 'ESRI Shapefile', 'GeoJSON').
        **kwargs: Additional keyword arguments passed to gdf.to_file.
    """
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Infer driver if not specified
    if driver is None:
        suffix = path.suffix.lower()
        if suffix == ".gpkg":
            driver = "GPKG"
        elif suffix == ".shp":
            driver = "ESRI Shapefile"
        elif suffix == ".geojson":
            driver = "GeoJSON"
        else:
            logger.warning(
                f"Could not infer driver for {path}. Attempting save without explicit driver."
            )

    try:
        logger.info(
            f"Saving {len(gdf)} features to: {path} (Layer: {layer}, Driver: {driver})"
        )
        gdf.to_file(path, layer=layer, driver=driver, **kwargs)
        logger.info("Save complete.")
    except Exception as e:
        logger.error(f"Error saving vector file to {path}: {e}")
        raise


def load_raster_data(path: Path, **kwargs) -> Tuple[np.ndarray, dict]:
    """
    Loads a raster file using Rasterio.

    Args:
        path (Path): Path to the raster file (e.g., .tif).
        **kwargs: Additional keyword arguments passed to rasterio.open.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing the raster data as a NumPy array
                                 (first band only) and the raster profile (metadata).

    Raises:
        FileNotFoundError: If the input file does not exist.
        Exception: For other Rasterio loading errors.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input raster file not found: {path}")
    try:
        logger.info(f"Loading raster data from: {path}")
        with rasterio.open(path, **kwargs) as src:
            profile = src.profile
            # Read the first band
            # TODO: Handle multi-band rasters if necessary
            data = src.read(1)
            logger.info(
                f"Loaded raster ({data.shape[0]}x{data.shape[1]}) with CRS: {profile.get('crs', 'N/A')}"
            )
            return data, profile
    except Exception as e:
        logger.error(f"Error loading raster file {path}: {e}")
        raise


def save_raster_data(array: np.ndarray, profile: dict, path: Path, **kwargs):
    """
    Saves a NumPy array as a raster file using Rasterio.

    Args:
        array (np.ndarray): NumPy array containing raster data.
        profile (dict): Raster profile (metadata), typically obtained from a source raster
                        or created manually. Should include 'driver', 'height', 'width',
                        'count', 'dtype', 'crs', 'transform', 'nodata'.
        path (Path): Output file path.
        **kwargs: Additional keyword arguments to update the profile before writing.
    """
    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Update profile with provided kwargs and ensure essential keys are present
    profile.update(kwargs)
    profile.update(
        {
            "height": array.shape[0],
            "width": array.shape[1],
            "count": 1,  # Assuming single band output
            "dtype": array.dtype,
        }
    )

    # Ensure essential keys are present
    required_keys = ["driver", "height", "width", "count", "dtype", "crs", "transform"]
    missing_keys = [key for key in required_keys if key not in profile]
    if missing_keys:
        raise ValueError(f"Missing required keys in raster profile: {missing_keys}")

    try:
        logger.info(
            f"Saving raster data ({array.shape[0]}x{array.shape[1]}) to: {path}"
        )
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array, 1)  # Write to the first band
        logger.info("Save complete.")
    except Exception as e:
        logger.error(f"Error saving raster file to {path}: {e}")
        raise


def reproject_gdf(
    gdf: gpd.GeoDataFrame, target_crs: Union[str, int, CRS]
) -> gpd.GeoDataFrame:
    """
    Reprojects a GeoDataFrame to the target CRS.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame.
        target_crs (Union[str, int, CRS]): Target Coordinate Reference System
                                           (e.g., 'EPSG:4326', 4326, CRS.from_epsg(4326)).

    Returns:
        gpd.GeoDataFrame: Reprojected GeoDataFrame.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS defined. Cannot reproject.")

    try:
        target_crs_obj = CRS.from_user_input(target_crs)
        logger.info(
            f"Reprojecting GeoDataFrame from {gdf.crs} to {target_crs_obj.to_string()}"
        )
        gdf_reprojected = gdf.to_crs(target_crs_obj)
        logger.info("Reprojection complete.")
        return gdf_reprojected
    except Exception as e:
        logger.error(f"Error during reprojection to {target_crs}: {e}")
        raise


def polyline_to_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Converts Polyline/LineString geometries in a GeoDataFrame to Point geometries
    representing the vertices.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with LineString or MultiLineString geometries.

    Returns:
        gpd.GeoDataFrame: Output GeoDataFrame with Point geometries. Attributes are
                          duplicated for each vertex from the original line.
    """
    points_list = []
    original_columns = gdf.columns.tolist()  # Keep track of original columns

    for index, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        attributes = row.drop("geometry").to_dict()

        if geom.geom_type == "LineString":
            for point_coords in geom.coords:
                point_geom = Point(point_coords)
                points_list.append({**attributes, "geometry": point_geom})
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                for point_coords in line.coords:
                    point_geom = Point(point_coords)
                    points_list.append({**attributes, "geometry": point_geom})
        # Add handling for other geometry types if necessary, e.g., Points
        elif geom.geom_type == "Point":
            points_list.append({**attributes, "geometry": geom})  # Keep existing points
        else:
            logger.warning(
                f"Skipping unsupported geometry type at index {index}: {geom.geom_type}"
            )

    if not points_list:
        logger.warning("No points generated from polylines.")
        # Return an empty GeoDataFrame with the original schema and CRS
        return gpd.GeoDataFrame([], columns=original_columns, crs=gdf.crs)

    # Ensure the columns match the original, adding geometry if it wasn't there
    # (though it should be for a GeoDataFrame)
    final_columns = (
        original_columns
        if "geometry" in original_columns
        else original_columns + ["geometry"]
    )

    points_gdf = gpd.GeoDataFrame(points_list, columns=final_columns, crs=gdf.crs)
    logger.info(
        f"Converted/processed {len(gdf)} input features to {len(points_gdf)} points."
    )
    return points_gdf
