import base64
import logging
import geopandas as gpd
import io
import numpy as np
import pandas as pd 
import rasterio
import rasterio.warp
import shutil

from pyproj import CRS
from pathlib import Path
from typing import (
    Any,
    Dict,
    List, 
    Optional, 
    Tuple,
    Union, 
)
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


def get_raster_profile(path: Path) -> dict:
    """
    Reads and returns the profile (metadata) of a raster file.

    Args:
        path (Path): Path to the raster file.

    Returns:
        dict: The raster profile.

    Raises:
        FileNotFoundError: If the input file does not exist.
        Exception: For other Rasterio loading errors.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input raster file not found: {path}")
    try:
        logger.info(f"Reading profile from raster: {path}")
        with rasterio.open(path) as src:
            profile = src.profile
            logger.info(f"Profile read successfully for {path}.")
            return profile
    except Exception as e:
        logger.error(f"Error reading profile from raster file {path}: {e}")
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


def align_rasters(
    source_raster_path: Path,
    template_raster_path: Path,
    output_raster_path: Path,
    resampling_method: str = "bilinear",
):
    """
    Aligns a source raster to match the grid (CRS, transform, dimensions) of a template raster.

    Uses rasterio.warp.reproject. 
    For more info on resampling, see: https://pygis.io/docs/e_raster_resample.html

    Args:
        source_raster_path (Path): Path to the raster to be aligned.
        template_raster_path (Path): Path to the raster defining the target grid.
        output_raster_path (Path): Path to save the aligned raster.
        resampling_method (str): Resampling method to use (e.g., 'nearest', 'bilinear', 'cubic').
                                 Defaults to 'bilinear'.

    Raises:
        FileNotFoundError: If source or template raster does not exist.
        Exception: For errors during reprojection/alignment.
    """
    if not source_raster_path.is_file():
        raise FileNotFoundError(f"Source raster not found: {source_raster_path}")
    if not template_raster_path.is_file():
        raise FileNotFoundError(f"Template raster not found: {template_raster_path}")

    # Ensure output directory exists
    output_raster_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Aligning raster '{source_raster_path.name}' to template '{template_raster_path.name}' -> '{output_raster_path.name}'"
    )

    try:
        with rasterio.open(template_raster_path) as template_ds:
            template_profile = template_ds.profile
            dst_crs = template_ds.crs
            dst_transform = template_ds.transform
            dst_height = template_ds.height
            dst_width = template_ds.width

        with rasterio.open(source_raster_path) as src_ds:
            # Prepare output profile based on template, but update dtype from source
            out_profile = template_profile.copy()
            out_profile.update(
                {
                    "crs": dst_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                    "dtype": src_ds.profile['dtype'], # Correctly access dtype from profile
                    "nodata": src_ds.nodata, # Use source nodata value
                    "count": src_ds.count, # Use source band count
                }
            )

            # Open the output file for writing
            with rasterio.open(output_raster_path, "w", **out_profile) as dst_ds:
                # Iterate through bands if multi-band
                for i in range(1, src_ds.count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(src_ds, i),
                        destination=rasterio.band(dst_ds, i),
                        src_transform=src_ds.transform,
                        src_crs=src_ds.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=rasterio.warp.Resampling[resampling_method],
                    )
        logger.info(f"Successfully aligned raster saved to: {output_raster_path}")

    except Exception as e:
        logger.error(
            f"Error aligning raster {source_raster_path} to {template_raster_path}: {e}"
        )
        # Clean up potentially partially written output file
        if output_raster_path.exists():
            try:
                output_raster_path.unlink()
                logger.debug(f"Removed partially written output file: {output_raster_path}")
            except OSError as rm_e:
                logger.warning(f"Could not remove partial output file {output_raster_path}: {rm_e}")
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

        # --- Validity Check Before Reprojection ---
        valid_before = gdf.geometry.is_valid
        num_valid_before = valid_before.sum()
        num_total_before = len(gdf)
        logger.info(f"Validity before reprojection: {num_valid_before}/{num_total_before} valid.")
        if num_valid_before < num_total_before:
            invalid_indices_before = gdf[~valid_before].index.tolist()
            logger.warning(f"Indices invalid BEFORE reprojection: {invalid_indices_before[:10]}...") # Log first 10
        # --- End Check ---

        gdf_reprojected = gdf.to_crs(target_crs_obj)
        logger.info("Reprojection complete.")
        # Basic validity check after reprojection (optional, can be done in calling function)
        # num_invalid_after = (~gdf_reprojected.geometry.is_valid).sum()
        # if num_invalid_after > 0:
        #     logger.warning(f"{num_invalid_after} geometries became invalid during reprojection to {target_crs_obj.to_string()}.")
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


def display_raster_on_folium_map(
    raster_path_str: str,
    output_html_path_str: str,
    target_crs_epsg: int, # Added argument instead of relying on settings
    nodata_transparent: bool = True,
    cmap_name: str = 'viridis',
    opacity: float = 0.7,
    zoom_start: int = 12,
    tiles: str = 'CartoDB positron',
):
    """
    Displays a raster layer on a Folium map and saves it as an HTML file.

    Args:
        raster_path_str (str): Path to the input raster file.
        output_html_path_str (str): Path to save the output HTML map.
        target_crs_epsg (int): The expected EPSG code of the input raster's CRS.
                               Used for verification and bounds transformation.
        nodata_transparent (bool): Whether to make NoData pixels transparent. Defaults to True.
        cmap_name (str): Name of the matplotlib colormap to use. Defaults to 'viridis'.
        opacity (float): Opacity of the raster layer (0.0 to 1.0). Defaults to 0.7.
        zoom_start (int): Initial zoom level for the map. Defaults to 12.
        tiles (str): Folium tile layer name. Defaults to 'CartoDB positron'.
    """
    # Import necessary libraries here to avoid top-level dependency
    try:
        import folium
        import rasterio
        import rasterio.warp
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import io
        import base64
        from pathlib import Path
    except ImportError as e:
        logger.error(f"Folium display skipped: Missing required library ({e}). Install folium, matplotlib, rasterio.")
        return

    raster_path = Path(raster_path_str)
    output_html_path = Path(output_html_path_str)
    output_html_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    logger.info(f"Attempting to display raster: {raster_path} on Folium map.")

    if not raster_path.is_file():
        logger.error(f"Raster file not found: {raster_path}")
        return

    try:
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            raster_crs = src.crs
            logger.info(f"Folium Display: Raster CRS read from file: {raster_crs}")
            data = src.read(1, masked=True) # Read first band as masked array

            # Verify the CRS before transforming
            # Updated check: Remove deprecated 'is_valid'
            if not raster_crs:
                logger.error(
                    f"Folium Display: Missing CRS ({raster_crs}) read from {raster_path}. Cannot transform bounds accurately."
                )
                return # Cannot proceed without valid CRS
            elif raster_crs.to_epsg() != target_crs_epsg:
                 logger.warning(
                    f"Folium Display: CRS read from raster ({raster_crs}) does not match expected EPSG:{target_crs_epsg}. Transformation might be incorrect."
                )
                # Proceed with caution

            # Transform bounds to WGS84 (EPSG:4326) for Folium
            logger.info(f"Folium Display: Original bounds ({raster_crs}): {bounds}")
            bounds_4326 = None
            try:
                bounds_4326 = rasterio.warp.transform_bounds(
                    raster_crs, "EPSG:4326", *bounds # left, bottom, right, top
                )
                logger.info(f"Folium Display: Transformed bounds (EPSG:4326): {bounds_4326}")
            except Exception as warp_e:
                logger.error(f"Folium Display: Error during bounds transformation: {warp_e}", exc_info=True)
                return # Cannot proceed without transformed bounds

            # Prepare data for image overlay
            cmap = plt.get_cmap(cmap_name) # Use argument
            valid_data = data.compressed() # Get only valid (unmasked) data
            rgba_data_calc = None
            if valid_data.size > 0:
                # Normalize based on valid data range (e.g., 5th to 95th percentile for better contrast)
                vmin = np.percentile(valid_data, 5)
                vmax = np.percentile(valid_data, 95)
                # Handle case where vmin == vmax (e.g., constant raster)
                if vmin == vmax:
                    norm = colors.Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6) # Avoid division by zero
                else:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)

                rgba_data_calc = cmap(norm(data), bytes=True) # Apply colormap

                # Make masked pixels transparent if requested
                if nodata_transparent and data.mask.any():
                    rgba_data_calc[data.mask] = (0, 0, 0, 0)
            else:
                logger.warning("Raster data contains no valid pixels. Cannot create image overlay.")
                return # Cannot create overlay without data

            # Check if we have valid pixel data (rgba_data_calc) and transformed bounds
            if rgba_data_calc is not None and bounds_4326 is not None:
                map_bounds_folium = [
                    [bounds_4326[1], bounds_4326[0]], # SouthWest (lat, lon)
                    [bounds_4326[3], bounds_4326[2]], # NorthEast (lat, lon)
                ]
                center_lat = (bounds_4326[1] + bounds_4326[3]) / 2
                center_lon = (bounds_4326[0] + bounds_4326[2]) / 2

                # Save RGBA data to a PNG in memory
                buf = io.BytesIO()
                plt.imsave(buf, rgba_data_calc, format="png")
                buf.seek(0)
                png_base64 = base64.b64encode(buf.read()).decode("utf-8")
                img_uri = f"data:image/png;base64,{png_base64}"

                # Create Folium map
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=zoom_start, # Use argument
                    tiles=tiles, # Use argument
                )

                # Add raster as ImageOverlay
                img_overlay = folium.raster_layers.ImageOverlay(
                    image=img_uri,
                    bounds=map_bounds_folium,
                    opacity=opacity, # Use argument
                    name="Raster Overlay", # Generic name
                )
                img_overlay.add_to(m)
                folium.LayerControl().add_to(m)

                # Save map
                m.save(str(output_html_path))
                logger.info(f"Folium map with raster overlay saved to: {output_html_path}")
            else:
                # This condition should ideally be caught earlier
                logger.warning("Skipping Folium map generation due to missing data or bounds.")

    except rasterio.RasterioIOError as rio_e:
        logger.error(f"Error opening or reading raster file {raster_path}: {rio_e}", exc_info=True)
    except Exception as map_e:
        logger.error(f"Error generating Folium map for {raster_path}: {map_e}", exc_info=True)


def display_overlay_folium_map(
    overlay_gdfs: dict[str, gpd.GeoDataFrame],
    input_rasters: dict[str, Path],
    output_html_path_str: str,
    tooltip_columns: Optional[list] = None,
    zoom_start: int = 12,
    tiles: str = 'CartoDB positron',
):
    """
    Displays multiple vector overlay layers (A, B, C, D) and their corresponding
    input feature rasters (speed, traffic, cost, etc.) on a single Folium map.

    Args:
        overlay_gdfs (dict[str, gpd.GeoDataFrame]): Dictionary mapping overlay keys
            (e.g., 'A', 'B', 'C', 'D') to their GeoDataFrames. Assumes GDFs have CRS.
        input_rasters (dict[str, Path]): Dictionary mapping input feature names
            (e.g., 'speed', 'traffic', 'cost', 'popularity') to their raster file Paths.
        output_html_path_str (str): Path to save the output HTML map.
        tooltip_columns (Optional[list]): List of column names from overlay GDFs
                                          to show in tooltips.
        zoom_start (int): Initial zoom level for the map. Defaults to 12.
        tiles (str): Folium tile layer name. Defaults to 'CartoDB positron'.
    """
    # Import necessary libraries here
    import folium
    import rasterio
    import rasterio.warp
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import io
    import base64
    from pathlib import Path

    output_html_path = Path(output_html_path_str)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating combined overlay Folium map: {output_html_path}")

    # --- Determine Map Bounds and Center ---
    # Use bounds from the final overlay (e.g., 'D') if available, otherwise first available
    bounds_gdf = None
    for key in ['D', 'C', 'B', 'A']:
        gdf = overlay_gdfs.get(f'overlay_{key.lower()}') # Match keys used in combine_features
        if gdf is not None and not gdf.empty:
            try:
                if gdf.crs is None:
                     logger.warning(f"Overlay {key} GDF missing CRS, cannot use for bounds.")
                     continue
                if gdf.crs.to_epsg() != 4326:
                    bounds_gdf = gdf.to_crs(epsg=4326)
                else:
                    bounds_gdf = gdf
                # Validate geometry after potential reprojection
                bounds_gdf = bounds_gdf[bounds_gdf.geometry.is_valid & ~bounds_gdf.geometry.is_empty]
                if not bounds_gdf.empty:
                    logger.info(f"Using bounds from Overlay {key} for map extent.")
                    break # Found a valid GDF for bounds
            except Exception as reproj_e:
                logger.warning(f"Could not reproject Overlay {key} for bounds: {reproj_e}")
                bounds_gdf = None # Reset if reprojection failed

    if bounds_gdf is None or bounds_gdf.empty:
        logger.warning("No valid overlay GDF found to determine map bounds. Using default location.")
        center_lat, center_lon = 59.9139, 10.7522 # Default Oslo center
        map_bounds = None
    else:
        total_bounds = bounds_gdf.total_bounds
        if np.isnan(total_bounds).any() or np.isinf(total_bounds).any():
             logger.error(f"Invalid bounds calculated from overlays: {total_bounds}. Using default location.")
             center_lat, center_lon = 59.9139, 10.7522 # Default Oslo center
             map_bounds = None
        else:
            center_lat = (total_bounds[1] + total_bounds[3]) / 2
            center_lon = (total_bounds[0] + total_bounds[2]) / 2
            # Define bounds for potential fit_bounds call later if needed
            map_bounds = [[total_bounds[1], total_bounds[0]], [total_bounds[3], total_bounds[2]]] # SW, NE

    # --- Create Base Map ---
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles=tiles)

    # --- Add Input Raster Layers (Hidden by Default) ---
    raster_cmaps = {
        'speed': 'plasma',
        'traffic': 'inferno',
        'cost': 'viridis_r', # Reversed viridis for cost (lower is better)
        'popularity': 'magma',
        # Add more specific rasters if needed
        'traffic_morning': 'coolwarm',
        'traffic_daytime': 'hot',
        'traffic_evening': 'bone',
        'slope': 'coolwarm',
        'elevation': 'terrain',
    }
    default_raster_cmap = 'gray'

    for name, raster_path in input_rasters.items():
        if raster_path and raster_path.exists():
            logger.info(f"Processing raster layer: {name} ({raster_path})")
            try:
                with rasterio.open(raster_path) as src:
                    r_bounds = src.bounds
                    r_crs = src.crs
                    r_data = src.read(1, masked=True)

                    if not r_crs:
                        logger.warning(f"Skipping raster '{name}': Missing CRS.")
                        continue
                    # Verify CRS matches expected target_crs_epsg for consistency check
                    # if r_crs.to_epsg() != target_crs_epsg:
                    #      logger.warning(f"Raster '{name}' CRS ({r_crs}) differs from expected target EPSG:{target_crs_epsg}.")
                         # Proceeding, but transformation might be less accurate if CRS is wrong

                    # Transform bounds to WGS84
                    r_bounds_4326 = rasterio.warp.transform_bounds(r_crs, "EPSG:4326", *r_bounds)
                    r_map_bounds = [[r_bounds_4326[1], r_bounds_4326[0]], [r_bounds_4326[3], r_bounds_4326[2]]]

                    # Prepare image data
                    cmap = plt.get_cmap(raster_cmaps.get(name, default_raster_cmap))
                    valid_data = r_data.compressed()
                    if valid_data.size == 0:
                        logger.warning(f"Skipping raster '{name}': No valid data.")
                        continue

                    # Normalize (consider percentile for robustness)
                    vmin = np.percentile(valid_data, 5)
                    vmax = np.percentile(valid_data, 95)
                    if vmin == vmax: 
                        vmin, vmax = vmin - 1e-6, vmax + 1e-6 # Avoid constant value issue
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    rgba_data = cmap(norm(r_data), bytes=True)
                    if r_data.mask.any():
                        rgba_data[r_data.mask] = (0, 0, 0, 0) # Transparency

                    # Convert to base64 PNG
                    buf = io.BytesIO()
                    plt.imsave(buf, rgba_data, format="png")
                    img_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

                    # Add ImageOverlay
                    folium.raster_layers.ImageOverlay(
                        image=img_uri,
                        bounds=r_map_bounds,
                        opacity=0.6, # Slightly less opaque for context layers
                        name=f"Input: {name.replace('_', ' ').title()}", # Layer name
                        show=False, # Hidden by default
                    ).add_to(m)
                    logger.info(f"Added raster layer '{name}' to map (hidden).")

            except Exception as raster_e:
                logger.error(f"Failed to process raster layer '{name}' ({raster_path}): {raster_e}", exc_info=True)
        elif raster_path:
            logger.warning(f"Input raster file not found: {raster_path} for key '{name}'")


    # --- Add Vector Overlay Layers ---
    overlay_colors = {
        'A': '#1f77b4', # Blue
        'B': '#2ca02c', # Green
        'C': '#ff7f0e', # Orange
        'D': '#d62728', # Red
    }
    default_overlay_color = '#888888' # Grey

    for key_upper in ['A', 'B', 'C', 'D']:
        key_lower = f'overlay_{key_upper.lower()}'
        gdf = overlay_gdfs.get(key_lower)

        if gdf is not None and not gdf.empty:
            logger.info(f"Processing vector overlay layer: {key_upper}")
            # Reproject and validate
            try:
                if gdf.crs is None:
                    raise ValueError("Missing CRS")
                gdf_4326 = gdf.to_crs(epsg=4326) if gdf.crs.to_epsg() != 4326 else gdf.copy()

                # Clean geometries (null, empty, invalid with buffer(0))
                gdf_4326 = gdf_4326.dropna(subset=['geometry'])
                gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty]
                invalid_mask = ~gdf_4326.geometry.is_valid
                if invalid_mask.sum() > 0:
                    logger.info(f"Attempting buffer(0) on {invalid_mask.sum()} invalid geometries in Overlay {key_upper}")
                    gdf_4326.loc[invalid_mask, 'geometry'] = gdf_4326.loc[invalid_mask].geometry.buffer(0)
                    # Re-filter after buffer
                    gdf_4326 = gdf_4326.dropna(subset=['geometry'])
                    gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty]

                if gdf_4326.empty:
                    logger.warning(f"Overlay {key_upper} is empty after cleaning. Skipping.")
                    continue

                # drop potentially erroneous columns
                gdf_4326 = gdf_4326.drop(columns=["created_at", "points"], errors="ignore")

                # Prepare tooltip/popup (use only columns present in this specific GDF)
                available_tooltip_cols = [col for col in (tooltip_columns or []) if col in gdf_4326.columns]
                tooltip = folium.features.GeoJsonTooltip(fields=available_tooltip_cols) if available_tooltip_cols else None
                # popup = folium.features.GeoJsonPopup(fields=available_tooltip_cols) if available_tooltip_cols else None # Optional popup

                # Define style for this layer
                layer_color = overlay_colors.get(key_upper, default_overlay_color)
                def layer_style_function(feature, color=layer_color): # Capture color
                    return {
                        'color': color,
                        'weight': 3,
                        'opacity': 0.85,
                    }

                # Add GeoJson layer
                folium.GeoJson(
                    gdf_4326,
                    style_function=layer_style_function,
                    tooltip=tooltip,
                    # popup=popup,
                    name=f"Overlay {key_upper}", # Simple name for layer control
                ).add_to(m)
                logger.info(f"Added vector overlay layer '{key_upper}' to map.")

            except Exception as vector_e:
                 logger.error(f"Failed to process vector overlay layer '{key_upper}': {vector_e}", exc_info=True)

        elif gdf is not None and gdf.empty:
             logger.info(f"Skipping Overlay {key_upper} as it is empty.")
        else:
             logger.warning(f"Overlay GDF not found for key: {key_lower}")


    # --- Finalize Map ---
    if map_bounds:
        try:
            m.fit_bounds(map_bounds) # Fit map to the bounds of the final overlay
        except Exception as fit_e:
            logger.warning(f"Could not fit map bounds: {fit_e}")

    folium.LayerControl().add_to(m)

    # Save map
    try:
        m.save(str(output_html_path))
        logger.info(f"Combined overlay Folium map saved successfully to: {output_html_path}")
    except Exception as save_e:
        logger.error(f"Error saving combined Folium map: {save_e}", exc_info=True)


def display_vectors_on_folium_map(
    gdf: gpd.GeoDataFrame,
    output_html_path_str: str,
    style_column: Optional[str] = None,
    cmap_name: str = 'viridis',
    line_weight: int = 3,
    point_radius: int = 5,
    tooltip_columns: Optional[list] = None,
    popup_columns: Optional[list] = None,
    zoom_start: int = 12,
    tiles: str = 'CartoDB positron',
):
    """
    Displays vector features (Points, Lines) from a GeoDataFrame on a Folium map,
    optionally styling them based on a numeric column, and saves it as an HTML file.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame with vector features.
        output_html_path_str (str): Path to save the output HTML map.
        style_column (Optional[str]): Name of the numeric column to use for styling
                                      (color intensity). If None, uses default color.
        cmap_name (str): Name of the matplotlib colormap to use if styling. Defaults to 'viridis'.
        line_weight (int): Weight (thickness) for LineString features. Defaults to 3.
        point_radius (int): Radius for Point features. Defaults to 5.
        tooltip_columns (Optional[list]): List of column names to show in tooltips.
        popup_columns (Optional[list]): List of column names to show in popups.
        zoom_start (int): Initial zoom level for the map. Defaults to 12.
        tiles (str): Folium tile layer name. Defaults to 'CartoDB positron'.
    """
    # Import necessary libraries here
    try:
        import folium
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        from branca.colormap import LinearColormap # Direct import for color legend
        from pathlib import Path
    except ImportError as e:
        logger.error(f"Folium display skipped: Missing required library ({e}). Install folium, matplotlib, branca.")
        return

    output_html_path = Path(output_html_path_str)
    output_html_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    logger.info(f"Attempting to display {len(gdf)} vector features on Folium map.")

    if gdf.empty:
        logger.warning("Input GeoDataFrame is empty. Cannot create map.")
        return

    # Ensure data is in WGS84 (EPSG:4326) for Folium
    try:
        if gdf.crs is None:
            raise ValueError("Input GeoDataFrame has no CRS defined.")
        if gdf.crs.to_epsg() != 4326:
            logger.info(f"Reprojecting vector data from {gdf.crs} to EPSG:4326 for Folium.")
            gdf_4326 = gdf.to_crs(epsg=4326)
        else:
            gdf_4326 = gdf.copy() # Work on a copy
    except Exception as reproj_e:
        logger.error(f"Error reprojecting GeoDataFrame to EPSG:4326: {reproj_e}", exc_info=True)
        return

    logger.info(f"GDF shape after reprojection: {gdf_4326.shape}") # Log shape after reprojection

    # --- Log geometry status immediately after reprojection ---
    num_null_after_reproj = gdf_4326["geometry"].isna().sum()
    num_empty_after_reproj = gdf_4326.geometry.is_empty.sum()
    logger.info(f"Post-reprojection check: Null geometries = {num_null_after_reproj}, Empty geometries = {num_empty_after_reproj}")
    # --- End logging ---


    # --- Attempt to fix potential validity issues with buffer(0) ---
    try:
        original_count = len(gdf_4326)
        invalid_mask = ~gdf_4326.geometry.is_valid
        num_invalid = invalid_mask.sum()
        logger.info(f"Found {num_invalid} invalid geometries after reprojection (out of {original_count}).")

        if num_invalid > 0:
            logger.info(f"Attempting to fix {num_invalid} invalid geometries with buffer(0).")
            # Apply buffer(0) only to invalid geometries
            gdf_4326.loc[invalid_mask, 'geometry'] = gdf_4326.loc[invalid_mask].geometry.buffer(0)
            logger.info(f"Applied buffer(0) to {num_invalid} geometries.")
            logger.info(f"Shape after buffer(0): {gdf_4326.shape}")

            # Re-check validity after buffering
            fixed_mask = gdf_4326.loc[invalid_mask].geometry.is_valid
            num_fixed = fixed_mask.sum()
            num_still_invalid = num_invalid - num_fixed
            if num_still_invalid > 0:
                 logger.warning(f"{num_still_invalid} geometries remain invalid after buffer(0).")

        # Check if any geometries became None/empty after buffer(0) or were initially null
        null_empty_mask = gdf_4326["geometry"].isna() | gdf_4326.geometry.is_empty
        num_null_empty = null_empty_mask.sum()

        if num_null_empty > 0:
            logger.warning(f"{num_null_empty} geometries are null or empty after processing. Removing them.")
            gdf_4326 = gdf_4326[~null_empty_mask]

        logger.info(f"Shape after buffer(0) and null/empty removal: {gdf_4326.shape}")

    except Exception as buffer_e:
        logger.warning(f"Error during geometry fixing with buffer(0): {buffer_e}. Proceeding cautiously.")
        # Ensure we still drop obviously bad geometries before proceeding
        gdf_4326 = gdf_4326.dropna(subset=["geometry"])
        gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty]


    # --- Ensure valid geometries before calculating bounds ---
    # Drop rows with invalid or missing geometries *after* reprojection and fixing attempts
    # Re-check validity on the potentially modified gdf_4326
    gdf_4326_validated = gdf_4326[gdf_4326.geometry.is_valid].copy() # Use copy to avoid SettingWithCopyWarning later

    logger.info(f"GDF shape after geometry validation (is_valid check): {gdf_4326_validated.shape}") # Log shape after validation

    if gdf_4326_validated.empty:
        logger.warning("GeoDataFrame is empty after reprojection and geometry validation. Cannot create map.")
        return

    # --- Calculate map center and bounds ---
    # Use the validated GDF
    bounds = gdf_4326_validated.total_bounds # minx, miny, maxx, maxy

    # --- Check for invalid bounds ---
    if np.isnan(bounds).any() or np.isinf(bounds).any():
        logger.error(f"Invalid bounds calculated after reprojection: {bounds}. Cannot create map.")
        # Attempt to use a default location if bounds fail? Or just return.
        # For now, just return to avoid the NaN error in folium.Map
        return

    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # --- Create Folium map ---
    # Check if center coordinates are valid just in case
    if pd.isna(center_lat) or pd.isna(center_lon):
         logger.error(f"Calculated center coordinates are NaN ({center_lat}, {center_lon}). Cannot create map.")
         return

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=tiles,
    )

    # Prepare styling
    colormap = None
    norm = None
    # Use the validated GDF for styling checks
    if style_column and style_column in gdf_4326_validated.columns and pd.api.types.is_numeric_dtype(gdf_4326_validated[style_column]):
        try:
            values = gdf_4326_validated[style_column].dropna()
            if not values.empty:
                vmin = values.min()
                vmax = values.max()
                if vmin == vmax: # Handle constant value case
                    vmin -= 0.5
                    vmax += 0.5
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                cmap = cm.get_cmap(cmap_name)
                # Create a colormap for the legend using the directly imported class
                colormap = LinearColormap(
                    [colors.rgb2hex(cmap(norm(i))) for i in np.linspace(vmin, vmax, num=10)],
                    vmin=vmin, vmax=vmax,
                    caption=f"{style_column} Intensity"
                )
                m.add_child(colormap) # Add legend to map
            else:
                logger.warning(f"Style column '{style_column}' contains no valid numeric data. Using default style.")
                style_column = None # Fallback to default style
        except Exception as style_e:
            logger.error(f"Error setting up style based on column '{style_column}': {style_e}", exc_info=True)
            style_column = None # Fallback to default style
    elif style_column:
        logger.warning(f"Style column '{style_column}' not found or not numeric. Using default style.")
        style_column = None # Fallback to default style

    # Function to get color based on value
    def get_color(value):
        if style_column and norm and cmap and pd.notna(value):
            return colors.rgb2hex(cmap(norm(value)))
        return '#3388ff' # Default Folium blue

    # Add features to map
    # Using GeoJson for potentially better performance with large datasets
    # Prepare tooltip and popup fields
    tooltip = folium.features.GeoJsonTooltip(fields=tooltip_columns) if tooltip_columns else None
    popup = folium.features.GeoJsonPopup(fields=popup_columns) if popup_columns else None

    # Define style function for GeoJson
    def style_function(feature):
        style = {
            'color': get_color(feature['properties'].get(style_column)),
            'weight': line_weight,
            'opacity': 0.8,
            # 'fillColor': get_color(feature['properties'].get(style_column)), # Uncomment for polygons
            # 'fillOpacity': 0.5, # Uncomment for polygons
        }
        return style

    gdf_map_display = gdf_4326_validated.copy() # Work on a copy for display modifications
    # drop 'points' column due to HTML errors
    if 'points' in gdf_map_display.columns:
        gdf_map_display = gdf_map_display.drop(columns=['points'])
        logger.info("Dropped 'points' column from GeoDataFrame for Folium display.")

    # Add GeoJson layer
    # Use the cleaned GDF copy
    geojson_layer = folium.GeoJson(
        gdf_map_display, # Use the cleaned data
        style_function=style_function,
        tooltip=tooltip,
        popup=popup,
        name="Vector Features" # Layer name
    )
    geojson_layer.add_to(m)

    # Add Layer Control
    folium.LayerControl().add_to(m)

    # Save map
    try:
        m.save(str(output_html_path))
        logger.info(f"Folium map with vector features saved to: {output_html_path}")
    except Exception as save_e:
        logger.error(f"Error saving Folium map to {output_html_path}: {save_e}", exc_info=True)


def display_dem_slope_on_folium_map(
    raster_path_str: str,
    output_html_path_str: str,
    target_crs_epsg: int,
    layer_name: Optional[str] = None, # Added layer_name parameter
    nodata_transparent: bool = True,
    cmap_name: str = 'viridis',
    opacity: float = 0.7,
    zoom_start: int = 12,
    tiles: str = 'CartoDB positron',
):
    """
    Displays a DEM or Slope raster layer on a Folium map and saves it as an HTML file.
    Allows specifying a layer name for the overlay.

    Args:
        raster_path_str (str): Path to the input raster file.
        output_html_path_str (str): Path to save the output HTML map.
        target_crs_epsg (int): The expected EPSG code of the input raster's CRS.
                               Used for verification and bounds transformation.
        layer_name (Optional[str]): Name for the raster layer in the LayerControl.
                                    Defaults to the raster filename stem if None.
        nodata_transparent (bool): Whether to make NoData pixels transparent. Defaults to True.
        cmap_name (str): Name of the matplotlib colormap to use. Defaults to 'viridis'.
        opacity (float): Opacity of the raster layer (0.0 to 1.0). Defaults to 0.7.
        zoom_start (int): Initial zoom level for the map. Defaults to 12.
        tiles (str): Folium tile layer name. Defaults to 'CartoDB positron'.
    """
    # Import necessary libraries here to avoid top-level dependency
    try:
        import folium
        import rasterio
        import rasterio.warp
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import io
        import base64
        from pathlib import Path
    except ImportError as e:
        logger.error(f"Folium display skipped: Missing required library ({e}). Install folium, matplotlib, rasterio.")
        return

    raster_path = Path(raster_path_str)
    output_html_path = Path(output_html_path_str)
    output_html_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # Use filename stem as default layer name if not provided
    if layer_name is None:
        layer_name = raster_path.stem
        logger.info(f"Layer name not provided, using default: {layer_name}")

    logger.info(f"Attempting to display raster: {raster_path} as layer '{layer_name}' on Folium map.")

    if not raster_path.is_file():
        logger.error(f"Raster file not found: {raster_path}")
        return

    try:
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            raster_crs = src.crs
            logger.info(f"Folium Display: Raster CRS read from file: {raster_crs}")
            data = src.read(1, masked=True) # Read first band as masked array

            # Verify the CRS before transforming
            if not raster_crs:
                logger.error(
                    f"Folium Display: Missing CRS ({raster_crs}) read from {raster_path}. Cannot transform bounds accurately."
                )
                return # Cannot proceed without valid CRS
            elif raster_crs.to_epsg() != target_crs_epsg:
                 logger.warning(
                    f"Folium Display: CRS read from raster ({raster_crs}) does not match expected EPSG:{target_crs_epsg}. Transformation might be incorrect."
                )
                # Proceed with caution

            # Transform bounds to WGS84 (EPSG:4326) for Folium
            logger.info(f"Folium Display: Original bounds ({raster_crs}): {bounds}")
            bounds_4326 = None
            try:
                bounds_4326 = rasterio.warp.transform_bounds(
                    raster_crs, "EPSG:4326", *bounds # left, bottom, right, top
                )
                logger.info(f"Folium Display: Transformed bounds (EPSG:4326): {bounds_4326}")
            except Exception as warp_e:
                logger.error(f"Folium Display: Error during bounds transformation: {warp_e}", exc_info=True)
                return # Cannot proceed without transformed bounds

            # Prepare data for image overlay
            cmap = plt.get_cmap(cmap_name) # Use argument
            valid_data = data.compressed() # Get only valid (unmasked) data
            rgba_data_calc = None
            if valid_data.size > 0:
                # Normalize based on valid data range (e.g., 5th to 95th percentile for better contrast)
                vmin = np.percentile(valid_data, 5)
                vmax = np.percentile(valid_data, 95)
                # Handle case where vmin == vmax (e.g., constant raster)
                if vmin == vmax:
                    norm = colors.Normalize(vmin=vmin - 1e-6, vmax=vmax + 1e-6) # Avoid division by zero
                else:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)

                rgba_data_calc = cmap(norm(data), bytes=True) # Apply colormap

                # Make masked pixels transparent if requested
                if nodata_transparent and data.mask.any():
                    rgba_data_calc[data.mask] = (0, 0, 0, 0)
            else:
                logger.warning("Raster data contains no valid pixels. Cannot create image overlay.")
                return # Cannot create overlay without data

            # Check if we have valid pixel data (rgba_data_calc) and transformed bounds
            if rgba_data_calc is not None and bounds_4326 is not None:
                map_bounds_folium = [
                    [bounds_4326[1], bounds_4326[0]], # SouthWest (lat, lon)
                    [bounds_4326[3], bounds_4326[2]], # NorthEast (lat, lon)
                ]
                center_lat = (bounds_4326[1] + bounds_4326[3]) / 2
                center_lon = (bounds_4326[0] + bounds_4326[2]) / 2

                # Save RGBA data to a PNG in memory
                buf = io.BytesIO()
                plt.imsave(buf, rgba_data_calc, format="png")
                buf.seek(0)
                png_base64 = base64.b64encode(buf.read()).decode("utf-8")
                img_uri = f"data:image/png;base64,{png_base64}"

                # Create Folium map
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=zoom_start, # Use argument
                    tiles=tiles, # Use argument
                )

                # Add raster as ImageOverlay using the provided layer_name
                img_overlay = folium.raster_layers.ImageOverlay(
                    image=img_uri,
                    bounds=map_bounds_folium,
                    opacity=opacity, # Use argument
                    name=layer_name, # Use the layer_name parameter here
                )
                img_overlay.add_to(m)
                folium.LayerControl().add_to(m)

                # Save map
                m.save(str(output_html_path))
                logger.info(f"Folium map with raster layer '{layer_name}' saved to: {output_html_path}")
            else:
                # This condition should ideally be caught earlier
                logger.warning("Skipping Folium map generation due to missing data or bounds.")

    except rasterio.RasterioIOError as rio_e:
        logger.error(f"Error opening or reading raster file {raster_path}: {rio_e}", exc_info=True)
    except Exception as map_e:
        logger.error(f"Error generating Folium map for {raster_path}: {map_e}", exc_info=True)


def display_multi_layer_on_folium_map(
    layers: List[Dict[str, Any]],
    output_html_path_str: str,
    map_center: Optional[Tuple[float, float]] = None,
    map_zoom: int = 8,
    map_tiles: str = 'CartoDB positron',
):
    """
    Displays multiple vector and raster layers on a Folium map with layer control.
    NOTE: Folium expects all layers to be in EPSG:4326 (WGS84) for proper display.
    This funciton includes inline reprojection to EPSG:4326 on mismatch.

    Args:
        layers (List[Dict[str, Any]]): A list of dictionaries, each defining a layer.
            Required keys per layer:
                'path': Path object or string to the data file.
                'name': String name for the layer in the LayerControl.
                'type': String, either 'vector' or 'raster'.
            Optional keys:
                'vector': {
                    'style_column': String, column for numeric styling (optional).
                    'cmap': String, matplotlib colormap name (default 'viridis').
                    'color': String, hex color if not using style_column (default '#3388ff').
                    'weight': Int, line weight (default 3).
                    'radius': Int, point radius (default 5).
                    'tooltip_cols': List[str], columns for tooltip (optional).
                    'popup_cols': List[str], columns for popup (optional).
                    'show': Bool, whether layer is visible initially (default True).
                }
                'raster': {
                    'cmap': String, matplotlib colormap name (default 'viridis').
                    'opacity': Float, layer opacity (default 0.7).
                    'nodata_transparent': Bool, make nodata transparent (default True).
                    'show': Bool, whether layer is visible initially (default True).
                    'target_crs_epsg': Int, expected source CRS for bounds transform (required if type is raster).
                }
        output_html_path_str (str): Path to save the output HTML map.
        map_center (Optional[Tuple[float, float]]): Initial map center (latitude, longitude).
                                                    If None, calculated from layer bounds.
        map_zoom (int): Initial map zoom level. Defaults to 12.
        map_tiles (str): Folium tile layer name. Defaults to 'CartoDB positron'.
        folium_crs (str): The CRS Folium uses internally (should generally be 'EPSG4326').
    """
    import folium
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    
    logger.info(f"Generating multi-layer Folium map: {output_html_path_str}")
    output_html_path = Path(output_html_path_str)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)

    all_bounds_4326 = [] # To store bounds of all layers in EPSG:4326

    # --- Create Base Map ---
    m = folium.Map(
        location=[59.9139, 10.7522],  # Oslo (59.9139, 10.7522) as default
        zoom_start=map_zoom, 
        tiles=map_tiles,
        # crs='EPSG4326', # WARNING: Setting crs in Folium causes unexpected behavior
    )

    # --- Process and Add Layers ---
    for layer_config in layers:
        layer_path = Path(layer_config['path'])
        layer_name = layer_config['name']
        layer_type = layer_config['type'].lower()
        layer_options = layer_config.get(layer_type, {}) # Get type-specific options
        show_layer = layer_options.get('show', True) # Default to visible

        logger.info(f"Processing layer '{layer_name}' ({layer_type}) from: {layer_path}")

        if not layer_path.exists():
            logger.warning(f"Skipping layer '{layer_name}': File not found at {layer_path}")
            continue

        # Create a FeatureGroup for each layer for toggling
        feature_group = folium.FeatureGroup(name=layer_name, show=show_layer)

        try:
            # --- Handle Vector Layers ---
            if layer_type == 'vector':
                gdf = load_vector_data(layer_path) # Assumes load_vector_data handles errors
                if gdf.empty:
                    logger.warning(f"Skipping layer '{layer_name}': Loaded GeoDataFrame is empty.")
                    continue

                # Reproject to Folium's CRS (EPSG:4326)
                if gdf.crs is None:
                    logger.warning(f"Skipping layer '{layer_name}': GeoDataFrame has no CRS defined.")
                    continue
                if gdf.crs.to_epsg() != 4326:
                    gdf_4326 = gdf.to_crs(epsg=4326)
                    logger.warning(f"Had to reproject layer from {gdf.crs} to EPSG:4326.")
                else:
                    gdf_4326 = gdf.copy()

                # Clean geometries (null, empty, invalid with buffer(0))
                gdf_4326 = gdf_4326.dropna(subset=['geometry'])
                gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty]
                invalid_mask = ~gdf_4326.geometry.is_valid
                if invalid_mask.sum() > 0:
                    gdf_4326.loc[invalid_mask, 'geometry'] = gdf_4326.loc[invalid_mask].geometry.buffer(0)
                    gdf_4326 = gdf_4326.dropna(subset=['geometry']) # Re-drop if buffer failed
                    gdf_4326 = gdf_4326[~gdf_4326.geometry.is_empty]

                if gdf_4326.empty:
                    logger.warning(f"Skipping layer '{layer_name}': GeoDataFrame empty after cleaning.")
                    continue
                else:
                    logger.info(f"Layer has {len(gdf_4326)} valid geometries after cleaning.")
                    logger.info(f"Columns: {gdf_4326.columns.tolist()}")

                # Store bounds for map extent calculation
                layer_bounds = gdf_4326.total_bounds
                if not (np.isnan(layer_bounds).any() or np.isinf(layer_bounds).any()):
                    # Convert minx, miny, maxx, maxy to [[miny, minx], [maxy, maxx]]
                    all_bounds_4326.append([[layer_bounds[1], layer_bounds[0]], [layer_bounds[3], layer_bounds[2]]])

                # Styling
                style_col = layer_options.get('style_column')
                cmap_name = layer_options.get('cmap', 'viridis')
                default_color = layer_options.get('color', '#3388ff')
                weight = layer_options.get('weight', 3)
                radius = layer_options.get('radius', 5) # For points
                tooltip_cols = layer_options.get('tooltip_cols')
                popup_cols = layer_options.get('popup_cols')

                # Prepare styling function and colormap if needed
                style_norm = None
                style_cmap = None
                style_vmin = None
                style_vmax = None
                if style_col and style_col in gdf_4326.columns and pd.api.types.is_numeric_dtype(gdf_4326[style_col]):
                    values = gdf_4326[style_col].dropna()
                    if not values.empty:
                        style_vmin = values.min()
                        style_vmax = values.max()
                        if style_vmin == style_vmax: 
                            style_vmin -= 0.5
                            style_vmax += 0.5
                        style_norm = colors.Normalize(vmin=style_vmin, vmax=style_vmax)
                        style_cmap = plt.get_cmap(cmap_name)
                    else: 
                        style_col = None # Fallback

                def vector_style_func(feature, scol=style_col, norm=style_norm, cmap=style_cmap, dcol=default_color, lweight=weight):
                    val = feature['properties'].get(scol)
                    color = dcol
                    if scol and norm and cmap and pd.notna(val):
                        color = colors.rgb2hex(cmap(norm(val)))
                    return {
                        'color': color,
                        'weight': lweight,
                        'fillColor': color, # For points/polygons
                        'fillOpacity': 0.6, # For points/polygons
                        'opacity': 0.8,
                    }

                # Add GeoJson to the feature group
                # Drop potentially problematic columns before creating GeoJson
                cols_to_drop = ['points'] # Add others if needed
                gdf_display = gdf_4326.drop(columns=[c for c in cols_to_drop if c in gdf_4326.columns])

                # Handle field name truncation in shapefiles - check actual available columns
                # ESRI Shapefile format truncates column names to 10 characters
                
                geometry_col_name = gdf_display.geometry.name # Get the geometry column name
                logger.debug(f"Identified geometry column as: '{geometry_col_name}' for layer '{layer_name}'")

                actual_tooltip_cols = []
                actual_popup_cols = []
                
                if tooltip_cols:
                    # Create a mapping from truncated to original column names
                    available_cols = set(gdf_display.columns)
                    # Try to match truncated names
                    for col in tooltip_cols:
                        if col == geometry_col_name:
                            logger.debug(f"Skipping geometry column '{col}' for tooltip fields in layer '{layer_name}'.")
                            continue
                        if col in available_cols:
                            actual_tooltip_cols.append(col)
                        elif len(col) > 10:
                            # Try to find the truncated version
                            truncated_col = col[:10]
                            if truncated_col in available_cols:
                                logger.info(f"Using truncated column name '{truncated_col}' instead of '{col}' for tooltip")
                                actual_tooltip_cols.append(truncated_col)
                            else:
                                logger.warning(f"Column '{col}' not found, even with truncation to '{truncated_col}'. Skipping.")
                        else:
                            logger.warning(f"Column '{col}' not found in the data. Skipping.")
                
                # Same for popup columns
                if popup_cols:
                    available_cols = set(gdf_display.columns)
                    for col in popup_cols:
                        if col == geometry_col_name:
                            logger.debug(f"Skipping geometry column '{col}' for popup fields in layer '{layer_name}'.")
                            continue
                        if col in available_cols:
                            actual_popup_cols.append(col)
                        elif len(col) > 10:
                            truncated_col = col[:10]
                            if truncated_col in available_cols:
                                logger.info(f"Using truncated column name '{truncated_col}' instead of '{col}' for popup")
                                actual_popup_cols.append(truncated_col)
                            else:
                                logger.warning(f"Column '{col}' not found, even with truncation to '{truncated_col}'. Skipping.")
                        else:
                            logger.warning(f"Column '{col}' not found in the data. Skipping.")

                if actual_tooltip_cols:
                    logger.info(f"Using tooltip columns: {actual_tooltip_cols}")
                    tooltip = folium.features.GeoJsonTooltip(fields=actual_tooltip_cols)
                else:
                    tooltip = None
                    
                if actual_popup_cols:
                    logger.info(f"Using popup columns: {actual_popup_cols}")
                    popup = folium.features.GeoJsonPopup(fields=actual_popup_cols)
                else:
                    popup = None

                logger.info(f"Columns right before foliumn.GeoJson: {gdf_display.columns.tolist()}")
                folium.GeoJson(
                    gdf_display,
                    style_function=vector_style_func,
                    tooltip=tooltip,
                    popup=popup,
                    marker=folium.CircleMarker(radius=radius, fill=True), # Style points
                    name=layer_name # Name for the sub-layer within the group (optional)
                ).add_to(feature_group)

                # Add colormap legend if styled numerically
                if style_col and style_cmap and style_vmin is not None:
                    try:
                        from branca.colormap import LinearColormap
                        colors_list = [colors.rgb2hex(style_cmap(style_norm(i))) for i in np.linspace(style_vmin, style_vmax, num=10)]
                        colormap = LinearColormap(
                            colors=colors_list,
                            vmin=style_vmin,
                            vmax=style_vmax,
                            caption=f"{layer_name}: {style_col}"
                        )
                        # Add legend outside the feature group so it's always visible
                        m.add_child(colormap)
                    except Exception as legend_e:
                        logger.warning(f"Could not create legend for layer '{layer_name}': {legend_e}")


            # --- Handle Raster Layers ---
            elif layer_type == 'raster':
                target_crs_epsg = "EPSG:4326"  # TODO clean inline 
                if target_crs_epsg is None:
                    logger.warning(f"Skipping raster layer '{layer_name}': 'target_crs_epsg' missing in raster options.")
                    continue

                cmap_name = layer_options.get('cmap', 'viridis')
                opacity = layer_options.get('opacity', 0.7)
                nodata_transparent = layer_options.get('nodata_transparent', True)

                with rasterio.open(layer_path) as src:
                    r_bounds = src.bounds
                    r_crs = src.crs
                    r_data = src.read(1, masked=True)

                    if not r_crs:
                        logger.warning(f"Skipping raster '{layer_name}': Missing CRS.")
                        continue
                    if int(r_crs.to_epsg()) != int(target_crs_epsg.split(":")[-1]):
                        logger.error(f"Raster '{layer_name}' CRS ({r_crs}) differs from expected target EPSG:{target_crs_epsg}.")
                        logger.warning("Resulting visuals will be warped.")

                    # Transform bounds to WGS84
                    r_bounds_4326 = rasterio.warp.transform_bounds(r_crs, "EPSG:4326", *r_bounds)
                    r_map_bounds = [[r_bounds_4326[1], r_bounds_4326[0]], [r_bounds_4326[3], r_bounds_4326[2]]] # SW, NE

                    # Store bounds for map extent calculation
                    if not (np.isnan(r_bounds_4326).any() or np.isinf(r_bounds_4326).any()):
                         all_bounds_4326.append(r_map_bounds)

                    # Prepare image data
                    cmap = plt.get_cmap(cmap_name)
                    valid_data = r_data.compressed()
                    if valid_data.size == 0:
                        logger.warning(f"Skipping raster '{layer_name}': No valid data.")
                        continue

                    vmin = np.percentile(valid_data, 5)
                    vmax = np.percentile(valid_data, 95)
                    if vmin == vmax: 
                        vmin, vmax = vmin - 1e-6, vmax + 1e-6
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                    rgba_data = cmap(norm(r_data), bytes=True)
                    if nodata_transparent and r_data.mask.any():
                        rgba_data[r_data.mask] = (0, 0, 0, 0)

                    # Convert to base64 PNG
                    buf = io.BytesIO()
                    plt.imsave(buf, rgba_data, format="png")
                    img_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                    buf.close()

                    # Add ImageOverlay to the feature group
                    folium.raster_layers.ImageOverlay(
                        image=img_uri,
                        bounds=r_map_bounds,
                        opacity=opacity,
                        name=layer_name, # Name for the sub-layer (optional)
                    ).add_to(feature_group)

            else:
                logger.warning(f"Skipping layer '{layer_name}': Unknown layer type '{layer_type}'. Use 'vector' or 'raster'.")
                continue

            # Add the completed FeatureGroup to the map
            feature_group.add_to(m)
            logger.info(f"Added layer '{layer_name}' to map feature group.")

        except Exception as e:
            logger.error(f"Failed to process layer '{layer_name}' from {layer_path}: {e}", exc_info=True)

    # --- Finalize Map ---
    # Calculate overall bounds and set map view
    if map_center is None:
        if all_bounds_4326:
            # Calculate the total bounds encompassing all layers
            min_lat = min(b[0][0] for b in all_bounds_4326)
            min_lon = min(b[0][1] for b in all_bounds_4326)
            max_lat = max(b[1][0] for b in all_bounds_4326)
            max_lon = max(b[1][1] for b in all_bounds_4326)
            final_bounds = [[min_lat, min_lon], [max_lat, max_lon]]
            try:
                m.fit_bounds(final_bounds)
            except Exception as fit_e:
                 logger.warning(f"Could not fit map bounds automatically: {fit_e}. Using default zoom.")
                 # Calculate center manually if fit_bounds failed
                 center_lat = (min_lat + max_lat) / 2
                 center_lon = (min_lon + max_lon) / 2
                 if not (pd.isna(center_lat) or pd.isna(center_lon)):
                     m.location = [center_lat, center_lon]
                     m.zoom_start = map_zoom # Use default zoom if center is valid
                 else: # Fallback if center calculation also failed
                     m.location = [59.9139, 10.7522] # Oslo default
                     m.zoom_start = map_zoom
        else:
            logger.warning("No valid layer bounds found. Using default map center and zoom.")
            m.location = [59.9139, 10.7522] # Oslo default
            m.zoom_start = map_zoom
    else:
        # Use provided center and zoom
        m.location = map_center
        m.zoom_start = map_zoom

    # Add Layer Control to toggle FeatureGroups
    folium.LayerControl().add_to(m)

    # Save map
    try:
        m.save(str(output_html_path))
        logger.info(f"Multi-layer Folium map saved successfully to: {output_html_path}")
    except Exception as save_e:
        logger.error(f"Error saving multi-layer Folium map: {save_e}", exc_info=True)
