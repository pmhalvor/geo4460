import os
import shutil
import geopandas as gpd
from pathlib import Path
from typing import List, Tuple, Optional
from pyproj import CRS  # For CRS handling


def setup_output_dir(directory: Path):
    """Creates or clears the output directory."""
    dir_str = str(directory)  # Use string representation for os.path and shutil
    if os.path.exists(dir_str):
        print(f"Removing existing output directory: {dir_str}")
        shutil.rmtree(dir_str)
    print(f"Creating output directory: {dir_str}")
    os.makedirs(dir_str)


def get_common_extent_and_crs(
    layers: List[gpd.GeoDataFrame],
) -> Tuple[float, float, float, float, Optional[CRS]]:
    """
    Calculates the total bounds of a list of GeoDataFrames and ensures they share a common CRS.

    Args:
        layers: A list of GeoDataFrames.

    Returns:
        A tuple containing (minx, miny, maxx, maxy, common_crs).
        Returns None for CRS if the first layer has no CRS set.

    Raises:
        ValueError: If the input list is empty or if layers cannot be reprojected.
    """
    if not layers:
        raise ValueError("Need at least one layer to determine extent and CRS.")

    first_layer = layers[0]
    common_crs = first_layer.crs

    if common_crs is None:
        print(
            "Warning: First input layer CRS is None. Cannot ensure CRS consistency or reproject."
        )
        # Proceed with bounds calculation, but CRS will be None

    total_bounds = first_layer.total_bounds
    minx, miny, maxx, maxy = total_bounds

    for i, layer in enumerate(layers[1:], start=1):
        layer_crs = layer.crs
        if common_crs is not None and layer_crs != common_crs:
            print(
                f"Warning: CRS mismatch found in layer {i+1}. Expected {common_crs}, got {layer_crs}. Reprojecting..."
            )
            try:
                layer = layer.to_crs(common_crs)
                print(f"  - Layer {i+1} reprojected successfully.")
            except Exception as e:
                raise ValueError(
                    f"Failed to reproject layer {i+1} to {common_crs}: {e}"
                )
        elif common_crs is None and layer_crs is not None:
            print(
                f"Warning: First layer had no CRS, but layer {i+1} has CRS {layer_crs}. Cannot guarantee consistency."
            )
            # In this case, we can't reliably reproject, so we just use the bounds as-is

        # Update total bounds
        bounds = layer.total_bounds
        minx = min(minx, bounds[0])
        miny = min(miny, bounds[1])
        maxx = max(maxx, bounds[2])
        maxy = max(maxy, bounds[3])

    return minx, miny, maxx, maxy, common_crs


def find_elevation_field(
    gdf: gpd.GeoDataFrame, field_candidates: List[str]
) -> Optional[str]:
    """
    Attempts to find a suitable elevation field from a list of candidates.

    Args:
        gdf: The GeoDataFrame to search within.
        field_candidates: A list of potential field names.

    Returns:
        The name of the found field, or None if no candidate is found.
    """
    for field in field_candidates:
        if field in gdf.columns:
            print(
                f"Elevation field found: '{field}'. Sample values: {gdf[field].head().tolist()}"
            )
            return field

    print(
        f"Warning: Could not automatically identify elevation field from candidates: {field_candidates}."
    )
    print(f"Available columns: {gdf.columns.tolist()}")
    return None
