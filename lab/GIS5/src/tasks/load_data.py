import geopandas as gpd
import warnings
import logging
from typing import Tuple, Dict, Optional, Any
from pathlib import Path

from pydantic import BaseModel
from pyproj import CRS

try:
    from src.utils import get_common_extent_and_crs, find_elevation_field
except ImportError:
    # Allow running as a script for potential testing/debugging
    from utils import get_common_extent_and_crs, find_elevation_field

# Get a logger for this module
logger = logging.getLogger(__name__)


class FeatureDataLoader:
    """
    Handles loading and initial preparation of input vector data layers.
    """

    def __init__(self, settings: BaseModel):
        """
        Initializes the data loader.

        Args:
            settings: The application configuration object.
        """
        self.settings = settings
        self.paths = settings.paths
        self.input_layers = settings.input_layers
        self.output_files = settings.output_files
        self.output_dir = self.paths.output_dir
        self.gdb_path = str(self.paths.gdb_path)  # Geopandas might prefer string

        self.loaded_gdfs: Dict[str, gpd.GeoDataFrame] = {}
        self.common_extent: Optional[Tuple[float, float, float, float]] = None
        self.common_crs: Optional[CRS] = None
        self.contour_elev_field: Optional[str] = None
        self.point_elev_field: Optional[str] = None

    def _log(self, message: str, level: str = "info", indent: int = 1):
        """Helper for logging messages with indentation."""
        indent_str = "  " * indent
        level_map = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "debug": logging.DEBUG,
        }
        log_level = level_map.get(level.lower(), logging.INFO)
        logger.log(log_level, f"{indent_str}{message}")

    def _read_layers(self) -> bool:
        """Reads vector layers from the Geodatabase."""
        self._log(f"Reading data from: {self.gdb_path}", indent=0)
        try:
            contours_gdf = gpd.read_file(
                self.gdb_path, layer=self.input_layers.contour_layer
            )
            rivers_gdf = gpd.read_file(
                self.gdb_path, layer=self.input_layers.river_layer
            )
            lakes_gdf = gpd.read_file(self.gdb_path, layer=self.input_layers.lake_layer)
            points_gdf = gpd.read_file(
                self.gdb_path, layer=self.input_layers.points_layer
            )

            self._log(f"Read {len(contours_gdf)} contours.")
            self._log(f"Read {len(rivers_gdf)} river segments.")
            self._log(f"Read {len(lakes_gdf)} lakes.")
            self._log(f"Read {len(points_gdf)} elevation points.")

            self.loaded_gdfs = {
                "contours": contours_gdf,
                "rivers": rivers_gdf,
                "lakes": lakes_gdf,
                "points": points_gdf,
            }
            return True
        except ImportError as ie:
            self._log(f"Error reading data: {ie}", level="error", indent=0)
            self._log(
                "Ensure the necessary driver (e.g., GDAL FileGDB driver) is correctly installed and accessible.",
                level="error",
                indent=1,
            )
            return False
        except Exception as e:
            self._log(
                f"An unexpected error occurred during layer reading: {e}",
                level="error",
                indent=0,
                exc_info=True,
            )
            return False

    def _calculate_common_extent(self) -> bool:
        """Calculates the common extent and CRS of the loaded layers."""
        if not self.loaded_gdfs:
            self._log("No layers loaded, cannot calculate extent.", level="error")
            return False

        self._log("Calculating common extent and CRS...")
        try:
            all_layers_list = list(self.loaded_gdfs.values())
            minx, miny, maxx, maxy, common_crs = get_common_extent_and_crs(
                all_layers_list
            )
            self._log(f"Common CRS: {common_crs}")
            self._log(
                f"Combined Extent: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})"
            )
            self.common_extent = (minx, miny, maxx, maxy)
            self.common_crs = common_crs
            return True
        except Exception as e:
            self._log(
                f"Failed to calculate common extent/CRS: {e}",
                level="error",
                exc_info=True,
            )
            return False

    def _save_shapefiles(self) -> bool:
        """Saves intermediate copies of the loaded layers as Shapefiles."""
        if not self.loaded_gdfs:
            self._log("No layers loaded, cannot save shapefiles.", level="error")
            return False

        self._log("Saving intermediate Shapefiles...")
        try:
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
                self.loaded_gdfs["contours"].to_file(
                    self.output_files.get_full_path("contour_shp", self.output_dir)
                )
                self.loaded_gdfs["rivers"].to_file(
                    self.output_files.get_full_path("river_shp", self.output_dir)
                )
                self.loaded_gdfs["lakes"].to_file(
                    self.output_files.get_full_path("lake_shp", self.output_dir)
                )
                self.loaded_gdfs["points"].to_file(
                    self.output_files.get_full_path("points_shp", self.output_dir)
                )
            self._log("...Shapefile saving done.")
            return True
        except Exception as e:
            self._log(f"Failed to save shapefiles: {e}", level="error", exc_info=True)
            return False

    def _identify_elevation_fields(self) -> bool:
        """Identifies the elevation fields in contour and points layers."""
        if "contours" not in self.loaded_gdfs or "points" not in self.loaded_gdfs:
            self._log(
                "Contour or points layer not loaded, cannot identify elevation fields.",
                level="error",
            )
            return False

        self._log("Identifying elevation fields...")
        try:
            # Contour Elevation Field
            contours_gdf = self.loaded_gdfs["contours"]
            contour_elev_field = find_elevation_field(
                contours_gdf, [self.input_layers.contour_elevation_field]
            )
            if contour_elev_field is None:
                # Try candidates if primary fails
                contour_elev_field = find_elevation_field(
                    contours_gdf, self.input_layers.point_elevation_field_candidates
                )
                if contour_elev_field is None:
                    self._log(
                        f"Could not find contour elevation field (tried '{self.input_layers.contour_elevation_field}' and candidates) in {self.input_layers.contour_layer}.",
                        level="error",
                    )
                    return False  # Critical failure
                else:
                    self._log(
                        f"Configured contour field '{self.input_layers.contour_elevation_field}' not found, using '{contour_elev_field}' instead.",
                        level="warning",
                    )
            else:
                self._log(f"Using contour elevation field: '{contour_elev_field}'")
            self.contour_elev_field = contour_elev_field

            # Point Elevation Field
            points_gdf = self.loaded_gdfs["points"]
            point_elev_field = find_elevation_field(
                points_gdf, self.input_layers.point_elevation_field_candidates
            )
            if point_elev_field is None:
                self._log(
                    "Could not identify elevation field in points layer using candidates.",
                    level="warning",
                )
                # Allow workflow to continue, but RMSE might fail later
            else:
                self._log(f"Using point elevation field: '{point_elev_field}'")
            self.point_elev_field = point_elev_field  # Store even if None

            return True
        except Exception as e:
            self._log(
                f"Error identifying elevation fields: {e}", level="error", exc_info=True
            )
            return False

    def load(
        self,
    ) -> Tuple[
        Optional[Dict[str, gpd.GeoDataFrame]],
        Optional[Tuple[float, float, float, float]],
        Optional[CRS],
        Optional[str],
        Optional[str],
    ]:
        """
        Orchestrates the data loading and preparation steps.

        Returns:
            A tuple containing:
            - A dictionary of loaded GeoDataFrames or None on failure.
            - A tuple representing the common extent or None on failure.
            - The common CRS object or None on failure.
            - The identified contour elevation field name or None on failure.
            - The identified point elevation field name (can be None if not found).
            Returns (None, None, None, None, None) if a critical step fails.
        """
        logger.info("--- Starting Data Loading and Preparation ---")

        if not self._read_layers():
            logger.error("--- Data Loading Failed (Layer Reading) ---")
            return None, None, None, None, None

        if not self._calculate_common_extent():
            logger.error("--- Data Loading Failed (Extent Calculation) ---")
            return None, None, None, None, None

        if not self._save_shapefiles():
            # Log error but potentially continue if needed elsewhere? For now, treat as failure.
            logger.error("--- Data Loading Failed (Shapefile Saving) ---")
            return None, None, None, None, None

        if not self._identify_elevation_fields():
            # Log error but return potentially partial results if needed?
            # For now, treat as failure if contour field is missing.
            logger.error("--- Data Loading Failed (Elevation Field Identification) ---")
            # If contour field is essential, fail completely
            if self.contour_elev_field is None:
                return None, None, None, None, None
            # Otherwise, maybe allow continuation with warning? Let's stick to stricter failure for now.
            # return None, None, None, None, None # Re-enable if strict failure needed

        logger.info("--- Data Loading and Preparation Complete ---")
        return (
            self.loaded_gdfs,
            self.common_extent,
            self.common_crs,
            self.contour_elev_field,
            self.point_elev_field,  # Return the identified field (or None)
        )


def load_and_prepare_data(
    settings: BaseModel,
) -> Tuple[
    Optional[Dict[str, gpd.GeoDataFrame]],
    Optional[Tuple[float, float, float, float]],
    Optional[CRS],
    Optional[str],
    Optional[str],
]:
    """
    Loads input vector data, calculates common extent/CRS, saves intermediate
    shapefiles, and identifies elevation fields using the FeatureDataLoader class.

    Args:
        settings: The application configuration object.

    Returns:
        A tuple containing:
        - A dictionary of loaded GeoDataFrames (or None on failure).
        - A tuple representing the common extent (or None on failure).
        - The common CRS object (or None on failure).
        - The identified contour elevation field name (or None on failure).
        - The identified point elevation field name (or None if not found/error).

    Raises:
        Catches exceptions within the loader, logs them, and returns None tuple
        to indicate failure, preventing crashes in the calling workflow.
    """
    try:
        loader = FeatureDataLoader(settings=settings)
        result = loader.load()
        # Check if a critical step failed (indicated by None for essential outputs)
        if (
            result[0] is None
            or result[1] is None
            or result[2] is None
            or result[3] is None
        ):
            # Raise an exception if the calling code expects one on failure
            # raise RuntimeError("Data loading failed. Check logs for details.")
            # Or return the None tuple to signal failure without crashing
            return None, None, None, None, None
        return result
    except Exception as e:
        # Catch any unexpected error during instantiation or the load call itself
        logger.error(
            f"An unexpected error occurred in the load_and_prepare_data wrapper: {e}",
            exc_info=True,
        )
        # Raise the exception upwards if the workflow needs to handle it explicitly
        # raise
        # Or return None tuple to indicate failure
        return None, None, None, None, None
