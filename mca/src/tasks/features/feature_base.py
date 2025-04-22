import logging
from abc import ABC, abstractmethod
from pathlib import Path

import geopandas as gpd
from whitebox import WhiteboxTools

logger = logging.getLogger(__name__)  # Define logger earlier

# Assuming utils are in ../../src relative to this file
# Adjust relative path if needed based on execution context
try:
    from src.config import AppConfig
    from src.utils import save_vector_data, save_raster_data, reproject_gdf
except ImportError:
    # Fallback for potential execution from different relative paths
    # This might happen if running tests or individual scripts directly
    logger.warning(
        "Could not import from src.* directly, attempting relative import..."
    )
    from ...config import AppConfig
    from ...utils import save_vector_data, save_raster_data, reproject_gdf


class FeatureBase(ABC):
    """Abstract base class for feature generation."""

    def __init__(self, settings: AppConfig, wbt: WhiteboxTools):
        self.settings = settings
        self.wbt = wbt
        self.gdf: gpd.GeoDataFrame | None = None  # Loaded and preprocessed data
        self.output_paths: dict = {}  # To store paths of generated outputs

    @abstractmethod
    def load_data(self):
        """Load and preprocess data specific to the feature."""
        pass

    @abstractmethod
    def build(self):
        """Build the feature layer(s) (potentially using Dask)."""
        pass

    def _get_output_path(self, key: str) -> Path:
        """Helper to get a full output path from settings."""
        filename = getattr(self.settings.output_files, key)
        return self.settings.paths.output_dir / filename

    def _save_intermediate_gdf(self, gdf: gpd.GeoDataFrame, output_key: str):
        """Saves an intermediate GeoDataFrame."""
        if gdf is None or gdf.empty:
            logger.warning(
                f"Attempted to save empty or None GeoDataFrame for {output_key}. Skipping."
            )
            return
        path = self._get_output_path(output_key)
        save_vector_data(gdf, path, driver="GPKG")  # Use GeoPackage for intermediates
        self.output_paths[output_key] = path
        logger.info(f"Saved intermediate vector data: {path}")

    def _save_raster(self, array, profile, output_key: str, metric_name: str = None):
        """Saves a raster file, optionally appending a metric name."""
        if array is None:
            logger.warning(f"Attempted to save None array for {output_key}. Skipping.")
            return
        path = self._get_output_path(output_key)
        if metric_name:
            # Append metric name before the suffix
            path = path.parent / f"{path.stem}_{metric_name}{path.suffix}"
        save_raster_data(array, profile, path)
        # Store the actual path used, including the metric name
        output_key_metric = f"{output_key}_{metric_name}" if metric_name else output_key
        self.output_paths[output_key_metric] = path
        logger.info(f"Saved raster data: {path}")

    def _reproject_if_needed(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Reprojects GeoDataFrame to target CRS if not already matching."""
        if gdf is None:
            return None
        target_crs = f"EPSG:{self.settings.processing.output_crs_epsg}"
        if gdf.crs is None:
            logger.warning("Input GDF has no CRS, assuming EPSG:4326 for reprojection.")
            gdf.crs = "EPSG:4326"  # Common default, adjust if needed
        # Use CRS objects for reliable comparison
        if not gdf.crs.equals(target_crs):
            logger.info(f"Reprojecting GDF from {gdf.crs} to {target_crs}")
            return reproject_gdf(gdf, target_crs)
        return gdf
