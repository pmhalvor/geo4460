import logging
import dask
import geopandas as gpd

logger = logging.getLogger(__name__)  # Define logger

# Local imports (adjust relative paths as needed)
try:
    from .feature_base import FeatureBase
    from src.config import AppConfig
    from src.utils import load_vector_data
except ImportError:
    logger.warning(
        "Could not import from src.* or .feature_base directly, attempting relative import..."
    )
    from feature_base import FeatureBase
    from ...config import AppConfig
    from ...utils import load_vector_data


class Roads(FeatureBase):
    """Handles N50 road and bike lane data."""

    def load_data(self):
        logger.info("Loading N50 road and bike lane data...")
        if (
            not self.settings.paths.n50_gdb_path
            or not self.settings.paths.n50_gdb_path.exists()
        ):
            logger.warning(
                "N50 GDB path not configured or not found. Skipping Roads loading."
            )
            self.gdf_roads = None
            self.gdf_lanes = None
            return

        try:
            # Load roads
            self.gdf_roads = load_vector_data(
                self.settings.paths.n50_gdb_path,
                layer=self.settings.input_data.n50_roads_layer,
            )
            self.gdf_roads = self._reproject_if_needed(self.gdf_roads)
            self._save_intermediate_gdf(self.gdf_roads, "prepared_roads_gpkg")

            # Load bike lanes
            self.gdf_lanes = load_vector_data(
                self.settings.paths.n50_gdb_path,
                layer=self.settings.input_data.n50_bike_lanes_layer,
            )
            self.gdf_lanes = self._reproject_if_needed(self.gdf_lanes)
            self._save_intermediate_gdf(self.gdf_lanes, "prepared_bike_lanes_gpkg")

            # Calculate roads without lanes
            logger.info("Calculating roads without bike lanes...")
            if self.gdf_roads is not None and self.gdf_lanes is not None:
                # Ensure consistent geometry types if needed before difference
                # Buffer lanes slightly for more robust difference?
                buffered_lanes = self.gdf_lanes.buffer(0.1)  # Small buffer
                # Perform overlay using GeoPandas
                # Note: overlay might be slow for large datasets
                gdf_roads_no_lanes = gpd.overlay(
                    self.gdf_roads,
                    gpd.GeoDataFrame(geometry=buffered_lanes, crs=self.gdf_lanes.crs),
                    how="difference",
                    keep_geom_type=True,  # Keep original geometry type (LineString)
                )
                # Clean up potential invalid geometries resulting from difference
                gdf_roads_no_lanes = gdf_roads_no_lanes[
                    gdf_roads_no_lanes.geometry.is_valid
                    & ~gdf_roads_no_lanes.geometry.is_empty
                ]

                self._save_intermediate_gdf(
                    gdf_roads_no_lanes, "prepared_roads_no_lanes_gpkg"
                )
                self.output_paths["roads_no_lanes"] = self._get_output_path(
                    "prepared_roads_no_lanes_gpkg"
                )
            else:
                logger.warning(
                    "Could not calculate roads without lanes due to missing inputs."
                )

            logger.info("N50 Roads/Lanes loaded and preprocessed.")

        except Exception as e:
            logger.error(f"Error loading N50 data: {e}", exc_info=True)
            self.gdf_roads = None
            self.gdf_lanes = None

    def build(self):
        """Build tasks related to roads (e.g., rasterization if needed)."""
        if getattr(self, "gdf_roads", None) is None:  # Check if loaded
            self.load_data()

        # Currently, load_data saves the prepared vector files.
        # This build step could rasterize them if needed for overlays.
        roads_no_lanes_path = self.output_paths.get("roads_no_lanes")
        if roads_no_lanes_path and roads_no_lanes_path.exists():
            # TODO: Implement rasterization using WBT vector_lines_to_raster or similar
            logger.warning("Road rasterization not implemented.")
            pass
        else:
            logger.warning("Prepared roads_no_lanes file not found, cannot rasterize.")

        # Return paths to the *vector* files generated in load_data for now
        return {
            "roads": self.output_paths.get("prepared_roads_gpkg"),
            "lanes": self.output_paths.get("prepared_bike_lanes_gpkg"),
            "roads_no_lanes": roads_no_lanes_path,
        }
