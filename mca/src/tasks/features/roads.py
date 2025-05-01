import logging
import geopandas as gpd
from pydantic import BaseModel

# Local imports
from src.tasks.features.feature_base import FeatureBase
from src.utils import load_vector_data

logger = logging.getLogger(__name__)


class Roads(FeatureBase):
    """
    Handles N50 road network data (Samferdsel), filters specific road types,
    and calculates differences between road layers and bike lanes.
    """

    def __init__(self, settings: BaseModel , wbt: object = None):
        super().__init__(settings, None)
        self.gdf_samferdsel = None
        self.gdf_bike_lanes = None
        self.gdf_roads_simple = None
        self.gdf_roads_simple_diff_lanes = None
        self.gdf_roads_all_diff_lanes = None

    def _load_samferdsel_layer(self) -> gpd.GeoDataFrame | None:
        """Loads the main N50 Samferdsel layer from the configured GDB."""
        logger.info("Loading N50 Samferdsel layer...")
        if (
            not self.settings.paths.n50_gdb_path
            or not self.settings.paths.n50_gdb_path.exists()
        ):
            logger.warning(
                "N50 GDB path not configured or not found. Skipping Samferdsel (Roads) loading."
            )
            return None

        try:
            gdf = load_vector_data(
                self.settings.paths.n50_gdb_path,
                layer=self.settings.input_data.n50_samferdsel_layer,
            )
            gdf = self._reproject_if_needed(gdf)
            self._save_intermediate_gdf(gdf, "prepared_roads_gpkg") # Save raw samferdsel
            logger.info(f"Loaded Samferdsel layer with {len(gdf)} features.")
            return gdf
        except Exception as e:
            logger.error(f"Error loading N50 Samferdsel layer: {e}", exc_info=True)
            return None

    def _filter_by_typeveg(
        self, gdf: gpd.GeoDataFrame, typeveg_value: str, output_filename_attr: str
    ) -> gpd.GeoDataFrame | None:
        """Filters the GeoDataFrame based on the 'typeveg' field."""
        if gdf is None:
            logger.warning(f"Input GDF is None, cannot filter for {typeveg_value}.")
            return None

        typeveg_field = self.settings.input_data.n50_samferdsel_typeveg_field
        if typeveg_field not in gdf.columns:
            logger.error(
                f"Field '{typeveg_field}' not found in Samferdsel GDF. Cannot filter."
            )
            return None

        logger.info(f"Filtering for {typeveg_field} = '{typeveg_value}'...")
        filtered_gdf = gdf[gdf[typeveg_field] == typeveg_value].copy()

        if filtered_gdf.empty:
            logger.warning(f"No features found for typeveg = '{typeveg_value}'.")
            return None

        logger.info(f"Found {len(filtered_gdf)} features for '{typeveg_value}'.")
        self._save_intermediate_gdf(filtered_gdf, output_filename_attr)
        return filtered_gdf

    def _calculate_difference(
        self,
        gdf_base: gpd.GeoDataFrame | None,
        gdf_subtract: gpd.GeoDataFrame | None,
        output_filename_attr: str,
        buffer_subtract: float = 0.1,
    ) -> gpd.GeoDataFrame | None:
        """Calculates the geometric difference (base - subtract)."""
        if gdf_base is None or gdf_subtract is None:
            logger.warning(
                "Cannot calculate difference due to missing input GeoDataFrames."
            )
            return None
        if gdf_base.empty or gdf_subtract.empty:
             logger.warning(
                f"One or both input GDFs for difference ('{output_filename_attr}') are empty. Result will likely be the base GDF or empty."
            )
             # If subtract is empty, difference is just the base
             if gdf_subtract.empty:
                 diff_gdf = gdf_base.copy()
                 self._save_intermediate_gdf(diff_gdf, output_filename_attr)
                 return diff_gdf
             # If base is empty, difference is empty
             if gdf_base.empty:
                 diff_gdf = gdf_base.copy() # Already empty
                 self._save_intermediate_gdf(diff_gdf, output_filename_attr)
                 return diff_gdf

        logger.info(f"Calculating difference for '{output_filename_attr}'...")
        try:
            # Buffer the bike lanes layer
            logger.info("Buffering bike lanes...")
            buffer_size = self.settings.processing.bike_lane_buffer  # Get buffer size from config
            gdf_subtract_buffered = gdf_subtract.copy()
            gdf_subtract_buffered['geometry'] = gdf_subtract_buffered.geometry.buffer(buffer_size)

            # Ensure CRS is consistent before buffering/overlay
            if gdf_base.crs != gdf_subtract_buffered.crs:
                 logger.warning(f"CRS mismatch for difference operation '{output_filename_attr}'. Reprojecting subtract layer.")
                 gdf_subtract_buffered = gdf_subtract_buffered.to_crs(gdf_base.crs)

            # Perform overlay using GeoPandas
            # Note: overlay might be slow for large datasets
            diff_gdf = gpd.overlay(
                gdf_base,
                gdf_subtract_buffered,
                how="difference",
                keep_geom_type=True, # Keep original geometry type (LineString)
            )

            # Clean up potential invalid geometries resulting from difference
            original_count = len(diff_gdf)
            diff_gdf = diff_gdf[
                diff_gdf.geometry.is_valid & ~diff_gdf.geometry.is_empty
            ]
            cleaned_count = len(diff_gdf)
            if original_count != cleaned_count:
                logger.debug(f"Removed {original_count - cleaned_count} invalid/empty geometries after difference.")

            logger.info(f"Difference calculation complete for '{output_filename_attr}', resulting in {len(diff_gdf)} features.")
            self._save_intermediate_gdf(diff_gdf, output_filename_attr)
            return diff_gdf

        except Exception as e:
            logger.error(
                f"Error calculating difference for '{output_filename_attr}': {e}",
                exc_info=True,
            )
            return None

    def _calculate_length_ratio(
            self,
            gdf_numerator: gpd.GeoDataFrame | None,
            gdf_denominator: gpd.GeoDataFrame | None
        ) -> float | None:
        """Calculates the ratio of total lengths between two GeoDataFrames."""
        if gdf_numerator is None or gdf_denominator is None:
            logger.warning("Cannot calculate length ratio due to missing input GDFs.")
            return None
        if gdf_numerator.empty or gdf_denominator.empty:
            logger.warning("Cannot calculate length ratio due to empty input GDFs.")
            # Return 0 if numerator is empty, handle division by zero if denominator is empty
            if gdf_denominator.empty:
                return None # Or raise error? Or return infinity? None seems safest.
            else:
                return 0.0

        try:
            len_num = gdf_numerator.geometry.length.sum()
            len_den = gdf_denominator.geometry.length.sum()
            logger.info(f"Numerator total length: {len_num:.2f} meters")
            logger.info(f"Denominator total length: {len_den:.2f} meters")

            if len_den == 0:
                 logger.warning("Denominator total length is zero, cannot calculate ratio.")
                 return None

            ratio = len_num / len_den
            logger.info(f"Calculated length ratio: {ratio:.4f}")
            return ratio
        except Exception as e:
            logger.error(f"Error calculating length ratio: {e}", exc_info=True)
            return None

    def load_data(self):
        """Loads, filters, and processes N50 road data."""
        logger.info("--- Starting N50 Roads Processing ---")

        # 1. Load base Samferdsel layer
        self.gdf_samferdsel = self._load_samferdsel_layer()
        if self.gdf_samferdsel is None:
            logger.error("Failed to load base Samferdsel layer. Aborting Roads processing.")
            return # Cannot proceed without base data

        # 2. Filter Bike Lanes
        self.gdf_bike_lanes = self._filter_by_typeveg(
            self.gdf_samferdsel,
            self.settings.input_data.n50_typeveg_bike_lane,
            "prepared_bike_lanes_filtered_gpkg",
        )

        # 3. Filter Simple Roads
        self.gdf_roads_simple = self._filter_by_typeveg(
            self.gdf_samferdsel,
            self.settings.input_data.n50_typeveg_road_simple,
            "prepared_roads_simple_filtered_gpkg",
        )

        # 4. Calculate Difference (Simple Roads - Bike Lanes)
        self.gdf_roads_simple_diff_lanes = self._calculate_difference(
            self.gdf_roads_simple,
            self.gdf_bike_lanes,
            "prepared_roads_simple_diff_lanes_gpkg",
        )

        # 5. Calculate Difference (All Roads - Bike Lanes) - Bonus
        self.gdf_roads_all_diff_lanes = self._calculate_difference(
            self.gdf_samferdsel,
            self.gdf_bike_lanes,
            "prepared_roads_all_diff_lanes_gpkg",
        )

        # 6. Calculate Ratio Difference - Bonus
        logger.info("Calculating ratio of (Simple Roads - Lanes) / (All Roads - Lanes)...")
        self.length_ratio = self._calculate_length_ratio(
            self.gdf_roads_simple_diff_lanes,
            self.gdf_roads_all_diff_lanes
        )
        if self.length_ratio is not None:
            logger.info(f"Ratio of lengths (simple_diff / all_diff): {self.length_ratio:.4f}")
        else:
            logger.warning("Could not calculate length ratio.")

        logger.info("--- N50 Roads Processing Finished ---")


    def build(self):
        """
        Ensures data is loaded and returns paths to the generated vector files.
        Could be extended to rasterize outputs if needed.
        """
        if self.gdf_samferdsel is None: # Check if load_data has run
            self.load_data()

        # Return paths to the generated files
        output_paths = {
            "samferdsel_all": self._get_output_path("prepared_roads_gpkg"),
            "bike_lanes_filtered": self._get_output_path("prepared_bike_lanes_filtered_gpkg"),
            "roads_simple_filtered": self._get_output_path("prepared_roads_simple_filtered_gpkg"),
            "roads_simple_diff_lanes": self._get_output_path("prepared_roads_simple_diff_lanes_gpkg"),
            "roads_all_diff_lanes": self._get_output_path("prepared_roads_all_diff_lanes_gpkg"),
        }
        # Log paths for confirmation
        for name, path in output_paths.items():
            if path and path.exists():
                logger.debug(f"Output file '{name}' available at: {path}")
            elif path:
                 logger.warning(f"Output file '{name}' path generated ({path}), but file does not exist.")
            else:
                 logger.warning(f"Could not determine output path for '{name}'.")

        return output_paths


if __name__ == "__main__":
    from src.config import settings

    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running Roads feature generation directly...")

    # Ensure output directory exists for the test run
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {settings.paths.output_dir}")

    # Instantiate and run
    roads_processor = Roads(settings)
    roads_processor.load_data() # Load and process data
    output_files = roads_processor.build() # Get output file paths

    # Print summary
    logger.info("--- Roads Processing Test Summary ---")
    if roads_processor.gdf_samferdsel is not None:
        logger.info(f"Total Samferdsel features loaded: {len(roads_processor.gdf_samferdsel)}")
    if roads_processor.gdf_bike_lanes is not None:
        logger.info(f"Filtered Bike Lanes features: {len(roads_processor.gdf_bike_lanes)}")
    if roads_processor.gdf_roads_simple is not None:
        logger.info(f"Filtered Simple Roads features: {len(roads_processor.gdf_roads_simple)}")
    if roads_processor.gdf_roads_simple_diff_lanes is not None:
        logger.info(f"Simple Roads (diff Lanes) features: {len(roads_processor.gdf_roads_simple_diff_lanes)}")
        logger.info(f"  Output file: {output_files.get('roads_simple_diff_lanes')}")
    if roads_processor.gdf_roads_all_diff_lanes is not None:
        logger.info(f"All Roads (diff Lanes) features: {len(roads_processor.gdf_roads_all_diff_lanes)}")
        logger.info(f"  Output file: {output_files.get('roads_all_diff_lanes')}")
    if hasattr(roads_processor, 'length_ratio') and roads_processor.length_ratio is not None:
         logger.info(f"Length Ratio (simple_diff / all_diff): {roads_processor.length_ratio:.4f}")
    else:
        logger.warning("Length ratio was not calculated or available.")

    logger.info("Check the output directory for generated GeoPackage files.")
    logger.info("--- Test Run Complete ---")
