import logging
import geopandas as gpd
import pandas as pd
from pydantic import BaseModel
from pathlib import Path # Added Path

# Local imports
from src.tasks.features.feature_base import FeatureBase
from src.utils import load_vector_data, display_multi_layer_on_folium_map # Added display_multi_layer_on_folium_map

logger = logging.getLogger(__name__)


class BikeLanes(FeatureBase):
    """
    Loads and filters Oslo Sykkelfelt KML data for specific bike lane classifications.
    The category of interest is "Sykkelfelt (Bicycle lane)".
    """
    def __init__(self, settings: BaseModel, wbt: object = None):
        super().__init__(settings, wbt)
        self.gdf_kml_bike_lanes = None 

    def load_data(self) -> None:
        """Loads bike lane data from KML, filters, and prepares it."""
        logger.info(f"--- Starting KML Bike Lanes Processing ---")
        if not self.settings.paths.oslo_sykkelfelt_kml_path or \
           not self.settings.paths.oslo_sykkelfelt_kml_path.exists():
            logger.warning("Oslo Sykkelfelt KML path not configured or not found. Skipping KML bike lanes loading.")
            self.gdf_kml_bike_lanes = gpd.GeoDataFrame() # Ensure it's an empty GDF
            return

        try:
            gdf = None 
            for layer in self.settings.input_data.oslo_bike_path_layers:
                # Load all interesting layers from KML file into single gdf
                gdf_temp = load_vector_data(
                    self.settings.paths.oslo_sykkelfelt_kml_path,
                    layer= layer
                )
                if not gdf_temp.empty:
                    if gdf is None:
                        gdf = gdf_temp.copy()
                    else:
                        gdf = pd.concat([gdf, gdf_temp], ignore_index=True)
            gdf.reset_index(inplace=True, drop=True)

            if gdf.empty:
                logger.warning("KML file loaded an empty GeoDataFrame.")
                self.gdf_kml_bike_lanes = gdf
                return

            logger.info(f"Loaded {len(gdf)} features from KML before filtering.")
            
            self.gdf_kml_bike_lanes = self._reproject_if_needed(gdf)
            self._save_intermediate_gdf(self.gdf_kml_bike_lanes, "prepared_kml_bike_lanes_gpkg")
            logger.info(f"Processed and saved KML bike lanes with {len(self.gdf_kml_bike_lanes)} features.")

        except Exception as e:
            logger.error(f"Error loading or filtering KML bike lanes: {e}", exc_info=True)
            self.gdf_kml_bike_lanes = gpd.GeoDataFrame() # Ensure it's an empty GDF on error
        logger.info("--- KML Bike Lanes Processing Finished ---")

    def build(self) -> dict:
        """
        Ensures KML bike lane data is loaded and returns path to the generated vector file.
        """
        if self.gdf_kml_bike_lanes is None:
            self.load_data()
        
        output_path = self._get_output_path("prepared_kml_bike_lanes_gpkg")
        if output_path and output_path.exists():
            logger.debug(f"Output file for KML bike lanes available at: {output_path}")
        elif output_path:
            logger.warning(f"Output file path for KML bike lanes generated ({output_path}), but file does not exist.")
        else:
            logger.warning("Could not determine output path for KML bike lanes.")
            
        return {"kml_bike_lanes": output_path}


class Roads(FeatureBase):
    """
    Handles N50 road network data (Samferdsel), filters specific road types,
    and calculates differences between road layers and bike lanes.
    """

    def __init__(
            self, 
            settings: BaseModel , 
            wbt: object = None,
            gdf_bike_lanes: gpd.GeoDataFrame = None
        ):
        super().__init__(settings, None)
        self.gdf_samferdsel = None
        self.gdf_bike_lanes = gdf_bike_lanes
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

            # Perform diff overlay using GeoPandas
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

        # # 2. Filter Bike Lanes (only if none provided from KML data)
        if self.gdf_bike_lanes is None:
            logger.info("No KML bike lanes found. Using N50 bike lanes (inconclusive data).")
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

        # 6. Calculate Ratio Difference - Bonus  # TODO remove
        logger.info("Calculating ratio of (Simple Roads - Lanes) / (Simple Roads)...")
        self.length_ratio = self._calculate_length_ratio(
            self.gdf_roads_simple_diff_lanes,
            self.gdf_roads_simple
        )
        if self.length_ratio is not None:
            logger.info(f"Ratio of lengths (simple_diff / simple): {self.length_ratio:.4f}")
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

    def create_multi_layer_visualization(self, output_files_dict: dict):
        """
        Creates a single multi-layer visualization of all generated road layers.
        """
        logger.info("Creating multi-layer visualization for all road layers...")

        if not output_files_dict:
            logger.warning("No output_files_dict provided for visualization.")
            return None

        output_html_path = self.settings.paths.output_dir / "roads_multi_layer_map.html"
        
        # Define distinct colors for each layer for better visual differentiation
        # Using a dictionary to map layer keys to colors and display names
        layer_styles = {
            "kml_bike_lanes": {"name": "All Bike Lanes", "color": "#ff7780", "show": True}, # Red (color of bike lanes in city)
            "roads_simple_filtered": {"name": "Simple Roads", "color": "#cd8f93", "show": False},  # Salmon-grey
            "samferdsel_all": {"name": "All Roads (Samferdsel)", "color": "#b4a8a9", "show": False}, # Grey
            "roads_simple_diff_lanes": {"name": "Simple Roads (No Bike Lanes)", "color": "#77cc9c", "show": True}, # Mint green
            "roads_all_diff_lanes": {"name": "All Roads (No Bike Lanes)", "color": "#328656", "show": False}  # Dark green
        }

        layers_for_map = []
        default_line_weight = 2
        default_tooltip_cols = ["OBJTYPE", "VEGTYPE", "MOTORVEG"] # Common N50 attributes, adjust as needed

        for key, file_path_obj in output_files_dict.items():
            if not file_path_obj or not isinstance(file_path_obj, Path) or not file_path_obj.exists():
                logger.warning(f"Skipping layer '{key}' as its path is invalid or file does not exist: {file_path_obj}")
                continue

            style_info = layer_styles.get(key)
            if not style_info:
                logger.warning(f"No style defined for layer '{key}'. Skipping.")
                continue
            else:
                logger.info(f"Style defined for layer '{key}': {style_info}")

            
            # Attempt to load GDF to check for columns for tooltip/popup
            # This is optional but good for richer maps.
            try:
                gdf_check = gpd.read_file(file_path_obj)
                current_tooltip_cols = [col for col in default_tooltip_cols if col in gdf_check.columns]
                if not current_tooltip_cols and not gdf_check.empty: # If default cols not found, use first few
                    current_tooltip_cols = gdf_check.columns.tolist()[:3]
            except Exception as e:
                logger.debug(f"Could not read GDF for {key} to determine tooltip columns: {e}")
                current_tooltip_cols = []


            layers_for_map.append({
                'path': file_path_obj,
                'name': style_info["name"],
                'type': 'vector',
                'vector': {
                    'color': style_info["color"],
                    '+weight': default_line_weight,
                    'tooltip_cols': current_tooltip_cols,
                    'popup_cols': current_tooltip_cols, # Same as tooltip for simplicity
                    'show': style_info["show"]
                }
            })

        if not layers_for_map:
            logger.warning("No valid layers could be prepared for the map.")
            return None

        try:
            display_multi_layer_on_folium_map(
                layers=layers_for_map,
                output_html_path_str=str(output_html_path),
            )
            return str(output_html_path)
        except Exception as e:
            logger.error(f"Error creating multi-layer roads visualization: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    from src.config import settings

    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running Roads feature generation directly...")

    # Ensure output directory exists for the test run
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {settings.paths.output_dir}")

    # 1. Instantiate and run BikeLanes (KML)
    logger.info("--- Processing KML Bike Lanes ---")
    bike_lanes_processor = BikeLanes(settings)
    bike_lanes_processor.load_data()
    kml_output_files = bike_lanes_processor.build()
    
    if bike_lanes_processor.gdf_kml_bike_lanes is not None and not bike_lanes_processor.gdf_kml_bike_lanes.empty:
        logger.info(f"KML Bike Lanes loaded: {len(bike_lanes_processor.gdf_kml_bike_lanes)} features.")
        logger.info(f"  Output file: {kml_output_files.get('kml_bike_lanes')}")
    else:
        logger.warning(f"No KML bike lanes found for '{settings.input_data.oslo_bike_path_layers}'.")

    # 2. Instantiate and run Roads, passing the KML bike lanes GDF
    logger.info("--- Processing N50 Roads with KML Bike Lanes ---")
    roads_processor = Roads(
        settings=settings, 
        gdf_bike_lanes= bike_lanes_processor.gdf_kml_bike_lanes,
    )
    roads_processor.load_data() 
    roads_output_files = roads_processor.build() 

    # Print summary for Roads
    logger.info("--- Roads Processing with KML Test Summary ---")
    if roads_processor.gdf_samferdsel is not None:
        logger.info(f"Total N50 Samferdsel features loaded: {len(roads_processor.gdf_samferdsel)}")
    
    # Use the updated attribute names for diff layers and corresponding output file keys
    if roads_processor.gdf_roads_simple_diff_lanes is not None:
        logger.info(f"Simple roads - bike lanes overlay gave {len(roads_processor.gdf_roads_simple_diff_lanes)} features")
        logger.info(f"  Output file: {roads_output_files.get('prepared_roads_simple_diff_lanes_gpkg')}")
    if roads_processor.gdf_roads_all_diff_lanes is not None:
        logger.info(f"Simple roads - bike lanes overlay gave {len(roads_processor.gdf_roads_all_diff_lanes)} features")
        logger.info(f"  Output file: {roads_output_files.get('prepared_roads_all_diff_lanes_gpkg')}")
    
    logger.info("Check the output directory for generated GeoPackage files.")

    # Combine output files for visualization
    combined_output_files = {}
    if kml_output_files:
        combined_output_files.update(kml_output_files)
    if roads_output_files:
        combined_output_files.update(roads_output_files)
    
    # Create multi-layer visualization
    if combined_output_files:
        logger.info("Creating multi-layer visualization for all layers...")
        visualization_path = roads_processor.create_multi_layer_visualization(combined_output_files)
        if visualization_path:
            logger.info(f"Created multi-layer visualization: {visualization_path}")
        else:
            logger.warning("Failed to create multi-layer visualization.")
    else:
        logger.warning("No output files available to create visualization.")

    logger.info("--- Test Run Complete ---")
