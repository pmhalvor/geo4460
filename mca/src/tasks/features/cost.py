import logging
import dask
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.warp
import rasterio.crs
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

# Local imports
from src.tasks.features.feature_base import FeatureBase
from src.config import AppConfig
from src.utils import (
    load_raster_data,
    load_vector_data,
    get_raster_profile,
    align_rasters,
    reproject_gdf,
    display_multi_layer_on_folium_map,
)


logger = logging.getLogger(__name__)


class CostLayer(FeatureBase):
    """
    Calculates a cost surface based on slope, speed, and road restrictions.

    Input layers (slope, speed impact) are normalized and then weighted to calculate cost.
    The cost function prioritizes lower slopes and adjusts based on speed relative
    to a threshold (6 m/s). Travel is restricted to buffered road areas.
    The final output represents a cost value (lower is better), with non-road areas
    assigned a very high cost.
    """

    # Define constants for clarity
    ROAD_MASK_VALUE = 1
    NO_ROAD_MASK_VALUE = 0
    IMPOSSIBLE_COST_FACTOR = 1e4 # Factor to multiply max valid cost for impossible areas

    def __init__(
        self,
        settings: AppConfig,
        wbt, # WhiteboxTools instance (might not be needed if using rasterio)
        slope_raster_path: Path | str,
        roads_vector_path: Path | str, # Expecting vector path (e.g., GPKG)
        speed_raster_path: Optional[Path | str] = None,
    ):
        super().__init__(settings, wbt) # Pass wbt even if unused for consistency
        self.slope_raster_path = Path(slope_raster_path) if slope_raster_path else None
        self.roads_vector_path = Path(roads_vector_path) if roads_vector_path else None
        self.speed_raster_path = Path(speed_raster_path) if speed_raster_path else None
        self.template_profile = None # To store the profile of the base raster (slope)
        self.load_data() # Validate inputs

    def load_data(self):
        """Validates required input file paths."""
        logger.info("Validating CostLayer input paths...")
        if not self.slope_raster_path or not self.slope_raster_path.exists():
            raise FileNotFoundError(
                f"Slope raster path is required and not found: {self.slope_raster_path}"
            )
        if not self.roads_vector_path or not self.roads_vector_path.exists():
            raise FileNotFoundError(
                f"Roads vector path is required and not found: {self.roads_vector_path}"
            )
        if self.speed_raster_path and not self.speed_raster_path.exists():
            logger.warning(
                f"Provided speed raster path not found: {self.speed_raster_path}. Proceeding without speed cost adjustment."
            )
            self.speed_raster_path = None # Reset if not found

        # Load template profile from slope raster early
        self.template_profile = get_raster_profile(self.slope_raster_path)
        logger.info("Input paths validated and template profile loaded.")

    def _get_template_profile(self) -> dict:
        """Returns the loaded template raster profile."""
        if self.template_profile is None:
             # Should have been loaded in load_data, but load again if needed
             self.template_profile = get_raster_profile(self.slope_raster_path)
        return self.template_profile

    def _normalize_array_0_1(self, array_data: np.ndarray, nodata_val_in_source=None) -> np.ndarray:
        """
        Normalizes a numpy array to the 0-1 range.
        NaNs or specified nodata_val_in_source in the input are preserved as NaNs in the output.
        """
        normalized_array = array_data.copy().astype(np.float32)
        
        nan_mask_orig = np.isnan(normalized_array)
        if nodata_val_in_source is not None:
            # Combine original NaNs with new NaNs from nodata_val_in_source
            nan_mask = nan_mask_orig | (normalized_array == nodata_val_in_source)
            normalized_array[nan_mask] = np.nan 
        else:
            nan_mask = nan_mask_orig

        valid_values = normalized_array[~nan_mask]

        if valid_values.size > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)

            if max_val > min_val:
                normalized_array[~nan_mask] = (valid_values - min_val) / (max_val - min_val)
            else: # All valid values are the same (or only one valid value)
                normalized_array[~nan_mask] = 0.0 # Assign 0.0 if no variation or single value
        
        # Ensure original NaNs (if nodata_val_in_source was None) or new NaNs are set
        normalized_array[nan_mask] = np.nan
        return normalized_array
        
    def _postprocess_raster_crs(self, input_path: str, src_epsg=25833, dst_epsg=4326):
        """
        Post-processes a raster by:
        1. Assigning the correct CRS if missing
        2. Reprojecting to the visualization CRS (EPSG:4326) for mapping
        
        Args:
            input_path: Path to the input raster
            src_epsg: Source EPSG code (processing CRS), defaults to 25833
            dst_epsg: Destination EPSG code (visualization CRS), defaults to 4326
            
        Returns:
            Path to the reprojected raster
        """
        logger.info(f"Post-processing raster: Reprojecting from EPSG:{src_epsg} to EPSG:{dst_epsg}")
        input_path = Path(input_path)
        output_path = input_path.with_name(f"{input_path.stem}_4326{input_path.suffix}")

        # Step 1: Assign missing CRS if needed
        with rasterio.open(input_path, mode="r+") as src:
            if src.crs is None or src.crs.to_epsg() != src_epsg:
                logger.info(f"Assigning CRS EPSG:{src_epsg} to raster")
                src.crs = rasterio.crs.CRS.from_epsg(src_epsg)
        
        # Step 2: Reproject to visualization CRS (EPSG:4326)
        with rasterio.open(input_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_epsg, src.width, src.height, *src.bounds
            )

            with rasterio.open(output_path, 'w', driver="GTiff",
                height=height, width=width,
                count=1, dtype=src.dtypes[0],
                crs=dst_epsg, transform=transform) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_epsg,
                        resampling=Resampling.bilinear
                    )

        logger.info(f"Reprojected raster saved to: {output_path}")
        return output_path
        
    def _rasterize_roads(self) -> Tuple[Optional[np.ndarray], Optional[Path]]:
        """
        Loads roads vector, buffers it, and rasterizes it based on the template profile.

        Returns:
            Tuple[Optional[np.ndarray], Optional[Path]]: Rasterized road mask array and its path, or (None, None) on failure.
        """
        logger.info("Rasterizing roads vector...")
        profile = self._get_template_profile()
        target_crs = profile.get("crs")
        if not target_crs:
            logger.error("Template profile is missing CRS. Cannot proceed.")
            return None, None

        output_key = "rasterized_roads_mask"
        output_path = self._get_output_path(output_key)

        try:
            # Load roads vector
            roads_gdf = load_vector_data(self.roads_vector_path)
            if roads_gdf.empty:
                logger.error("Roads vector file loaded but is empty.")
                return None, None

            # Reproject if necessary
            if roads_gdf.crs != target_crs:
                logger.info(f"Reprojecting roads from {roads_gdf.crs} to {target_crs}")
                roads_gdf = reproject_gdf(roads_gdf, target_crs)

            # Buffer roads
            buffer_dist = self.settings.processing.cost_road_buffer_meters
            logger.info(f"Buffering roads by {buffer_dist} meters...")
            # Ensure geometry column exists and is valid before buffering
            if 'geometry' not in roads_gdf.columns:
                 logger.error("Geometry column not found in roads GeoDataFrame.")
                 return None, None
            roads_gdf = roads_gdf[roads_gdf.geometry.is_valid & ~roads_gdf.geometry.is_empty]
            if roads_gdf.empty:
                logger.error("No valid geometries remain after filtering before buffering.")
                return None, None

            # TODO refactor manual reprojection
            logger.info("Before "*50)
            logger.info(roads_gdf.crs)
            roads_gdf = roads_gdf.to_crs(epsg=25833)
            logger.info("After "*50)
            logger.info(roads_gdf.crs)

            buffered_roads = roads_gdf.buffer(buffer_dist)
            logger.info("Buffering complete.")
            logger.warning(roads_gdf.crs)
            logger.warning(buffered_roads.crs)

            # Prepare shapes for rasterization
            # shapes = [(geom, self.ROAD_MASK_VALUE) for geom in roads_gdf]
            shapes = [(geom, self.ROAD_MASK_VALUE) for geom in buffered_roads]

            # Rasterize
            logger.info("Rasterizing buffered roads...")
            mask_array = rasterio.features.rasterize(  # TODO if fails, try using WBT
                shapes=shapes,
                out_shape=(profile["height"], profile["width"]),
                transform=profile["transform"],
                fill=self.NO_ROAD_MASK_VALUE, # Fill value for non-road areas
                dtype=profile["dtype"], # Match template dtype initially
                all_touched=True, # Consider pixels touched by the polygon edge
            )

            # Ensure mask is boolean or integer type suitable for masking
            mask_array = mask_array.astype(np.uint8) # Use uint8 for mask (0 or 1)

            # Update profile for the mask raster (single band, uint8)
            mask_profile = profile.copy()
            mask_profile.update({
                'dtype': mask_array.dtype,
                'count': 1,
                'nodata': None # Explicitly set no nodata for the mask itself
            })

            # Save the intermediate mask raster
            self._save_raster(mask_array, mask_profile, output_key) # Uses base class method
            logger.info(f"Rasterized roads mask saved to {output_path}")

            return mask_array, output_path

        except Exception as e:
            logger.error(f"Error rasterizing roads: {e}", exc_info=True)
            return None, None

    def _calculate_slope_cost(self) -> Optional[np.ndarray]:
        """Loads slope raster and calculates initial cost based on slope weight."""
        logger.info("Calculating slope cost component...")
        try:
            slope_data, slope_profile = load_raster_data(self.slope_raster_path)
            # Ensure slope data is float for calculations
            slope_data = slope_data.astype(np.float32) # Ensure float for calculations

            # Handle potential NoData values in slope
            nodata_val = slope_profile.get("nodata")
            # Normalization handles nodata/NaNs internally
            
            # Normalize raw slope data to a [0,1] factor
            # Higher slope values will result in higher normalized factors (closer to 1)
            logger.info("Normalizing raw slope data to [0,1] factor...")
            normalized_slope_factor = self._normalize_array_0_1(slope_data, nodata_val_in_source=nodata_val)
            
            # Apply weight: cost increases with normalized slope factor
            slope_weight = self.settings.processing.cost_slope_weight
            slope_cost = normalized_slope_factor * slope_weight
            
            # NaNs from normalization will propagate.
            # Values will be in range [0, slope_weight] or NaN.
            logger.info(f"Slope cost calculation complete. Range: [0, {slope_weight}] (plus NaNs).")
            return slope_cost

        except Exception as e:
            logger.error(f"Error calculating slope cost: {e}", exc_info=True)
            return None

    def _calculate_speed_cost(self) -> Optional[np.ndarray]:
        """
        Loads and aligns speed raster (if available) and calculates speed cost modifier.
        Cost increases for speed < 6 m/s, decreases for speed > 6 m/s.
        """
        if not self.speed_raster_path:
            logger.info("No speed raster provided. Skipping speed cost calculation.")
            return None # Return None to indicate no speed cost adjustment

        logger.info("Calculating speed cost component...")
        aligned_speed_path = self._get_output_path("aligned_speed_raster")

        try:
            # Align speed raster to slope raster template
            align_rasters(
                source_raster_path=self.speed_raster_path,
                template_raster_path=self.slope_raster_path,
                output_raster_path=aligned_speed_path,
                resampling_method="bilinear", # Use bilinear for continuous speed data
            )

            # Load aligned speed data
            speed_data, speed_profile = load_raster_data(aligned_speed_path)
            speed_data = speed_data.astype(np.float32) # TODO float64?

            # Handle potential NoData values in speed
            nodata_val = speed_profile.get("nodata")
            speed_mask = None
            if nodata_val is not None:
                speed_mask = (speed_data == nodata_val)
                speed_data[speed_mask] = np.nan # Use NaN internally

            # Calculate speed cost modifier
            speed_threshold = self.settings.processing.cost_speed_threshold_ms # 6.0 m/s
            speed_weight = self.settings.processing.cost_speed_weight

            # Cost modifier: positive for slow speeds, negative for fast speeds
            # Formula: weight * (threshold - speed)
            # Example: speed=4, threshold=6 -> raw_effect = 2 (higher cost implication)
            # Example: speed=8, threshold=6 -> raw_effect = -2 (lower cost implication/benefit)
            raw_speed_effect = speed_threshold - speed_data

            # Normalize this raw_speed_effect to a [0,1] factor
            # Lowest raw_speed_effect (max benefit) maps to 0
            # Highest raw_speed_effect (max penalty) maps to 1
            logger.info("Normalizing raw speed effect to [0,1] factor...")
            # Pass nodata_val from speed_profile if speed_mask was created from it
            # If speed_mask is None, it means nodata_val was None or not used for masking raw_speed_effect
            speed_nodata_for_norm = nodata_val if speed_mask is not None else None
            normalized_speed_effect_factor = self._normalize_array_0_1(raw_speed_effect, nodata_val_in_source=speed_nodata_for_norm)
            # If raw_speed_effect had NaNs (from speed_mask), normalized_speed_effect_factor will also have them.

            # Apply weight: cost modifier increases with normalized speed effect factor
            speed_cost_modifier = speed_weight * normalized_speed_effect_factor
            # Values will be in range [0, speed_weight] or NaN.
            
            logger.info(f"Speed cost modifier calculation complete. Range: [0, {speed_weight}] (plus NaNs).")
            return speed_cost_modifier
            
        except FileNotFoundError:
             logger.error(f"Aligned speed raster not found after alignment attempt: {aligned_speed_path}. Skipping speed cost.")
             return None
        except Exception as e:
            logger.error(f"Error calculating speed cost: {e}", exc_info=True)
            return None

    def _combine_costs(
        self,
        slope_cost: np.ndarray,
        road_mask: np.ndarray,
        speed_cost_modifier: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Combines slope cost, optional speed cost modifier, and applies road mask.

        Sets cost in non-road areas to a very high value.
        """
        logger.info("Combining cost components and applying road mask...")
        try:
            # Start with slope cost as base
            combined_cost = slope_cost.copy() # Work on a copy

            # Add speed cost modifier if available
            if speed_cost_modifier is not None:
                # Ensure alignment (should be guaranteed by previous steps, but double-check shape)
                if combined_cost.shape != speed_cost_modifier.shape:
                    logger.error("Shape mismatch between slope cost and speed cost modifier. Cannot combine.")
                    return None
                
                # Add speed_cost_modifier to combined_cost.
                # If slope_cost is NaN, result is NaN.
                # If speed_cost_modifier is NaN, it's treated as 0 for the sum, so combined_cost effectively remains slope_cost.
                speed_modifier_for_sum = np.nan_to_num(speed_cost_modifier, nan=0.0)
                combined_cost = combined_cost + speed_modifier_for_sum
                
                # Apply normalization
                combined_cost = self._normalize_array_0_1(combined_cost)

            # Apply road mask
            # Ensure mask is aligned (should be guaranteed by _rasterize_roads)
            if combined_cost.shape != road_mask.shape:
                 logger.error("Shape mismatch between combined cost and road mask. Cannot apply mask.")
                 return None

            # Identify non-road areas (where mask is not ROAD_MASK_VALUE)
            non_road_areas = (road_mask != self.ROAD_MASK_VALUE)

            # Handle NoData in combined_cost before setting high cost
            # We want impossible cost only where roads are absent, not where input data was missing
            cost_nodata_mask = np.isnan(combined_cost)

            # Calculate a very high cost value based on the max *valid* cost
            valid_costs = combined_cost[~cost_nodata_mask & ~non_road_areas]
            if valid_costs.size > 0:
                max_valid_cost = np.max(valid_costs)
                # Add a buffer to max valid cost before multiplying, in case max is 0
                impossible_cost_value = (max_valid_cost + 1) * self.IMPOSSIBLE_COST_FACTOR
            else:
                # Fallback if there are no valid costs (e.g., all masked or NaN)
                impossible_cost_value = self.IMPOSSIBLE_COST_FACTOR # Use the factor directly

            logger.info(f"Setting non-road areas cost to ~{impossible_cost_value:.2e}")
            combined_cost[non_road_areas] = impossible_cost_value

            # Restore NaNs where original cost data was NaN (unless it's a non-road area)
            combined_cost[cost_nodata_mask & ~non_road_areas] = np.nan

            logger.info("Cost combination and masking complete.")
            return combined_cost

        except Exception as e:
            logger.error(f"Error combining costs: {e}", exc_info=True)
            return None

    def create_multi_layer_visualization(
            self, 
            cost_raster_path: Path, 
            slope_raster_path: Path, 
            speed_raster_path: Optional[Path] = None
        ) -> Optional[str]:
        """
        Creates a multi-layer visualization showing cost, slope, and speed (if available)
        in a single interactive map.
        
        Args:
            cost_raster_path (Path): Path to the normalized cost raster.
            slope_raster_path (Path): Path to the slope raster.
            speed_raster_path (Optional[Path]): Path to the speed raster, if available.
            
        Returns:
            Optional[str]: Path to the generated HTML map, or None if visualization failed.
        """
        
        if not cost_raster_path.is_file():
            logger.error(f"Cost raster file not found for visualization: {cost_raster_path}")
            return None
            
        logger.info("Creating multi-layer visualization of cost components...")
        
        # Define output HTML path
        output_html_path = self._get_output_path("cost_layer_visualization_html")
        
        try:
            # Prepare layer configurations
            layers = []
            
            # 1. Add the cost raster layer
            cost_raster_4326 = self._postprocess_raster_crs(str(cost_raster_path))
            layers.append({
                'path': cost_raster_4326,
                'name': 'Calculated Cost (Lower is Better, High=Impossible)',
                'type': 'raster',
                'raster': {
                    'cmap': 'viridis_r',  # Reversed colormap (dark=high cost)
                    'opacity': 0.7,
                    'nodata_transparent': True,
                    'show': True
                }
            })
            
            # 2. Add slope raster layer
            if slope_raster_path and slope_raster_path.is_file():
                if "4326" not in str(slope_raster_path):
                    # If slope raster is not already in 4326, reproject it
                    slope_raster_path = self._postprocess_raster_crs(str(slope_raster_path))
                layers.append({
                    'path': slope_raster_path, # slope should already be in 4326
                    'name': f'Slope ({self.settings.processing.slope_units})',
                    'type': 'raster',
                    'raster': {
                        'cmap': 'terrain',
                        'opacity': 0.7,
                        'nodata_transparent': True,
                        'show': False  # Hidden by default
                    }
                })
                logger.info(f"Slope raster layer added to visualization: {slope_raster_path}")
            else:
                logger.warning(f"Slope raster not found at {str(slope_raster_path)}. Skipping slope layer.")
            
            # 3. Add speed raster layer (if available)
            if speed_raster_path and speed_raster_path.is_file():
                speed_raster_4326 = self._postprocess_raster_crs(str(speed_raster_path))
                layers.append({
                    'path': speed_raster_4326,
                    'name': 'Average Speed (m/s)',
                    'type': 'raster',
                    'raster': {
                        'cmap': 'plasma',
                        'opacity': 0.7,
                        'nodata_transparent': True,
                        'show': False  # Hidden by default
                    }
                })
                logger.info(f"Speed raster layer added to visualization: {speed_raster_4326}")
            else:
                logger.warning(f"Speed raster not found at {str(speed_raster_path)}. Skipping speed layer.")
            
            # # 4. Add road mask layer (if available) TODO remove
            # road_mask_path = self.output_paths.get("rasterized_roads_mask")
            # if road_mask_path and road_mask_path.is_file():
            #     road_mask_4326 = self._postprocess_raster_crs(str(road_mask_path))
            #     layers.append({
            #         'path': road_mask_4326,
            #         'name': 'Road Network (Buffered)',
            #         'type': 'raster',
            #         'raster': {
            #             'cmap': 'binary',
            #             'opacity': 0.5,
            #             'nodata_transparent': True,
            #             'show': False  # Hidden by default
            #         }
            #     })
            #     logger.info(f"Road mask layer added to visualization: {road_mask_4326}")
            # else:
            #     logger.warning(f"Road mask raster not found at {str(road_mask_path)}. Skipping road layer.")
            
            # Create the multi-layer map
            display_multi_layer_on_folium_map(
                layers=layers,
                output_html_path_str=str(output_html_path),
                map_zoom=9,
                map_tiles='CartoDB positron'
            )
            
            logger.info(f"Multi-layer visualization created and saved to: {output_html_path}")
            return str(output_html_path)
            
        except Exception as e:
            logger.error(f"Error creating multi-layer visualization: {e}", exc_info=True)
            return None

    @dask.delayed
    def _build_cost_raster(self):
        """
        Orchestrates the cost raster generation process:
        1. Rasterize roads.
        2. Calculate slope cost.
        3. Calculate speed cost modifier (optional).
        4. Combine costs and apply road mask.
        5. Save the result (final normalization step removed).
        """
        logger.info("--- Starting Cost Raster Generation ---")
        final_output_key = "calculated_cost_layer"
        final_output_path = self._get_output_path(final_output_key)

        # 1. Rasterize Roads
        road_mask_array, road_mask_path = self._rasterize_roads()
        if road_mask_array is None:
            logger.error("Failed to generate road mask. Aborting cost calculation.")
            return None

        # 2. Calculate Slope Cost
        slope_cost_array = self._calculate_slope_cost()
        logger.info(f"slope cost: {slope_cost_array}")
        if slope_cost_array is None:
            logger.error("Failed to calculate slope cost. Aborting.")
            return None

        # 3. Calculate Speed Cost
        speed_cost_modifier_array = self._calculate_speed_cost()
        # Proceed even if speed cost is None

        # 4. Combine Costs
        combined_cost_array = self._combine_costs(
            slope_cost_array, road_mask_array, speed_cost_modifier_array
        )

        # 5. normalize cost 
        combined_cost_array = self._normalize_array_0_1(combined_cost_array)

        if combined_cost_array is None:
            logger.error("Failed to combine cost components. Aborting.")
            return None

        # 5. Save Final Raster (Normalization of the final combined_cost_array is removed)
        # The combined_cost_array is now the final array to be saved.
        template_profile = self._get_template_profile()
        final_profile = template_profile.copy()
        final_profile.update({
            'dtype': np.float32, # Ensure float32 for the output
            'nodata': np.nan     # Use NaN as nodata
        })
        
        # Ensure combined_cost_array is float32 before saving
        final_cost_array = combined_cost_array.astype(np.float32)

        try:
            self._save_raster(final_cost_array, final_profile, final_output_key)
            logger.info(f"--- Successfully generated Calculated Cost Raster: {final_output_path} ---")
            
            # Return paths to all relevant files so visualization can be done externally
            result = {
                "cost_raster_path": str(final_output_path),
                "slope_raster_path": str(self.slope_raster_path),
                "road_mask_path": self.output_paths.get("rasterized_roads_mask")
            }
            
            # Add speed path if available
            if self.speed_raster_path:
                result["speed_raster_path"] = str(self.speed_raster_path)
                
            aligned_speed_path = self.output_paths.get("aligned_speed_raster")
            if aligned_speed_path and Path(aligned_speed_path).is_file():
                result["aligned_speed_path"] = str(aligned_speed_path)
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to save final normalized cost raster: {e}", exc_info=True)
            return None


    def build(self):
        """Creates a delayed task to build the normalized cost raster."""
        # load_data called in __init__
        logger.info("Creating delayed task for cost raster generation...")
        task = self._build_cost_raster() # Returns a dask.delayed object or None

        if task is None:
            logger.error("Failed to create delayed task for cost raster generation.")
            return None # Indicate task creation failure

        logger.info("Returning delayed task for cost raster computation.")
        # Return the delayed object directly
        return task


if __name__ == "__main__":
    # Imports needed for standalone testing/execution
    from dask.distributed import Client, LocalCluster
    from src.config import settings 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("--- Running cost_layer.py Standalone Test ---")

    if settings:
        # --- Basic Setup ---
        settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using Output Directory: {settings.paths.output_dir}")

        # Setup Dask client 
        wbt = None
        cluster = LocalCluster(n_workers=1, threads_per_worker=1)
        client = Client(cluster)
        logger.info(f"Dask client started: {client.dashboard_link}")

        # --- Define Input Paths (ASSUMPTIONS based on FeatureBase naming) ---
        # These paths MUST exist from previous workflow steps for the test to run
        # Replace with actual paths if testing outside the full workflow # TODO uncomment
        # assumed_slope_raster = settings.paths.output_dir / "slope_raster.tif" # Example name
        # assumed_roads_vector = settings.paths.output_dir / "prepared_roads_gpkg.gpkg" # Example name from Roads feature
        # assumed_speed_raster = settings.paths.output_dir / "average_speed_raster.tif" # Example name from Heatmap feature

        assumed_slope_raster = Path("output/mca_20250511_1414/slope.tif") 
        assumed_roads_vector = Path("output/mca_20250511_1414/prepared_roads_simple_filtered.gpkg")
        assumed_speed_raster = Path("output/mca_20250511_1414/average_speed.tif")

        # Check if assumed files exist
        inputs_ok = True
        if not assumed_slope_raster.exists():
            logger.error(f"Required input for test not found: {assumed_slope_raster}")
            inputs_ok = False
        if not assumed_roads_vector.exists():
            logger.error(f"Required input for test not found: {assumed_roads_vector}")
            inputs_ok = False
        if not assumed_speed_raster.exists():
            logger.warning(f"Optional speed raster not found: {assumed_speed_raster}. Test will run without speed cost.")
            assumed_speed_raster = None # Set to None if not found

        # --- Test CostLayer Feature ---
        if inputs_ok:
            try:
                logger.info("--- Testing CostLayer Feature ---")
                cost_feature = CostLayer(
                    settings=settings,
                    wbt=wbt,
                    slope_raster_path=assumed_slope_raster,
                    roads_vector_path=assumed_roads_vector,
                    speed_raster_path=assumed_speed_raster,
                )

                logger.info("1. Testing Cost Build...")
                # Build now returns a single delayed task or None
                delayed_task = cost_feature.build()

                if delayed_task is not None:
                    logger.info("Received delayed task from build(). Computing...")
                    # Compute the single task. dask.compute returns a tuple.
                    # The task itself returns a dictionary with paths instead of just the cost path
                    computed_results = dask.compute(delayed_task)[0]
                    logger.info("Build computation completed.")

                    # Check if we got valid results (should be a dictionary of paths)
                    if computed_results and isinstance(computed_results, dict):
                        # Extract the cost raster path
                        cost_raster_path = computed_results.get("cost_raster_path")
                        
                        if cost_raster_path and Path(cost_raster_path).exists():
                            logger.info(f"Generated Cost Raster: {cost_raster_path}")
                            
                            # Generate the multi-layer visualization externally
                            try:
                                logger.info("Creating multi-layer visualization...")
                                visualization_result = cost_feature.create_multi_layer_visualization(
                                    cost_raster_path=Path(cost_raster_path),
                                    slope_raster_path=Path(computed_results.get("slope_raster_path", "")),
                                    speed_raster_path=Path(computed_results.get("aligned_speed_path", computed_results.get("speed_raster_path", "")))
                                )
                                if visualization_result:
                                    logger.info(f"Multi-layer visualization created: {visualization_result}")
                            except Exception as vis_e:
                                logger.error(f"Error creating multi-layer visualization: {vis_e}", exc_info=True)
                        else:
                            logger.warning(f"Cost raster not found at expected path: {cost_raster_path}")
                    else:
                        logger.warning("Cost raster generation returned invalid or empty results.")
                else:
                    logger.warning("No task received from build process. Skipping visualization.")

                logger.info("--- CostDistance Feature Test Completed ---")

            except Exception as e:
                logger.error(f"Error during CostDistance test: {e}", exc_info=True)
            finally:
                # Clean up Dask client
                if client:
                    try:
                        client.close()
                        cluster.close()
                        logger.info("Dask client and cluster closed.")
                    except Exception as e:
                        logger.warning(f"Error closing Dask client/cluster: {e}")
        else:
            logger.error("Cannot run Cost test due to missing input files.")
            # Close dask client even if test didn't run fully
            if client: 
                client.close()
            if cluster: 
                cluster.close()

    else:
        logger.error("Settings could not be loaded. Cannot run standalone test.")

    logger.info("--- Standalone Test Finished ---")
