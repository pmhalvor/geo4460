# mca/src/tasks/features/elevation.py

import dask
import logging
import numpy as np
import pandas as pd
import rasterio
import rasterio.crs
import tempfile
import csv
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from src.tasks.features.feature_base import FeatureBase
from src.utils import (
    load_vector_data,
    polyline_to_points,
    save_vector_data,
    # Assuming a similar display function exists or can be adapted
    # from src.utils import display_dem_slope_on_folium_map
)

# Configure logging (similar to heatmap.py, but maybe keep file handler for elevation specifics)
logger = logging.getLogger(__name__)


class Elevation(FeatureBase):
    """
    Handles N50 elevation contour data to generate DEM and slope rasters
    using WhiteboxTools based on configured interpolation method.
    """

    def load_data(self):
        """Loads and preprocesses N50 contour data from the configured GDB."""
        logger.info("Loading N50 contour data...")
        gdb_path = self.settings.paths.n50_gdb_path
        layer_name = self.settings.input_data.n50_contour_layer
        elevation_field = self.settings.input_data.n50_contour_elevation_field

        if not gdb_path or not gdb_path.exists():
            logger.warning(
                f"N50 GDB path not configured or not found: {gdb_path}. Skipping Elevation loading."
            )
            self.gdf = None
            return

        try:
            self.gdf = load_vector_data(gdb_path, layer=layer_name)
            logger.info(f"Loaded {len(self.gdf)} features from layer '{layer_name}'. Initial CRS: {self.gdf.crs}")

            # --- Data Validation ---
            if "geometry" not in self.gdf.columns:
                logger.error("No 'geometry' column found in contour data.")
                self.gdf = None
                return
            self.gdf = self.gdf.dropna(subset=["geometry"])
            self.gdf = self.gdf[~self.gdf.geometry.is_empty]
            if self.gdf.empty:
                logger.warning("No valid contour geometries found after loading.")
                self.gdf = None
                return

            if not elevation_field or elevation_field not in self.gdf.columns:
                logger.error(
                    f"Configured elevation field '{elevation_field}' not found in contour data. "
                    "Update `n50_contour_elevation_field` in config."
                )
                self.gdf = None
                return
            else:
                # Store the original field name for potential use, though we'll sanitize for WBT
                self.elevation_field_name = elevation_field
                logger.info(f"Using elevation field: '{self.elevation_field_name}'")

            # Ensure elevation field is numeric, dropping invalid rows
            self.gdf[self.elevation_field_name] = pd.to_numeric(
                self.gdf[self.elevation_field_name], errors="coerce"
            )
            original_len = len(self.gdf)
            self.gdf = self.gdf.dropna(subset=[self.elevation_field_name])
            if len(self.gdf) < original_len:
                logger.warning(
                    f"Dropped {original_len - len(self.gdf)} contours due to non-numeric elevation values."
                )

            if self.gdf.empty:
                logger.error("No valid contours remaining after elevation validation.")
                self.gdf = None
                return

            # --- Reprojection and Saving Intermediate ---
            # self.gdf = self._reproject_if_needed(self.gdf) # Uses target_crs from settings
            self.gdf = self._reproject_if_needed(self.gdf) # Uses target_crs from settings
            logger.info(f"Contour GDF CRS after potential reprojection: {self.gdf.crs}")
            self._save_intermediate_gdf(self.gdf, "prepared_contours_gpkg")
            logger.info("N50 Contours loaded and preprocessed successfully.")

        except Exception as e:
            logger.error(f"Error loading N50 contour data: {e}", exc_info=True)
            self.gdf = None

    def _assign_crs_to_raster(self, raster_path: Path, target_epsg: int):
        """
        Assigns the target CRS to a raster file by rewriting it.
        This is generally safer than modifying in place.
        Returns True on success, False on failure.
        """
        if not raster_path.exists():
            logger.error(f"Cannot assign CRS: Raster file not found at {raster_path}")
            return False

        temp_output_path = raster_path.with_suffix(f".temp_crs{raster_path.suffix}")
        success = False
        try:
            with rasterio.open(raster_path) as src:
                profile = src.profile
                current_crs = src.crs
                target_crs_obj = rasterio.crs.CRS.from_epsg(target_epsg)

                if current_crs == target_crs_obj:
                    logger.info(f"CRS for {raster_path.name} is already correct (EPSG:{target_epsg}). No rewrite needed.")
                    return True # Already correct

                logger.info(f"Rewriting {raster_path.name} to assign CRS EPSG:{target_epsg} (Current: {current_crs}).")
                profile['crs'] = target_crs_obj
                # Ensure compression settings are preserved if they exist
                if 'compress' in profile:
                    logger.debug(f"Preserving compression: {profile['compress']}")

                with rasterio.open(temp_output_path, 'w', **profile) as dst:
                    dst.write(src.read()) # Copy all bands

            # Replace original file with the new one
            temp_output_path.replace(raster_path)
            logger.info(f"Successfully assigned CRS to {raster_path.name}.")
            success = True

        except Exception as e:
            logger.error(f"Failed to assign CRS to raster {raster_path.name}: {e}", exc_info=True)
            success = False
        finally:
            # Clean up temp file if it exists and assignment failed or succeeded
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except OSError as unlink_e:
                     logger.warning(f"Could not remove temporary CRS file {temp_output_path}: {unlink_e}")

        return success

    def _postprocess_raster_for_visualization(self, raster_path, src_epsg=25833, dst_epsg=4326):
        """
        Post-processes a raster for visualization by:
        1. Ensuring the correct source CRS is assigned (EPSG:25833)
        2. Reprojecting to the visualization CRS (EPSG:4326)
        
        Args:
            raster_path: Path to the raster file
            src_epsg: Source EPSG code, defaults to 25833
            dst_epsg: Destination EPSG code, defaults to 4326
            
        Returns:
            Path to the reprojected raster
        """
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        
        if not isinstance(raster_path, Path):
            raster_path = Path(raster_path)
            
        if not raster_path.exists():
            logger.error(f"Cannot postprocess: Raster file not found at {raster_path}")
            return None
            
        # Define output path
        output_path = raster_path.with_name(f"{raster_path.stem}_4326{raster_path.suffix}")
        
        logger.info(f"Post-processing raster: Reprojecting from EPSG:{src_epsg} to EPSG:{dst_epsg}")
        
        try:
            # First ensure the source raster has the correct CRS
            if not self._assign_crs_to_raster(raster_path, src_epsg):
                logger.warning(f"Warning: Could not assign source CRS to {raster_path}. Reprojection may fail.")
            
            # Reproject to the visualization CRS
            with rasterio.open(raster_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, dst_epsg, src.width, src.height, *src.bounds
                )
                
                profile = src.profile.copy()
                profile.update({
                    'crs': rasterio.CRS.from_epsg(dst_epsg),
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=rasterio.CRS.from_epsg(dst_epsg),
                            resampling=Resampling.bilinear
                        )
            
            logger.info(f"Successfully reprojected raster to EPSG:{dst_epsg}: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error reprojecting raster {raster_path}: {e}", exc_info=True)
            if output_path.exists():
                try:
                    output_path.unlink()  # Clean up partial output
                except OSError:
                    pass
            return None

    # --- RMSE Calculation Helper Functions (Adapted from heatmap.py) ---

    def _extract_raster_values(self, points_gdf, raster_path, value_col_name="predicted_elevation"):
        """Extracts raster values at point locations."""
        if not Path(raster_path).exists():
            logger.error(f"Cannot extract values: Raster file not found at {raster_path}")
            points_gdf[value_col_name] = np.nan
            return points_gdf

        logger.info(
            f"Extracting raster values from {Path(raster_path).name} for {len(points_gdf)} points..."
        )
        coords = [(p.x, p.y) for p in points_gdf.geometry]
        try:
            with rasterio.open(raster_path) as src:
                sampled_values = [val[0] for val in src.sample(coords)]
                points_gdf[value_col_name] = sampled_values
                # Handle NoData
                nodata_val = src.nodatavals[0]
                if nodata_val is not None:
                    points_gdf[value_col_name] = points_gdf[value_col_name].replace(
                        nodata_val, np.nan
                    )
                num_nan = points_gdf[value_col_name].isnull().sum()
                if num_nan > 0:
                    logger.warning(
                        f"Found {num_nan} points with NoData/NaN predicted values."
                    )
            logger.info("Raster value extraction complete.")
            return points_gdf
        except Exception as e:
            logger.error(f"Error extracting raster values: {e}", exc_info=True)
            points_gdf[value_col_name] = np.nan
            return points_gdf

    def _calculate_rmse(self, gdf, actual_col, predicted_col):
        """Calculates RMSE, ignoring NaN predictions."""
        valid_gdf = gdf.dropna(subset=[predicted_col, actual_col]) # Ensure actual is also valid
        if valid_gdf.empty:
            logger.warning(
                f"Cannot calculate RMSE: No valid prediction/actual pairs found (Predicted: '{predicted_col}', Actual: '{actual_col}')."
            )
            return np.nan
        if len(valid_gdf) < len(gdf):
            logger.info(
                f"Calculating RMSE using {len(valid_gdf)} points with valid predictions/actuals (out of {len(gdf)} total)."
            )

        actual = valid_gdf[actual_col]
        predicted = valid_gdf[predicted_col]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        logger.info(f"Calculated RMSE ({actual_col} vs {predicted_col}): {rmse:.4f}")
        return rmse

    def _save_rmse_results(
            self, 
            train_rmse, 
            test_rmse, 
            interpolation_method, 
            cell_size, 
            num_train_points, 
            num_test_points
        ):
        """Saves RMSE results to a CSV file."""
        try:
            results_csv_path = (
                self.settings.paths.output_dir.parent
                / "elevation_rmse_results.csv" # Save one level up
            )
            file_exists = results_csv_path.is_file()
            results_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_output_dir": self.settings.paths.output_dir.name,
                "interpolation_method": interpolation_method,
                "cell_size": cell_size,
                "train_rmse": f"{train_rmse:.4f}" if not np.isnan(train_rmse) else "NaN",
                "test_rmse": f"{test_rmse:.4f}" if not np.isnan(test_rmse) else "NaN",
                "num_train_points": num_train_points,
                "num_test_points": num_test_points,
                "dem_tin_max_triangle_edge_length": self.settings.processing.dem_tin_max_triangle_edge_length,
                "dem_idw_weight": self.settings.processing.dem_idw_weight,
                "dem_idw_radius": self.settings.processing.dem_idw_radius,
                "dem_idw_min_points": self.settings.processing.dem_idw_min_points,
            }
            with open(results_csv_path, "a", newline="") as csvfile:
                fieldnames = results_data.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(results_data)
            logger.info(f"Elevation RMSE results appended to: {results_csv_path}")
        except Exception as csv_e:
            logger.error(f"Error saving Elevation RMSE results to CSV: {csv_e}", exc_info=True)

    def _evaluate_dem_with_rmse(self, dem_path, train_gdf, test_gdf, sanitized_elev_field, interpolation_method, cell_size, num_train_points):
        """Calculates RMSE using test points against the generated DEM and saves results."""
        if not dem_path.exists():
            logger.warning(f"DEM file not found at {dem_path}. Skipping RMSE calculation.")
            return
        if train_gdf is None or train_gdf.empty or test_gdf is None or test_gdf.empty:
            logger.warning("Test data is missing or empty. Skipping RMSE calculation.")
            return

        logger.info("--- Calculating RMSE on Test Set ---")
        train_gdf_pred = self._extract_raster_values(
            train_gdf.copy(),
            dem_path,
            value_col_name="predicted_elevation"
        )
        test_gdf_pred = self._extract_raster_values(
            test_gdf.copy(),
            dem_path,
            value_col_name="predicted_elevation"
        )
        train_rmse = self._calculate_rmse(
            train_gdf_pred,
            actual_col=sanitized_elev_field, # The original elevation field
            predicted_col="predicted_elevation" # The extracted values
        )
        test_rmse = self._calculate_rmse(
            test_gdf_pred,
            actual_col=sanitized_elev_field, # The original elevation field
            predicted_col="predicted_elevation" # The extracted values
        )

        # Save RMSE results
        self._save_rmse_results(
            train_rmse=train_rmse,
            test_rmse=test_rmse,
            interpolation_method=interpolation_method,
            cell_size=cell_size,
            num_train_points=num_train_points,
            num_test_points=len(test_gdf) # Already checked test_gdf is not None
        )
        logger.info("--- RMSE Calculation Complete ---")


    @dask.delayed
    def _build_dem_and_slope(self):
        """
        Generates DEM and Slope rasters from contours using the configured
        interpolation method (natural_neighbor, tin, or idw) via WhiteboxTools,
        using a training subset of points derived from contours.
        Also calculates RMSE on a held-out test set of points.

        Returns:
            Tuple[Optional[str], Optional[str]]: Paths to the generated DEM and
            Slope rasters as strings, or (None, None) if generation fails.
        """
        logger.info("Starting DEM and Slope raster generation process (with RMSE)...")
        if self.gdf is None or not hasattr(self, "elevation_field_name"):
            logger.error(
                "Contour data (self.gdf) or elevation field name not available. "
                "Ensure load_data() was successful."
            )
            return None, None

        if self.wbt is None:
            logger.error("WhiteboxTools instance not available. Cannot generate rasters.")
            return None, None

        # --- Configuration ---
        interpolation_method = self.settings.processing.interpolation_method_dem.lower()
        cell_size = self.settings.processing.output_cell_size
        slope_units = self.settings.processing.slope_units
        
        # Paths for processing (25833) and visualization (4326)
        dem_path_key = "elevation_dem_raster"
        slope_path_key = "slope_raster"
        dem_path_25833 = self._get_output_path(dem_path_key)
        slope_path_25833 = self._get_output_path(slope_path_key)
        dem_path_25833.parent.mkdir(parents=True, exist_ok=True)

        # --- Prepare Data ---
        train_gdf, test_gdf, sanitized_elev_field = self._preprocess_and_split_points()
        if train_gdf is None or test_gdf is None or sanitized_elev_field is None:
            logger.error("Preprocessing and splitting points failed. Cannot proceed.")
            return None, None

        # --- Create temporary directory for WBT input ---
        with tempfile.TemporaryDirectory(prefix="elevation_wbt_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_wbt_path = temp_dir / "train_points.shp"
            
            try:
                # Save training points to shapefile for WBT
                save_vector_data(
                    train_gdf[[sanitized_elev_field, "geometry"]],
                    input_wbt_path,
                    driver="ESRI Shapefile"
                )
                logger.info(f"Saved temporary training points file: {input_wbt_path} ({len(train_gdf)} points)")
                
                # --- Run WBT Interpolation ---
                logger.info(f"Running WBT {interpolation_method} interpolation...")
                try:
                    if interpolation_method in ("natural_neighbor", "nn"):
                        self.wbt.natural_neighbour_interpolation(
                            i=str(input_wbt_path),
                            field=sanitized_elev_field,
                            output=str(dem_path_25833),
                            cell_size=cell_size,
                        )
                    elif interpolation_method == "tin":
                        self.wbt.tin_gridding(
                            i=str(input_wbt_path),
                            field=sanitized_elev_field,
                            output=str(dem_path_25833),
                            resolution=cell_size,
                            max_triangle_edge_length=self.settings.processing.dem_tin_max_triangle_edge_length,
                        )
                    elif interpolation_method == "idw":
                        self.wbt.idw_interpolation(
                            i=str(input_wbt_path),
                            field=sanitized_elev_field,
                            output=str(dem_path_25833),
                            cell_size=cell_size,
                            weight=self.settings.processing.dem_idw_weight,
                            radius=self.settings.processing.dem_idw_radius,
                            min_points=self.settings.processing.dem_idw_min_points,
                        )
                    else:
                        logger.error(f"Unsupported interpolation method: {interpolation_method}")
                        return None, None
                        
                except Exception as e:
                    logger.error(f"Error during WBT interpolation: {e}", exc_info=True)
                    return None, None
                    
                if not dem_path_25833.exists():
                    logger.error(f"WBT interpolation completed but output file not found: {dem_path_25833}")
                    return None, None
                    
                # Assign CRS to DEM (WBT output lacks CRS information)
                if not self._assign_crs_to_raster(dem_path_25833, 25833):
                    logger.error(f"Failed to assign CRS to DEM: {dem_path_25833}")
                    return None, None
                
                # --- Calculate Slope ---
                logger.info(f"Calculating slope from DEM (Units: {slope_units})...")
                try:
                    self.wbt.slope(
                        dem=str(dem_path_25833),
                        output=str(slope_path_25833),
                        units=slope_units
                    )
                except Exception as e:
                    logger.error(f"Error during slope calculation: {e}", exc_info=True)
                    return str(dem_path_25833), None
                    
                if not slope_path_25833.exists():
                    logger.error(f"Slope calculation completed but output file not found: {slope_path_25833}")
                    return str(dem_path_25833), None
                    
                # Assign CRS to Slope (WBT output lacks CRS information)
                if not self._assign_crs_to_raster(slope_path_25833, 25833):
                    logger.error(f"Failed to assign CRS to Slope: {slope_path_25833}")
                    return str(dem_path_25833), None
                
                # --- Evaluate DEM with RMSE ---
                # Both train and test are also in 25833 after preprocessing
                self._evaluate_dem_with_rmse(
                    dem_path=dem_path_25833,
                    train_gdf=train_gdf,
                    test_gdf=test_gdf,
                    sanitized_elev_field=sanitized_elev_field,
                    interpolation_method=interpolation_method,
                    cell_size=cell_size,
                    num_train_points=len(train_gdf)
                )
                
                # --- Reproject for Visualization (EPSG:4326) ---
                logger.info("Reprojecting rasters to EPSG:4326 for visualization...")
                dem_path_4326 = self._postprocess_raster_for_visualization(dem_path_25833)
                slope_path_4326 = self._postprocess_raster_for_visualization(slope_path_25833)
                
                if dem_path_4326 is None:
                    logger.warning("Failed to reproject DEM to EPSG:4326. Using EPSG:25833 version.")
                    dem_path_final = dem_path_25833
                else:
                    logger.info(f"DEM reprojected to EPSG:4326: {dem_path_4326}")
                    dem_path_final = dem_path_4326
                
                if slope_path_4326 is None:
                    logger.warning("Failed to reproject Slope to EPSG:4326. Using EPSG:25833 version.")
                    slope_path_final = slope_path_25833
                else:
                    logger.info(f"Slope reprojected to EPSG:4326: {slope_path_4326}")
                    slope_path_final = slope_path_4326
                
                # Store paths in object for reference
                self.output_paths[dem_path_key] = dem_path_final
                self.output_paths[slope_path_key] = slope_path_final
                
                return str(dem_path_final), str(slope_path_final)
                
            except Exception as e:
                logger.error(f"Unexpected error during DEM/Slope generation: {e}", exc_info=True)
                return None, None

    def _preprocess_and_split_points(self):
        """
        Converts contours to points, splits into train/test sets,
        and prepares them in the correct CRS for WBT processing (EPSG:25833).
        
        Returns:
            tuple: (train_gdf, test_gdf, sanitized_elev_field) or (None, None, None) on failure
        """
        logger.info("Preprocessing contours for interpolation...")
        
        # Sanitize elevation field name for shapefile compatibility (max 10 chars, alphanumeric)
        sanitized_elev_field = "".join(filter(str.isalnum, self.elevation_field_name))[:10]
        if not sanitized_elev_field:
            sanitized_elev_field = "elev"  # Default if name is unusable
        logger.info(f"Using sanitized elevation field for WBT: '{sanitized_elev_field}'")
        
        try:
            # Create a working copy of the GDF
            gdf_working = self.gdf.copy()
            
            # Rename the elevation field in the working copy
            gdf_working = gdf_working.rename(
                columns={self.elevation_field_name: sanitized_elev_field}
            )
            
            # Convert all contours to points first
            all_points_gdf = polyline_to_points(
                gdf_working[[sanitized_elev_field, "geometry"]]
            )
            if all_points_gdf.empty:
                logger.error("No points generated from contours. Cannot proceed.")
                return None, None, None
                
            logger.info(f"Generated {len(all_points_gdf)} total points from contours.")
            
            # Split points into training and testing sets
            logger.info("Splitting points into training and testing sets...")
            train_gdf, test_gdf = train_test_split(
                all_points_gdf,
                train_size=self.settings.processing.train_test_split_fraction,
                random_state=self.settings.processing.seed
            )
            
            if train_gdf.empty or test_gdf.empty:
                logger.error("Train or test set is empty after splitting. Cannot proceed.")
                return None, None, None
                
            logger.info(f"Train set size: {len(train_gdf)}, Test set size: {len(test_gdf)}")
            
            # Convert to EPSG:25833 for WBT processing
            train_gdf = self._prepare_for_wbt(train_gdf)
            test_gdf = self._prepare_for_wbt(test_gdf)
            
            logger.info(f"Prepared points for WBT: CRS={train_gdf.crs}")
            
            return train_gdf, test_gdf, sanitized_elev_field
            
        except Exception as e:
            logger.error(f"Error preprocessing contours: {e}", exc_info=True)
            return None, None, None

    def build(self):
        """
        Builds DEM and Slope rasters from the loaded N50 contour data.

        Returns:
            A dask.delayed task that resolves to a tuple (dem_path_str, slope_path_str)
            or (None, None) if generation fails.
        """
        # Ensure data is loaded first
        if self.gdf is None:
            logger.info("Contour data not loaded. Attempting to load now...")
            self.load_data()

        if self.gdf is None: # Still None if loading failed
            logger.error("Cannot build Elevation features: Data loading failed.")
            return dask.delayed(lambda: (None, None))() # Return delayed task with None values

        # Get the delayed task for DEM and Slope generation
        logger.info("Creating delayed task for DEM and Slope generation...")
        task = self._build_dem_and_slope() # This returns a dask.delayed object for (dem_path, slope_path)

        # Return the task directly - this maintains backward compatibility with build_features.py
        logger.info("Returning delayed task for DEM/Slope computation.")
        return task

if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    from src.config import settings 
    from whitebox import WhiteboxTools
    from src.utils import display_raster_on_folium_map, display_multi_layer_on_folium_map

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # handlers=[  NOTE: Uncomment if you want to log to a file
        #     logging.FileHandler("elevation.log"), # Keep specific log file
        #     logging.StreamHandler()
        # ]
    )
    logger.info("--- Running elevation.py Standalone Test ---")

    # Check if settings are loaded correctly
    if settings:
        # --- Basic Setup ---
        settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using Output Directory: {settings.paths.output_dir}")
        interpolation_method_config = settings.processing.interpolation_method_dem
        logger.info(f"Configured DEM Interpolation Method: {interpolation_method_config}")
        logger.info(f"Interpolation CRS: EPSG:{settings.processing.interpolation_crs_epsg}")
        logger.info(f"Map Visualization CRS: EPSG:{settings.processing.map_crs_epsg}")

        # --- Initialize WBT ---
        wbt = None
        try:
            wbt = WhiteboxTools()
            # Optional: Set verbosity, working dir etc. if needed for testing
            wbt.set_verbose_mode(settings.processing.wbt_verbose)
            logger.info("WhiteboxTools initialized.")
        except Exception as wbt_e:
            logger.error(f"Failed to initialize WhiteboxTools: {wbt_e}", exc_info=True)
            wbt = None # Ensure wbt is None if init fails

        # --- Setup Dask Client (Optional but recommended) ---
        cluster = None
        client = None
        try:
            # Simple local setup for testing
            cluster = LocalCluster(n_workers=1, threads_per_worker=2, memory_limit='4GB')
            client = Client(cluster)
            logger.info(f"Dask client started: {client.dashboard_link}")
        except Exception as dask_e:
            logger.error(f"Failed to start Dask client: {dask_e}", exc_info=True)
            # Test can proceed without Dask, but dask.compute will run sequentially

        # --- Test Elevation Feature ---
        elevation_feature = None
        try:
            if wbt is None:
                raise RuntimeError("Cannot run test: WhiteboxTools failed to initialize.")

            logger.info("--- Testing Elevation Feature ---")
            # Pass the initialized wbt instance
            elevation_feature = Elevation(settings, wbt)

            logger.info("1. Testing Elevation Load Data...")
            elevation_feature.load_data() # load_data logs its own success/failure

            # Only proceed to build if data loaded successfully
            if elevation_feature.gdf is not None:
                logger.info("2. Testing Elevation Build (DEM and Slope)...")
                # Build now returns a single delayed task or None
                delayed_task = elevation_feature.build()

                if delayed_task is not None:
                    logger.info("Received delayed task from build(). Computing...")
                    # Compute the single task. dask.compute returns a tuple.
                    # The task itself returns a tuple (dem_path_str, slope_path_str) or (None, None)
                    computed_results = dask.compute(delayed_task)
                    dem_result, slope_result = computed_results[0] if computed_results and isinstance(computed_results[0], tuple) and len(computed_results[0]) == 2 else (None, None)
                    logger.info("Build computation completed.")

                    # Process the computed results
                    raster_paths = []
                    if dem_result and Path(dem_result).exists():
                        logger.info(f"DEM generation successful: {dem_result}")
                        raster_paths.append(dem_result)
                    else:
                        logger.warning("DEM generation appears to have failed or file not found after compute.")

                    if slope_result and Path(slope_result).exists():
                        logger.info(f"Slope generation successful: {slope_result}")
                        raster_paths.append(slope_result)
                    else:
                        logger.warning("Slope generation appears to have failed or file not found after compute.")

                    # --- Display Generated Rasters ---
                    if len(raster_paths) > 0:
                        logger.info("--- Displaying Generated Rasters ---")
                        
                        # # 1. First create individual maps for each raster (backward compatibility)
                        # for path_str in raster_paths:
                        #     path_obj = Path(path_str)
                        #     logger.info(f"Processing single raster for display: {path_obj.name}")
                        #     if not path_obj.exists():
                        #         logger.warning(f"  - Raster file not found: {path_obj}. Skipping display.")
                        #         continue
                                
                        #     try:
                        #         # Define output path for the individual map
                        #         single_map_output_path = settings.paths.output_dir / f"{path_obj.stem}_single_map.html"
                                
                        #         # Determine colormap based on file type
                        #         cmap = 'terrain' if 'dem' in path_obj.stem.lower() else 'coolwarm' 
                        #         logger.info(f"  - Using colormap: {cmap}")
                                
                        #         # Display using the single-layer function
                        #         display_raster_on_folium_map(
                        #             raster_path_str=str(path_obj),
                        #             output_html_path_str=str(single_map_output_path),
                        #             target_crs_epsg=settings.processing.map_crs_epsg,  # Use map_crs_epsg (4326) for visualizing
                        #             cmap_name=cmap,
                        #             layer_name=path_obj.stem
                        #         )
                        #         logger.info(f"  - Saved single-layer visualization: {single_map_output_path}")
                        #     except Exception as display_e:
                        #         logger.error(f"  - Error displaying raster {path_str} on individual map: {display_e}", exc_info=True)
                        
                        # 2. Create a multi-layer map with all successfully generated rasters
                        try:
                            # Only proceed if we have at least one valid raster
                            if len(raster_paths) > 0:
                                logger.info("Creating multi-layer map with all generated rasters...")
                                
                                # Define output path for the multi-layer map
                                multi_map_output_path = settings.paths.output_dir / f"elevation_multi_layer_map.html"
                                
                                # Prepare layer configurations
                                layers = []
                                
                                # Add DEM layer if available
                                if dem_result and Path(dem_result).exists():
                                    dem_layer = {
                                        'path': dem_result,
                                        'name': 'Digital Elevation Model',
                                        'type': 'raster',
                                        'raster': {
                                            'cmap': 'terrain',
                                            'opacity': 0.7,
                                            'show': True,  # Show by default
                                            'target_crs_epsg': settings.processing.map_crs_epsg
                                        }
                                    }
                                    layers.append(dem_layer)
                                
                                # Add Slope layer if available
                                if slope_result and Path(slope_result).exists():
                                    slope_layer = {
                                        'path': slope_result,
                                        'name': 'Slope',
                                        'type': 'raster',
                                        'raster': {
                                            'cmap': 'coolwarm',
                                            'opacity': 0.7,
                                            'show': False,  # Hide by default (can be toggled on)
                                            'target_crs_epsg': settings.processing.map_crs_epsg
                                        }
                                    }
                                    layers.append(slope_layer)
                                
                                # Create multi-layer map
                                display_multi_layer_on_folium_map(
                                    layers=layers,
                                    output_html_path_str=str(multi_map_output_path),
                                    map_zoom=11,
                                    map_tiles='CartoDB positron'
                                )
                                
                                logger.info(f"Multi-layer elevation map saved to: {multi_map_output_path}")
                            else:
                                logger.warning("No valid rasters available for multi-layer display.")
                                
                        except ImportError as import_err:
                            logger.warning(f"Could not create multi-layer map: Missing dependency ({import_err}). Install folium and branca.")
                        except Exception as multi_map_err:
                            logger.error(f"Error creating multi-layer map: {multi_map_err}", exc_info=True)
                    
                    else:
                        logger.warning("No raster paths generated by build process. Skipping map display.")
                else:
                    logger.error("Build() returned None instead of a delayed task. Skipping computation.")
            else:
                logger.error("Skipping Elevation build test as data loading failed.")

            logger.info("--- Elevation Feature Test Completed ---")

        except Exception as e:
            logger.error(f"Error during Elevation test run: {e}", exc_info=True)
        finally:
            # --- Clean up Dask Client and Cluster ---
            logger.info("Cleaning up Dask client and cluster...")
            if client:
                try:
                    client.close()
                    logger.info("Dask client closed.")
                except Exception as e:
                    logger.warning(f"Error closing Dask client: {e}")
            if cluster:
                try:
                    cluster.close()
                    logger.info("Dask cluster closed.")
                except Exception as e:
                    logger.warning(f"Error closing Dask cluster: {e}")
            logger.info("Dask cleanup finished.")
    else:
        # This case handles if settings failed to load initially
        logger.error("Settings could not be loaded. Cannot run standalone test.")

    logger.info("--- Standalone Test Finished ---")
