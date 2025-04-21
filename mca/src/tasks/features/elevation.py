import logging
import dask
import geopandas as gpd
import pandas as pd
import tempfile
from pathlib import Path
import rasterio
import rasterio.crs

from src.tasks.features.feature_base import FeatureBase
from src.utils import load_vector_data, polyline_to_points, save_vector_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("elevation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Elevation(FeatureBase):
    """Handles N50 elevation contour data to generate DEM and slope."""

    def load_data(self):
        logger.info("Loading N50 contour data...")
        if (
            not self.settings.paths.n50_gdb_path
            or not self.settings.paths.n50_gdb_path.exists()
        ):
            logger.warning(
                "N50 GDB path not configured or not found. Skipping Elevation loading."
            )
            self.gdf = None
            return

        try:
            self.gdf = load_vector_data(
                self.settings.paths.n50_gdb_path,
                layer=self.settings.input_data.n50_contour_layer,
            )
            # Ensure geometry exists and is valid
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

            elevation_field = self.settings.input_data.n50_contour_elevation_field
            if not elevation_field or elevation_field not in self.gdf.columns:
                logger.error(
                    "Could not find a suitable elevation field in contour data. " \
                    "Update `n50_contour_elevation_field` in config"
                )
                self.gdf = None
                return
            else:
                self.elevation_field_name = elevation_field  

            # Ensure elevation field is numeric
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

            self.gdf = self._reproject_if_needed(self.gdf)
            self._save_intermediate_gdf(self.gdf, "prepared_contours_gpkg")
            logger.info("N50 Contours loaded and preprocessed.")
        except Exception as e:
            logger.error(f"Error loading N50 contour data: {e}", exc_info=True)
            self.gdf = None

    def _assign_crs_to_raster(self, raster_path: Path, target_epsg: int):
        """
        Rewrites a raster file, ensuring the target CRS is set in the profile.
        This is more robust than trying to modify the CRS in place.
        """
        temp_output_path = raster_path.with_suffix(f".temp_crs{raster_path.suffix}")
        try:
            with rasterio.open(raster_path) as src:
                profile = src.profile
                profile['crs'] = rasterio.crs.CRS.from_epsg(target_epsg)
                logger.info(f"Rewriting {raster_path} to temporary file {temp_output_path} with CRS EPSG:{target_epsg}")
                with rasterio.open(temp_output_path, 'w', **profile) as dst:
                    dst.write(src.read()) # Copy all bands

            # Replace original file with the new one
            temp_output_path.replace(raster_path)
            logger.info(f"Successfully rewrote {raster_path} with correct CRS.")

        except Exception as e:
            logger.error(f"Failed to rewrite raster {raster_path} with correct CRS: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_output_path.exists():
                try:
                    temp_output_path.unlink()
                except OSError:
                    pass
            # Decide if this should raise an error or just log and potentially cause downstream issues
            raise # Re-raise the exception to indicate failure in the build process

    @dask.delayed
    def _build_dem_and_slope(self):
        """
        Generates DEM and Slope rasters from contours using a configured
        interpolation method (Natural Neighbor, TIN, or IDW).
        """
        logger.info("Building DEM and Slope rasters...")
        if self.gdf is None or not hasattr(self, "elevation_field_name"):
            logger.warning(
                "Contour data or elevation field not available, skipping DEM/Slope generation."
            )
            return None, None

        # --- Configuration ---
        interpolation_method = getattr(
            self.settings.processing, "dem_interpolation_method", "natural_neighbor"
        ).lower() # Default to natural_neighbor if not set
        cell_size = self.settings.processing.output_cell_size
        slope_units = self.settings.processing.slope_units
        dem_path_key = "elevation_dem_raster"
        slope_path_key = "slope_raster"
        dem_path = self._get_output_path(dem_path_key)
        slope_path = self._get_output_path(slope_path_key)

        # --- Prepare Input Data and Field Name ---
        # Sanitize elevation field name for shapefile compatibility
        sanitized_elev_field = "".join(filter(str.isalnum, self.elevation_field_name))[
            :10
        ]
        if not sanitized_elev_field:
            sanitized_elev_field = "elev" # Default if name is unusable

        # Use a temporary directory for intermediate files
        with tempfile.TemporaryDirectory(prefix="elevation_wbt_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_path = None # Path to the file WBT will use
            requires_points = interpolation_method in ["natural_neighbor", "idw"]

            try:
                # Prepare input data (points or contours) based on method
                if requires_points:
                    logger.info(f"Method '{interpolation_method}' requires points. Converting contours...")
                    # Rename field *before* converting to points
                    gdf_renamed = self.gdf.rename(
                        columns={self.elevation_field_name: sanitized_elev_field}
                    )
                    points_gdf = polyline_to_points(
                        gdf_renamed[[sanitized_elev_field, "geometry"]]
                    )
                    if points_gdf.empty:
                        logger.error("No points generated from contours. Cannot proceed.")
                        return None, None
                    input_path = temp_dir / "contour_points.shp"
                    save_vector_data(points_gdf, input_path, driver="ESRI Shapefile")
                    logger.info(f"Saved temporary points shapefile: {input_path}")
                else: # TIN can use contours directly
                    logger.info(f"Method '{interpolation_method}' uses contours directly.")
                    # Rename field in the original contour GDF copy
                    gdf_renamed = self.gdf.rename(
                        columns={self.elevation_field_name: sanitized_elev_field}
                    )
                    input_path = temp_dir / "contours.shp"
                    save_vector_data(
                        gdf_renamed[[sanitized_elev_field, "geometry"]],
                        input_path,
                        driver="ESRI Shapefile"
                    )
                    logger.info(f"Saved temporary contours shapefile: {input_path}")

                # --- Run WBT Interpolation ---
                logger.info(
                    f"Running WBT '{interpolation_method}' interpolation (field: '{sanitized_elev_field}')..."
                )
                dem_generated = False
                if interpolation_method == "natural_neighbor":
                    self.wbt.natural_neighbour_interpolation(
                        i=str(input_path),
                        field=sanitized_elev_field,
                        output=str(dem_path),
                        cell_size=cell_size,
                    )
                    dem_generated = True
                    if dem_generated: self._assign_crs_to_raster(dem_path, self.settings.processing.output_crs_epsg)
                elif interpolation_method == "tin":
                     # TIN interpolation might need different parameters depending on WBT version/specifics
                     # Assuming basic usage here. Check WBT docs if needed.
                     # It might use the vector contours directly.
                    self.wbt.tin_interpolation(
                        i=str(input_path),
                        field=sanitized_elev_field,
                        output=str(dem_path),
                        resolution=cell_size, # Parameter name might differ
                        # Other parameters like `max_triangle_edge_length` might be useful
                    )
                    dem_generated = True
                    if dem_generated: self._assign_crs_to_raster(dem_path, self.settings.processing.output_crs_epsg)
                elif interpolation_method == "idw":
                    # IDW requires points and specific parameters (weight, radius, etc.)
                    # Get these from settings or use defaults
                    idw_weight = getattr(self.settings.processing, "dem_idw_weight", 2.0)
                    idw_radius = getattr(self.settings.processing, "dem_idw_radius", None) # None might mean search whole raster
                    idw_min_points = getattr(self.settings.processing, "dem_idw_min_points", 0) # 0 might mean no minimum

                    self.wbt.idw_interpolation(
                        i=str(input_path),
                        field=sanitized_elev_field,
                        output=str(dem_path),
                        cell_size=cell_size,
                        weight=idw_weight,
                        radius=idw_radius,
                        min_points=idw_min_points,
                    )
                    dem_generated = True
                    if dem_generated: self._assign_crs_to_raster(dem_path, self.settings.processing.output_crs_epsg)
                else:
                    logger.error(f"Unsupported interpolation method: {interpolation_method}")
                    return None, None

                if dem_generated:
                    logger.info(f"Generated DEM raster ({interpolation_method}): {dem_path}")
                    self.output_paths[dem_path_key] = dem_path # Store path with method in key
                else:
                     logger.error(f"DEM generation failed for method {interpolation_method}.")
                     return None, None # Exit if DEM failed

                # --- Calculate Slope from DEM ---
                logger.info(f"Calculating slope from {interpolation_method} DEM...")
                self.wbt.slope(
                    dem=str(dem_path),
                    output=str(slope_path),
                )
                # Assign CRS to the slope raster immediately after generation
                self._assign_crs_to_raster(slope_path, self.settings.processing.output_crs_epsg)

                logger.info(f"Generated Slope raster ({interpolation_method}): {slope_path}")
                self.output_paths[slope_path_key] = slope_path # Store path with method in key

                return str(dem_path), str(slope_path) # Return paths on success

            except Exception as e:
                logger.error(f"Error during WBT DEM/Slope generation ({interpolation_method}): {e}", exc_info=True)
                # Clean up potentially added paths on error
                if dem_path_key in self.output_paths: del self.output_paths[dem_path_key]
                if slope_path_key in self.output_paths: del self.output_paths[slope_path_key]
                return None, None
            # Temporary directory is automatically cleaned up here when 'with' block exits

    def build(self):
        """Builds DEM and Slope rasters using the configured method."""
        if self.gdf is None:
            self.load_data()
        if self.gdf is None:
            logger.error("Cannot build Elevation features: Data loading failed.")
            return [] # Return empty list for consistency

        task = self._build_dem_and_slope()
        # Compute returns a tuple, results[0] contains the output of the task (dem_path, slope_path)
        results = dask.compute(task)
        dem_result, slope_result = results[0] if results and len(results[0]) == 2 else (None, None)

        successful_outputs = []
        if dem_result:
            successful_outputs.append(dem_result)
        if slope_result:
            successful_outputs.append(slope_result)

        return successful_outputs


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    from src.config import settings
    from whitebox import WhiteboxTools
    from src.utils import display_raster_on_folium_map # Import the display function

    logger.info("--- Running elevation.py Standalone Test ---")

    # Check if settings are loaded correctly
    if settings:
        # --- Basic Setup ---
        # Ensure output directory exists
        settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using Output Directory: {settings.paths.output_dir}")
        # Log the configured interpolation method
        interpolation_method_config = getattr(settings.processing, 'dem_interpolation_method', 'natural_neighbor (default)')
        logger.info(f"Configured DEM Interpolation Method: {interpolation_method_config}")

        # --- Setup Dask Client ---
        # Initialize WhiteboxTools
        wbt = WhiteboxTools()
        # Optional: Set WBT verbosity based on settings
        # wbt.set_verbose_mode(settings.processing.wbt_verbose)

        # Setup Dask cluster and client
        cluster = None
        client = None
        try:
            cluster = LocalCluster(n_workers=1, threads_per_worker=1) # Simple setup for local testing
            client = Client(cluster)
            logger.info(f"Dask client started: {client.dashboard_link}")
        except Exception as dask_e:
            logger.error(f"Failed to start Dask client: {dask_e}", exc_info=True)
            # Decide if you want to proceed without Dask or exit
            # For now, we'll proceed, but dask.compute might fail later if client is None

        # --- Test Elevation Feature ---
        elevation_feature = None # Initialize to None
        try:
            logger.info("--- Testing Elevation Feature ---")
            # Pass the initialized wbt instance
            elevation_feature = Elevation(settings, wbt)

            logger.info("1. Testing Elevation Load Data...")
            elevation_feature.load_data()
            if elevation_feature.gdf is not None:
                # Log CRS after loading and potential reprojection in load_data
                logger.info(
                    f"N50 Contours loaded. Shape: {elevation_feature.gdf.shape}, CRS after load_data: {elevation_feature.gdf.crs}"
                )
                # Log the identified elevation field
                if hasattr(elevation_feature, 'elevation_field_name'):
                    logger.info(f"Using elevation field: '{elevation_feature.elevation_field_name}'")
            else:
                logger.error("Elevation GDF is None after loading. Cannot proceed with build.")
                # Optionally raise an error or exit if loading is critical for the test

            # Only proceed to build if data loaded successfully
            if elevation_feature.gdf is not None:
                logger.info("2. Testing Elevation Build (DEM and Slope)...")
                logger.info(f"CRS before build: {elevation_feature.gdf.crs}")
                logger.info(f"Target output CRS from settings: EPSG:{settings.processing.output_crs_epsg}")

                # Ensure Dask client is available if needed by build()
                if client is None and dask: # Check if dask is imported and client failed
                     logger.warning("Dask client not available, build might fail if using dask.delayed.")

                raster_paths = elevation_feature.build() # Returns list [dem_path, slope_path] or []
                logger.info("Elevation build process completed.")

                if raster_paths:
                    logger.info("--- Processing Generated Rasters for Display ---")
                    # Import rasterio here, only needed for CRS check in this block
                    import rasterio
                    import rasterio.crs

                    for path_str in raster_paths:
                        path_obj = Path(path_str) # Convert string path to Path object
                        logger.info(f"Processing raster: {path_obj}")
                        if not path_obj.exists():
                            logger.warning(f"Raster file not found: {path_obj}. Skipping display.")
                            continue

                        raster_crs = None
                        try:
                            with rasterio.open(path_obj) as src:
                                raster_crs = src.crs
                                logger.info(f"  - Raster CRS read from file: {raster_crs}")
                                # Check if CRS needs assignment (WBT might not always write it)
                                target_epsg = settings.processing.output_crs_epsg
                                # Updated check: Remove deprecated 'is_valid'
                                if raster_crs is None:
                                     logger.warning(f"  - Raster CRS is missing. Attempting to assign EPSG:{target_epsg}.")
                                     # Reopen in 'r+' mode to write CRS
                                     with rasterio.open(path_obj, 'r+') as dst:
                                         dst.crs = rasterio.crs.CRS.from_epsg(target_epsg)
                                     # Verify assignment
                                     with rasterio.open(path_obj) as updated_src:
                                         raster_crs = updated_src.crs
                                         logger.info(f"  - Raster CRS after assignment: {raster_crs}")
                                elif raster_crs.to_epsg() != target_epsg:
                                     logger.warning(f"  - Raster CRS ({raster_crs}) does not match target EPSG:{target_epsg}. Display function might reproject.")
                                else:
                                     logger.info(f"  - Raster CRS matches target EPSG:{target_epsg}.")

                        except Exception as rio_e:
                            logger.error(f"  - Error reading or assigning CRS for {path_obj}: {rio_e}", exc_info=True)
                            logger.warning(f"  - Skipping display for {path_obj} due to CRS read/assign error.")
                            continue # Skip display for this raster

                        # Proceed with display only if CRS seems okay or was assigned
                        try:
                            logger.info(f"  - Preparing to display raster: {path_obj}")
                            map_output_path = settings.paths.output_dir / f"{path_obj.stem}_map.html"
                            logger.info(f"  - Calling display_raster_on_folium_map with:")
                            logger.info(f"    - raster_path_str: {str(path_obj)}")
                            logger.info(f"    - output_html_path_str: {str(map_output_path)}")
                            logger.info(f"    - target_crs_epsg: {settings.processing.output_crs_epsg}")

                            # Call the utility function to display the raster
                            display_raster_on_folium_map(
                                raster_path_str=str(path_obj), # Pass path as string
                                output_html_path_str=str(map_output_path),
                                target_crs_epsg=settings.processing.output_crs_epsg,
                                cmap_name='terrain' # Good colormap for elevation/slope
                            )
                            logger.info(f"  - Saved visualization map: {map_output_path}")
                        except ImportError as import_err:
                             logger.warning(f"  - Could not display raster {path_str}: Missing dependency ({import_err}). Install folium and branca.")
                        except Exception as display_e:
                            logger.error(f"  - Error during display_raster_on_folium_map for {path_str}: {display_e}", exc_info=True)
                else:
                     logger.warning("No raster paths generated by build process. Skipping map display.")
            else:
                logger.warning("Skipping Elevation build test as data loading failed.")

            logger.info("--- Elevation Feature Test Completed ---")

        except Exception as e:
            logger.error(f"Error during Elevation test run: {e}", exc_info=True)
        finally:
            # --- Clean up Dask Client and Cluster ---
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
    else:
        # This case handles if settings failed to load initially
        logger.error("Settings could not be loaded. Cannot run standalone test.")

    logger.info("--- Standalone Test Finished ---")
