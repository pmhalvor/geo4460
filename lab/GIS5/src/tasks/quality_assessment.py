import geopandas as gpd
import numpy as np
import pandas as pd
import traceback
import logging
import shutil
import os
import tempfile
import rasterio
import rasterio.sample
from pathlib import Path
from whitebox import WhiteboxTools  # Keep for other potential uses, but not extraction
from math import sqrt
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Tuple, List, Set

from pydantic import BaseModel

# Get a logger for this module
logger = logging.getLogger(__name__)


class QualityAssessor:
    """
    Encapsulates the logic for assessing DEM quality by calculating RMSE
    between DEM rasters and elevation points.
    """

    def __init__(
        self,
        settings: BaseModel,
        wbt: WhiteboxTools,
        points_shp_path: Path,
        dem_paths: Dict[str, Optional[Path]],  # Allow None paths
        point_elev_field: Optional[str],
    ):
        """
        Initializes the QualityAssessor.

        Args:
            settings: The application configuration object.
            wbt: Initialized WhiteboxTools object.
            points_shp_path: Path to the original elevation points shapefile.
            dem_paths: Dictionary mapping DEM keys (e.g., 'dem_interp') to their paths (or None).
            point_elev_field: The name of the elevation field in the points shapefile.
        """
        self.settings = settings
        self.wbt = wbt
        self.points_shp_path = points_shp_path
        self.dem_paths = dem_paths  # Store the dict {key: path_or_none}
        self.point_elev_field = point_elev_field

        self.output_dir = self.settings.paths.output_dir
        self.output_files = self.settings.output_files
        self.processing = self.settings.processing

        self.points_extracted_path = self.output_files.get_full_path(
            "points_extracted_shp", self.output_dir
        )
        self.rmse_csv_path = self.output_files.get_full_path(
            "rmse_csv", self.output_dir
        )

        self.wbt.set_verbose_mode(self.processing.wbt_verbose)

        # State variables
        self.available_layers: Dict[str, bool] = (
            {}
        )  # Tracks if DEM file exists for a key
        self.points_extracted_gdf: Optional[gpd.GeoDataFrame] = None
        self.extracted_col_names: Dict[str, Optional[str]] = (
            {}
        )  # Map DEM key to final column name in GDF

    def _log(self, message: str, level: str = "info"):
        """Helper for logging messages using the module-level logger."""
        # Map simple levels to logging levels
        level_map = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "debug": logging.DEBUG,
        }
        log_level = level_map.get(level.lower(), logging.INFO)  # Default to INFO
        # Add indentation based on typical usage in this class
        indent_str = "  " if log_level == logging.INFO else "    "
        logger.log(log_level, f"{indent_str}{message}")

    def verify_inputs(self) -> bool:
        """
        Verifies that required input files exist and essential config is present.

        Returns:
            bool: True if the essential points shapefile and elevation field exist
                  and at least one DEM file exists, False otherwise.
                  Updates self.available_layers based on DEM file existence.
        """
        self._log("Verifying input files and configuration...")
        essential_inputs_ok = True

        if not self.points_shp_path.exists():
            self._log(
                f"Points shapefile not found ({self.points_shp_path}). Cannot proceed.",
                level="error",
            )
            essential_inputs_ok = False

        if self.point_elev_field is None:
            self._log(
                "Elevation field name not provided. Cannot calculate RMSE.",
                level="error",
            )
            essential_inputs_ok = False

        if not essential_inputs_ok:
            return False  # Cannot proceed without points file or elevation field name

        # Check each DEM provided in the dictionary for file existence
        found_any_dem = False
        for key, path in self.dem_paths.items():
            if path is not None and path.exists():
                self.available_layers[key] = True
                self._log(f"Found {key}: {path.name}")
                found_any_dem = True
            else:
                self.available_layers[key] = False
                if path is None:
                    self._log(f"Path for {key} is None. Skipping.", level="info")
                else:
                    self._log(
                        f"{key} file not found ({path}). Skipping.", level="warning"
                    )

        if not found_any_dem:
            self._log(
                "No DEM files found or provided. Cannot perform assessment.",
                level="error",
            )
            return False

        self._log("Input verification complete.")
        return True

    def prepare_points_layer(self) -> bool:
        """
        Copies the original points shapefile to a working location and loads it.

        Returns:
            bool: True if copying and loading was successful, False otherwise.
        """
        self._log(
            f"Copying original points shapefile to {self.points_extracted_path} for modification..."
        )
        try:
            # Ensure output directory exists
            self.points_extracted_path.parent.mkdir(parents=True, exist_ok=True)
            # Read first, then write to ensure GDF is loaded
            points_orig_gdf = gpd.read_file(str(self.points_shp_path))
            points_orig_gdf.to_file(
                str(self.points_extracted_path), driver="ESRI Shapefile"
            )
            self._log("Copy complete.")
            self.points_extracted_gdf = points_orig_gdf  # Store initial GDF
            return True
        except Exception as e:
            self._log(f"Failed to copy or load points shapefile: {e}", level="error")
            # Potentially log traceback here
            return False

    # _rename_extracted_column is no longer needed with the rasterio approach

    def extract_dem_points(self) -> bool:
        """
        Extracts elevation values from available DEMs to the points layer.
        Handles CRS reprojection of points for sampling if necessary.
        Accumulates results in memory and saves only once at the end.

        Returns:
            bool: True if the overall process completes, potentially with some
                  individual DEM extraction failures (logged).
                  False if a critical error occurs (e.g., initial load fails,
                  shapefile becomes unreadable, final save fails).
        """
        if self.points_extracted_gdf is None:
            self._log(
                "Initial points GeoDataFrame not loaded (prepare_points_layer likely failed).",
                level="error",
            )
            return False

        # Define target column names (max 10 chars for Shapefile)
        target_col_map = {
            "interp_contour": "DEMNNCont",  # Natural Neighbor (Contour)
            "topo_contour": "DEMTINCont",  # TIN (Contour)
            "interp_points": "DEMNNPts",  # Natural Neighbor (Points)
            "topo_points": "DEMTINPts",  # TIN (Points)
            "stream_burn": "DEMStream",  # Stream Burn (based on Contour TIN)
            "toporaster_all": "ANUDEM",  # ANUDEM (ArcGIS Pro) - Keep for comparison
        }

        # Start with the GDF prepared in prepare_points_layer
        # This GDF will accumulate the results in memory
        current_gdf = self.points_extracted_gdf.copy()
        points_crs = current_gdf.crs  # Get points CRS once

        extraction_errors = False  # Track if any non-critical errors occur

        for dem_key, is_available in self.available_layers.items():
            if not is_available:
                self.extracted_col_names[dem_key] = None
                continue  # Skip DEM if file wasn't found initially

            dem_path = self.dem_paths.get(dem_key)
            target_col_name = target_col_map.get(dem_key, f"DEM_{dem_key}")

            self._log(
                f"Processing extraction for {dem_key} ({dem_path.name}) using Rasterio..."
            )

            try:
                # 1. Open the DEM raster
                with rasterio.open(dem_path) as src:
                    raster_crs = src.crs
                    point_coords_for_sampling = []

                    # 2. Check CRS and reproject points *if necessary* for sampling
                    if points_crs != raster_crs:
                        self._log(
                            f"  CRS mismatch: Points ({points_crs}) vs Raster ({raster_crs}). Reprojecting points for sampling.",
                            level="info",  # Changed from warning as it's now handled
                        )
                        try:
                            # Reproject points geometry *temporarily* for sampling
                            points_geom_reprojected = current_gdf.geometry.to_crs(
                                raster_crs
                            )
                            point_coords_for_sampling = [
                                (p.x, p.y) for p in points_geom_reprojected
                            ]
                        except Exception as reproj_err:
                            self._log(
                                f"  ERROR during temporary point reprojection for {dem_key}: {reproj_err}",
                                level="error",
                            )
                            self.extracted_col_names[dem_key] = None
                            extraction_errors = True
                            continue  # Skip to next DEM if reprojection fails
                    else:
                        # CRS match, use original coordinates
                        point_coords_for_sampling = [
                            (p.x, p.y) for p in current_gdf.geometry
                        ]

                    if not point_coords_for_sampling:
                        self._log(
                            f"  No valid point coordinates available for sampling {dem_key} (original or reprojected).",
                            level="error",
                        )
                        self.extracted_col_names[dem_key] = None
                        extraction_errors = True
                        continue  # Skip to next DEM

                    # 3. Sample raster values at point locations (using original or reprojected coords)
                    # self._log(f"  Sampling raster values at {len(point_coords_for_sampling)} points...") # Keep less verbose
                    extracted_values = [
                        val[0] for val in src.sample(point_coords_for_sampling)
                    ]
                    # self._log(f"  Sampling complete. Extracted {len(extracted_values)} values.") # Keep less verbose

                # 4. Add extracted values to the main GeoDataFrame
                if len(extracted_values) == len(current_gdf):
                    # Use .loc with the original index to ensure correct assignment
                    current_gdf.loc[:, target_col_name] = extracted_values
                    self.extracted_col_names[dem_key] = target_col_name  # Mark success
                    self._log(
                        f"  Added column '{target_col_name}' to in-memory GeoDataFrame."
                    )
                else:
                    self._log(
                        f"  Error: Number of extracted values ({len(extracted_values)}) does not match number of points ({len(current_gdf)}). Skipping column add for {dem_key}.",
                        level="error",
                    )
                    self.extracted_col_names[dem_key] = None  # Mark failure
                    extraction_errors = True

            except rasterio.RasterioIOError as rio_err:
                self._log(
                    f"  ERROR opening or reading raster {dem_key}: {rio_err}",
                    level="error",
                )
                self.extracted_col_names[dem_key] = None
                extraction_errors = True
            except Exception as e:
                self._log(
                    f"  Failed during Rasterio extraction process for {dem_key}: {e}",
                    level="error",
                )
                self._log(f"  Traceback: {traceback.format_exc()}", level="debug")
                self.extracted_col_names[dem_key] = None  # Mark as failed for this DEM
                extraction_errors = True  # Mark that an error occurred

        # --- After the loop ---
        if extraction_errors:
            self._log(
                "Extraction process encountered errors for some DEMs using Rasterio. Proceeding with available data.",
                level="warning",
            )

        # 5. Save the final accumulated GDF (in memory) to the *actual* target shapefile ONCE
        try:
            self._log(
                f"Saving final accumulated points data to {self.points_extracted_path}..."
            )
            # Ensure the output directory exists before final save
            self.points_extracted_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure the index is handled correctly on final save
            current_gdf.to_file(
                str(self.points_extracted_path), index=False, driver="ESRI Shapefile"
            )
            self._log("Final save complete.")
            self.points_extracted_gdf = (
                current_gdf  # Update the instance variable with the final GDF
            )
            return (
                True  # Indicate overall process finished, possibly with partial results
            )
        except Exception as e:
            self._log(
                f"CRITICAL: Failed to save final points GeoDataFrame to {self.points_extracted_path}: {e}",
                level="error",
            )
            # Store the GDF we attempted to save, even if save failed
            self.points_extracted_gdf = current_gdf
            return False  # Critical failure on final save

    def verify_preprocessed_gdf(self) -> Tuple[bool, List[str]]:
        """
        Verifies the final GeoDataFrame has the required columns for RMSE calculation.

        Returns:
            Tuple[bool, List[str]]:
                - bool: True if the GDF is valid for RMSE calculation, False otherwise.
                - List[str]: List of valid DEM column names found in the GDF.
        """
        self._log("Verifying processed GeoDataFrame...")
        if self.points_extracted_gdf is None:
            self._log("Processed points GeoDataFrame is not available.", level="error")
            return False, []

        present_cols = self.points_extracted_gdf.columns.tolist()
        required_base_col = self.point_elev_field

        # This check should have been caught by verify_inputs, but double-check
        if required_base_col is None or required_base_col not in present_cols:
            self._log(
                f"Missing the original elevation point field '{required_base_col}' in the final GDF. Cannot calculate RMSE.",
                level="error",
            )
            return False, []

        # Identify columns that were successfully extracted *and* renamed *and* are still present
        valid_dem_cols = []
        for key, col_name in self.extracted_col_names.items():
            if col_name is not None and col_name in present_cols:
                valid_dem_cols.append(col_name)
            elif self.available_layers.get(key) and col_name:  # Expected but missing
                self._log(
                    f"Column '{col_name}' for {key} was expected but is missing from the final GDF.",
                    level="warning",
                )
            # No log needed if col_name is None (extraction/rename failed earlier) or layer wasn't available

        if not valid_dem_cols:
            self._log(
                "No valid DEM columns found in the final GDF. Cannot calculate RMSE.",
                level="warning",
            )
            return False, []

        self._log(
            f"Preprocessed GDF verified. Found elevation field '{required_base_col}' and DEM columns: {valid_dem_cols}"
        )
        return True, valid_dem_cols

    def calculate_rmse(self, valid_dem_cols: List[str]) -> Optional[pd.DataFrame]:
        """
        Calculates RMSE for each valid DEM column against the original elevation points.

        Args:
            valid_dem_cols: List of DEM column names confirmed to exist in the GDF.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing RMSE results, or None if calculation fails.
        """
        if self.points_extracted_gdf is None or self.point_elev_field is None:
            self._log(
                "Cannot calculate RMSE: Missing GDF or elevation field name.",
                level="error",
            )
            return None

        self._log("Calculating RMSE...")

        # Ensure columns are numeric, coercing errors to NaN
        cols_to_check = [self.point_elev_field] + valid_dem_cols
        gdf_numeric = self.points_extracted_gdf[cols_to_check].copy()
        for col in cols_to_check:
            gdf_numeric[col] = pd.to_numeric(gdf_numeric[col], errors="coerce")

        # Clean data: Replace Inf with NaN, then drop rows with NaN in essential columns
        gdf_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows where *any* of the essential columns are NaN
        points_valid_gdf = gdf_numeric.dropna(subset=cols_to_check).copy()
        n_points = len(points_valid_gdf)

        if n_points == 0:
            self._log(
                f"No valid points remaining after cleaning NaN/Inf values across columns: {cols_to_check}. Cannot calculate RMSE.",
                level="error",
            )
            return None

        measured = points_valid_gdf[self.point_elev_field]
        rmse_results_list = []  # Use a list of dicts

        # Access the map from the settings object
        dem_type_map = self.settings.output_files.dem_type_map

        # Calculate RMSE for each column that survived the cleaning
        for dem_key, col_name in self.extracted_col_names.items():
            # Check if the column was originally valid AND exists after cleaning
            if col_name in valid_dem_cols and col_name in points_valid_gdf.columns:
                dem_values = points_valid_gdf[col_name]
                try:
                    # Ensure both measured and dem_values are float type for calculation
                    rmse_value = sqrt(
                        mean_squared_error(
                            measured.astype(float), dem_values.astype(float)
                        )
                    )
                    dem_type_name = dem_type_map.get(
                        dem_key, col_name
                    )  # Use mapped name or column name
                    rmse_results_list.append(
                        {
                            "DEM_Type": dem_type_name,
                            "RMSE": rmse_value,
                            "N_Points": n_points,
                        }
                    )
                    self._log(
                        f"RMSE ({dem_type_name} vs Points): {rmse_value:.3f} (using {n_points} points)"
                    )
                except Exception as e:
                    self._log(
                        f"Error calculating RMSE for {col_name}: {e}", level="error"
                    )
            elif (
                col_name in valid_dem_cols and col_name not in points_valid_gdf.columns
            ):
                # This case indicates it was valid but removed by dropna
                self._log(
                    f"Skipping RMSE for {col_name} as all corresponding points had NaN/Inf values.",
                    level="warning",
                )
            # No log needed if col_name was not in valid_dem_cols to begin with

        if not rmse_results_list:
            self._log(
                "No RMSE values could be calculated successfully.", level="warning"
            )
            return None

        # Create DataFrame and save
        rmse_df = pd.DataFrame(rmse_results_list)
        try:
            # Ensure output directory exists
            self.rmse_csv_path.parent.mkdir(parents=True, exist_ok=True)
            rmse_df.to_csv(str(self.rmse_csv_path), index=False)
            self._log(f"RMSE results saved to: {self.rmse_csv_path}")
            return rmse_df
        except Exception as e:
            self._log(f"Failed to save RMSE results CSV: {e}", level="error")
            return None  # Indicate failure

    def run_assessment(self) -> Optional[pd.DataFrame]:
        """
        Runs the entire quality assessment workflow orchestrating the steps.

        Returns:
            Optional[pd.DataFrame]: DataFrame with RMSE results, or None if any critical step failed.
        """
        logger.info("--- Starting Task 3: Quality Assessment (RMSE) ---")
        try:
            if not self.verify_inputs():
                logger.error("--- Quality Assessment Failed (Input Verification) ---")
                return None

            if not self.prepare_points_layer():
                logger.error("--- Quality Assessment Failed (Prepare Points Layer) ---")
                return None

            if not self.extract_dem_points():
                # This indicates a critical file I/O error during extraction or final save
                logger.error(
                    "--- Quality Assessment Failed (Critical Error During Point Extraction/Save) ---"
                )
                return None
            # If it returns True, some extractions might have failed non-critically, proceed to verify

            is_gdf_valid, valid_dem_cols = self.verify_preprocessed_gdf()
            if not is_gdf_valid:
                logger.error(
                    "--- Quality Assessment Failed (Preprocessed GDF Verification) ---"
                )
                return None

            rmse_df = self.calculate_rmse(valid_dem_cols)
            if rmse_df is None:
                logger.error(
                    "--- Quality Assessment Failed (RMSE Calculation or Saving) ---"
                )
                return None  # calculation_rmse returns None on failure

            logger.info("--- Quality Assessment Complete ---")
            return rmse_df  # Return the DataFrame

        except Exception as e:
            logger.error(
                "--- Quality Assessment Failed (Unexpected Error) ---", exc_info=True
            )
            # self._log(f"An unexpected error occurred: {e}", level="error") # _log doesn't support exc_info
            # print(traceback.format_exc()) # No longer needed
            return None  # Indicate failure


# --- Original Function Wrapper ---
# Keep the original function signature for compatibility with the workflow script.
# It now acts as a simple wrapper around the QualityAssessor class.


def assess_dem_quality(
    settings: BaseModel,
    wbt: WhiteboxTools,
    points_shp_path: Path,
    # Paths for the 5 generated DEMs + ANUDEM
    dem_interp_contour_path: Optional[Path],
    dem_topo_contour_path: Optional[Path],
    dem_interp_points_path: Optional[Path],
    dem_topo_points_path: Optional[Path],
    dem_stream_burn_path: Optional[Path],
    dem_toporaster_all_path: Optional[Path],  # Keep ANUDEM for comparison
    point_elev_field: Optional[str],
) -> Optional[pd.DataFrame]:  # Return DataFrame or None
    """
    Performs quality assessment by calculating RMSE between DEMs and elevation points.
    This function now wraps the QualityAssessor class and handles multiple DEM inputs.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        points_shp_path: Path to the original elevation points shapefile (with 'VALUE' field).
        dem_interp_contour_path: Path to Natural Neighbor (Contour) DEM (or None).
        dem_topo_contour_path: Path to TIN Gridding (Contour) DEM (or None).
        dem_interp_points_path: Path to Natural Neighbor (Points) DEM (or None).
        dem_topo_points_path: Path to TIN Gridding (Points) DEM (or None).
        dem_stream_burn_path: Path to Stream Burn (Contour TIN based) DEM (or None).
        dem_toporaster_all_path: Path to ANUDEM (ArcGIS Pro) DEM (or None).
        point_elev_field: The name of the elevation field ('VALUE') in the points shapefile.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing RMSE results, or None if assessment fails.
    """

    # Structure the DEM paths into the dictionary expected by the class
    # Keys should match those used in target_col_map and dem_type_map within the class
    dem_paths = {
        "interp_contour": dem_interp_contour_path,
        "topo_contour": dem_topo_contour_path,
        "interp_points": dem_interp_points_path,
        "topo_points": dem_topo_points_path,
        "stream_burn": dem_stream_burn_path,
        "toporaster_all": dem_toporaster_all_path,  # Keep ANUDEM
    }

    assessor = QualityAssessor(
        settings=settings,
        wbt=wbt,
        points_shp_path=points_shp_path,
        dem_paths=dem_paths,
        point_elev_field=point_elev_field,
    )

    results_df = assessor.run_assessment()

    # Return the results DataFrame (which could be None if assessment failed)
    return results_df
