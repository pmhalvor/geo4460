import geopandas as gpd
import numpy as np
import pandas as pd
import traceback
from pathlib import Path
from whitebox import WhiteboxTools
from math import sqrt
from sklearn.metrics import mean_squared_error
from typing import Optional, Dict, Tuple, List, Set

from pydantic import BaseModel


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
        """Helper for logging messages."""
        if level == "info":
            prefix = "  -"
        elif level == "warning":
            prefix = "    - WARNING:"
        elif level == "error":
            prefix = "    - ERROR:"
        else:
            prefix = f"    - {level.upper()}:"
        print(f"{prefix} {message}")

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
        Copies the original points shapefile to a working location.

        Returns:
            bool: True if copying was successful, False otherwise.
        """
        self._log(
            f"Copying original points shapefile to {self.points_extracted_path} for modification..."
        )
        try:
            # Ensure output directory exists
            self.points_extracted_path.parent.mkdir(parents=True, exist_ok=True)
            points_orig_gdf = gpd.read_file(str(self.points_shp_path))
            points_orig_gdf.to_file(
                str(self.points_extracted_path), driver="ESRI Shapefile"
            )
            self._log("Copy complete.")
            self.points_extracted_gdf = points_orig_gdf  # Store initial GDF
            return True
        except Exception as e:
            self._log(f"Failed to copy points shapefile: {e}", level="error")
            # Potentially log traceback here
            return False

    def _rename_extracted_column(
        self,
        gdf: gpd.GeoDataFrame,
        cols_before_extraction: Set[str],
        target_col_name: str,
        dem_key: str,  # For logging
    ) -> Tuple[gpd.GeoDataFrame, Optional[str]]:
        """
        Robustly renames the newly added column after WBT extraction.

        Args:
            gdf: The GeoDataFrame after extraction.
            cols_before_extraction: Set of column names before the extraction.
            target_col_name: The desired final name for the extracted column.
            dem_key: The key identifying the DEM (e.g., 'dem_interp') for logging.

        Returns:
            Tuple containing the updated GeoDataFrame and the final column name (or None if failed).
        """
        cols_after_extraction = set(gdf.columns)
        new_cols = list(cols_after_extraction - cols_before_extraction)
        final_col_name = None

        if len(new_cols) == 1:
            added_col = new_cols[0]
            if added_col == target_col_name:
                self._log(
                    f"Extracted column already named '{target_col_name}' for {dem_key}."
                )
                final_col_name = target_col_name
            else:
                gdf.rename(columns={added_col: target_col_name}, inplace=True)
                self._log(
                    f"Renamed extracted column '{added_col}' to '{target_col_name}' for {dem_key}"
                )
                final_col_name = target_col_name
        elif (
            "VALUE1" in cols_after_extraction
            and target_col_name not in cols_after_extraction
        ):
            # Fallback: WBT sometimes defaults to VALUE1
            if "VALUE1" in new_cols:
                self._log(
                    f"Attempting to rename newly added 'VALUE1' to '{target_col_name}' for {dem_key}.",
                    level="warning",
                )
            else:
                # VALUE1 exists but wasn't the *only* new column, or wasn't new at all. Risky rename.
                self._log(
                    f"Attempting to rename existing 'VALUE1' to '{target_col_name}' for {dem_key} as new column not uniquely identified.",
                    level="warning",
                )
            try:
                # Check if target name already exists before renaming VALUE1
                if target_col_name in gdf.columns:
                    self._log(
                        f"Target column '{target_col_name}' already exists. Cannot rename 'VALUE1' to it.",
                        level="error",
                    )
                    final_col_name = None
                else:
                    gdf.rename(columns={"VALUE1": target_col_name}, inplace=True)
                    final_col_name = target_col_name
            except Exception as e:
                self._log(
                    f"Failed to rename 'VALUE1' to {target_col_name}: {e}",
                    level="error",
                )
                final_col_name = None  # Renaming failed
        elif target_col_name in cols_after_extraction:
            self._log(
                f"Target column '{target_col_name}' already exists for {dem_key}. Assuming it's correct.",
                level="info",
            )
            final_col_name = target_col_name  # Column already has the target name
        else:
            # This case happens if >1 new column was added, or 0 new columns were added and VALUE1 wasn't there either.
            self._log(
                f"Could not identify or rename column for {dem_key}. New cols: {new_cols}. All cols: {list(cols_after_extraction)}",
                level="warning",
            )
            final_col_name = None  # Flag that renaming failed

        return gdf, final_col_name

    def extract_dem_points(self) -> bool:
        """
        Extracts elevation values from available DEMs to the points layer.
        Handles renaming of extracted columns. Saves intermediate results.

        Returns:
            bool: True if extraction process completed without critical file I/O errors,
                  False otherwise. Individual DEM extraction failures are logged but don't cause False return.
        """
        if self.points_extracted_gdf is None:
            self._log("Initial points GeoDataFrame not loaded.", level="error")
            return False

        # Define target column names
        target_col_map = {
            "dem_interp": "DEMNN",
            "dem_topo": "DEMContour",
            "dem_toporaster_all": "ANUDEM",
            "dem_stream_burn": "DEMStream",
        }

        # Start with the GDF prepared in prepare_points_layer
        current_gdf = self.points_extracted_gdf.copy()

        # Ensure the initial state is saved before any extractions
        try:
            current_gdf.to_file(
                str(self.points_extracted_path), driver="ESRI Shapefile"
            )
        except Exception as e:
            self._log(f"Failed to save initial copied points file: {e}", level="error")
            return False  # Critical error

        for dem_key, is_available in self.available_layers.items():
            if not is_available:
                self.extracted_col_names[dem_key] = (
                    None  # Mark as unavailable (file didn't exist)
                )
                continue  # Skip this DEM

            dem_path = self.dem_paths.get(
                dem_key
            )  # Path is guaranteed to be non-None if is_available is True
            target_col_name = target_col_map.get(
                dem_key, f"DEM_{dem_key}"
            )  # Default name if not mapped

            self._log(f"Extracting values from {dem_key} ({dem_path.name})...")
            cols_before = set(current_gdf.columns)

            try:
                # WBT modifies the points file in place
                self.wbt.extract_raster_values_at_points(
                    inputs=str(dem_path),
                    points=str(self.points_extracted_path),
                )

                # Read the modified file back
                current_gdf = gpd.read_file(str(self.points_extracted_path))

                # Rename the newly added column
                current_gdf, final_col_name = self._rename_extracted_column(
                    current_gdf, cols_before, target_col_name, dem_key
                )
                self.extracted_col_names[dem_key] = (
                    final_col_name  # Store the result of renaming (can be None)
                )

                # Save the GDF state after this extraction/rename attempt
                current_gdf.to_file(
                    str(self.points_extracted_path), driver="ESRI Shapefile"
                )

            except Exception as e:
                self._log(
                    f"Failed during extraction or renaming for {dem_key}: {e}",
                    level="error",
                )
                self.extracted_col_names[dem_key] = None  # Mark as failed for this DEM
                # Attempt to reload the file to potentially continue with the next DEM
                try:
                    current_gdf = gpd.read_file(
                        str(self.points_extracted_path)
                    )  # Reload last known saved state
                    self._log(
                        "Reloaded points file state after error.", level="warning"
                    )
                except Exception as read_e:
                    self._log(
                        f"CRITICAL: Failed to reload points file after error ({read_e}). Aborting further extractions.",
                        level="error",
                    )
                    self.points_extracted_gdf = (
                        current_gdf  # Store potentially partial GDF
                    )
                    return False  # Critical error if we can't read the file

        self.points_extracted_gdf = current_gdf  # Store final GDF after all attempts
        self._log(
            f"Extraction process finished. Final points data potentially updated in: {self.points_extracted_path}"
        )
        return True

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

        # Map internal keys back to user-friendly names for the report
        dem_type_map = {
            "dem_interp": "Natural Neighbour",
            "dem_topo": "TIN (Contours)",
            "dem_toporaster_all": "TopoToRaster (ArcGIS Pro)",
            "dem_stream_burn": "TIN + Stream Burn",
        }

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
        print("\n3. Quality Assessment (RMSE)...")
        try:
            if not self.verify_inputs():
                print("--- Quality Assessment Failed (Input Verification) ---")
                return None

            if not self.prepare_points_layer():
                print("--- Quality Assessment Failed (Prepare Points Layer) ---")
                return None

            if not self.extract_dem_points():
                # This indicates a critical file I/O error during extraction
                print(
                    "--- Quality Assessment Failed (Critical Error During Point Extraction) ---"
                )
                return None
            # If it returns True, some extractions might have failed non-critically, proceed to verify

            is_gdf_valid, valid_dem_cols = self.verify_preprocessed_gdf()
            if not is_gdf_valid:
                print(
                    "--- Quality Assessment Failed (Preprocessed GDF Verification) ---"
                )
                return None

            rmse_df = self.calculate_rmse(valid_dem_cols)
            if rmse_df is None:
                print("--- Quality Assessment Failed (RMSE Calculation or Saving) ---")
                return None  # calculation_rmse returns None on failure

            print("--- Quality Assessment Complete ---")
            return rmse_df  # Return the DataFrame

        except Exception as e:
            print("\n--- Quality Assessment Failed (Unexpected Error) ---")
            self._log(f"An unexpected error occurred: {e}", level="error")
            print(traceback.format_exc())
            return None  # Indicate failure


# --- Original Function Wrapper ---
# Keep the original function signature for compatibility with the workflow script.
# It now acts as a simple wrapper around the QualityAssessor class.


def assess_dem_quality(
    settings: BaseModel,
    wbt: WhiteboxTools,
    points_shp_path: Path,
    dem_interp_path: Optional[Path],  # Allow None
    dem_topo_path: Optional[Path],  # Allow None
    dem_toporaster_all_path: Optional[Path],  # Allow None
    dem_stream_burn_path: Optional[Path],  # Allow None
    point_elev_field: Optional[str],
) -> Optional[pd.DataFrame]:  # Return DataFrame or None
    """
    Performs quality assessment by calculating RMSE between DEMs and elevation points.
    This function now wraps the QualityAssessor class.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        points_shp_path: Path to the original elevation points shapefile.
        dem_interp_path: Path to the interpolated DEM raster (or None).
        dem_topo_path: Path to the TIN gridded DEM raster (or None).
        dem_toporaster_all_path: Path to the TopoToRaster (ArcGIS Pro) DEM raster (or None).
        dem_stream_burn_path: Path to the TIN + Stream Burn DEM raster (or None).
        point_elev_field: The name of the elevation field in the points shapefile.

    Returns:
        Optional[pd.DataFrame]: DataFrame containing RMSE results, or None if assessment fails.
    """

    # Structure the DEM paths into the dictionary expected by the class
    # Ensure keys match those used internally (target_col_map, dem_type_map)
    dem_paths = {
        "dem_interp": dem_interp_path,
        "dem_topo": dem_topo_path,
        "dem_toporaster_all": dem_toporaster_all_path,
        "dem_stream_burn": dem_stream_burn_path,
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
