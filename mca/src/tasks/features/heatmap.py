import csv
import dask
import geopandas as gpd
import json
import logging
import numpy as np
import pandas as pd
import polyline
import rasterio
import rasterio.crs
import tempfile

from datetime import datetime
from pathlib import Path
from shapely.geometry import LineString, Point
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from whitebox import WhiteboxTools

from src.tasks.features.feature_base import FeatureBase
from src.utils import (
    save_vector_data,
    polyline_to_points,
)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Heatmap(FeatureBase):
    """
    Processes Strava activity data from JSON files, converting polyline splits
    with speed into an interpolated average speed raster using IDW.
    """

    def load_data(self):
        """
        Loads activity data from JSON files, decodes polylines, calculates
        cumulative distances, and creates LineString segments tagged with
        average speed from splits_metric.
        """
        logger.info("Loading and processing Strava activity JSON data...")
        activity_dir = self.settings.paths.activity_details_dir
        if not activity_dir.is_dir():
            logger.error(f"Activity details directory not found: {activity_dir}")
            self.gdf = None
            return

        json_files = list(activity_dir.glob("*.json"))
        if not json_files:
            logger.error(f"No JSON files found in {activity_dir}")
            self.gdf = None
            return

        logger.info(f"Found {len(json_files)} activity JSON files.")

        all_split_segments = []
        target_crs = self.settings.processing.output_crs_epsg
        logger.info(f"Target CRS read from settings: {target_crs}")

        for file_path in json_files:
            try:
                logger.debug(f"Processing file: {file_path}")
                with open(file_path, "r") as f:
                    data = json.load(f)

                activity_id = data.get("id")
                encoded_polyline = data.get("map", {}).get("polyline")
                splits_metric = data.get("splits_metric")

                if not activity_id:
                    logger.warning(f"Skipping {file_path.name}: Missing ID.")
                    continue
                elif not encoded_polyline:
                    logger.warning(f"Skipping {file_path.name}: Missing polyline.")
                    continue
                elif not splits_metric:
                    logger.warning(f"Skipping {file_path.name}: Missing splits.")
                    continue

                # Decode polyline (lat, lon) -> (lon, lat) for Shapely
                decoded_coords_latlon = polyline.decode(encoded_polyline)
                if len(decoded_coords_latlon) < 2:
                    logger.warning(
                        f"Skipping {activity_id}: Decoded polyline has < 2 points."
                    )
                    continue
                coords_lonlat = [(lon, lat) for lat, lon in decoded_coords_latlon]

                # Create GeoDataFrame of points to calculate distances accurately
                points_gdf = gpd.GeoDataFrame(
                    [{"geometry": Point(p)} for p in coords_lonlat],
                    crs="EPSG:4326",  # Polylines are always WGS84
                )
                points_gdf_proj = points_gdf.to_crs(target_crs)

                # Calculate cumulative distance along the projected path
                distances = points_gdf_proj.geometry.distance(
                    points_gdf_proj.geometry.shift()
                ).fillna(0)
                points_gdf_proj["cumulative_dist"] = distances.cumsum()
                # Add original coords back for easy LineString creation later
                points_gdf_proj["orig_coords"] = coords_lonlat

                # Process splits
                current_split_start_dist = 0.0
                last_point_idx_used = -1  # Track index to ensure segments connect

                # Iterate through splits, reconstructing full activity polyline
                for split_data in splits_metric:
                    split_distance = split_data.get("distance")
                    avg_speed = split_data.get("average_speed")
                    split_index = split_data.get("split")

                    if (
                        split_distance is None
                        or avg_speed is None
                        or split_index is None
                    ):
                        logger.warning(
                            f"Skipping split {split_index} for activity {activity_id}: Missing data."
                        )
                        continue

                    split_target_end_dist = current_split_start_dist + split_distance

                    # Find points within this split's distance range
                    # We need points from the end of the last split up to the end of this one.
                    # Start index: the point *after* the last one used in the previous segment
                    start_idx = last_point_idx_used + 1

                    # End index: the first point whose cumulative distance *exceeds* the target end distance
                    # We include the point *before* this one to close the segment.
                    end_idx_candidates = np.where(
                        points_gdf_proj["cumulative_dist"] >= split_target_end_dist
                    )[0]

                    if len(end_idx_candidates) > 0:
                        # Take the first point that meets or exceeds the distance
                        end_idx = end_idx_candidates[0]
                    else:
                        # If no point exceeds, it means the split ends after the last point (use all remaining)
                        end_idx = len(points_gdf_proj) - 1

                    # Ensure we don't create segments with identical start/end indices from rounding issues
                    # And ensure we have at least two points to make a line
                    if end_idx <= start_idx:
                        # This can happen if a split distance is very small or points are coincident
                        # Try to take at least one segment if possible, or log warning
                        if start_idx < len(points_gdf_proj) - 1:
                            end_idx = start_idx + 1
                        else:
                            logger.warning(
                                f"Activity {activity_id}, Split {split_index}: "
                                f"Cannot form line segment (start_idx={start_idx}, end_idx={end_idx}). "
                                "Skipping."
                            )
                            # We might lose the speed data for this tiny split.
                            # Update start distance for the next split, but don't update last_point_idx_used yet
                            current_split_start_dist = split_target_end_dist
                            continue  # Skip appending this segment

                    # Extract original (lon, lat) coordinates for this segment
                    segment_coords = points_gdf_proj.iloc[start_idx : end_idx + 1][
                        "orig_coords"
                    ].tolist()

                    if len(segment_coords) >= 2:
                        split_line = LineString(segment_coords)
                        all_split_segments.append(
                            {
                                "activity_id": activity_id,
                                "split": split_index,
                                "average_speed": avg_speed,
                                "geometry": split_line,
                                "split_dist_m": split_distance,  # Keep original split distance for reference
                            }
                        )
                        last_point_idx_used = end_idx  # Update the last point used
                    else:
                        logger.warning(
                            f"Activity {activity_id}, Split {split_index}: "
                            f"Not enough points ({len(segment_coords)}) to form line segment between "
                            f"indices {start_idx}-{end_idx}. "
                            f"Dist range: {current_split_start_dist:.1f}-{split_target_end_dist:.1f}. "
                            "Skipping."
                        )

                    # Update start distance for the next split
                    current_split_start_dist = split_target_end_dist

            except FileNotFoundError:
                logger.error(f"File not found during processing: {file_path}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from file: {file_path}")
            except Exception as e:
                logger.error(
                    f"Unexpected error processing file {file_path}: {e}", exc_info=True
                )

        if not all_split_segments:
            logger.error("No valid activity split segments were processed.")
            self.gdf = None
            return

        # Create final GeoDataFrame
        self.gdf = gpd.GeoDataFrame(all_split_segments, crs="EPSG:4326")  # inline crs?
        logger.info(
            f"Created GeoDataFrame with {len(self.gdf)} activity split segments."
            f"Initial CRS: {self.gdf.crs}"
        )

        self.gdf = self._reproject_if_needed(self.gdf)  # Uses target_crs from settings
        logger.info(f"Reprojected final LineString GDF CRS: {self.gdf.crs}")

        # Save intermediate file
        self._save_intermediate_gdf(self.gdf, "prepared_activity_splits_gpkg")
        logger.info("Activity split segments loaded and preprocessed.")

    def _extract_raster_values(self, points_gdf, raster_path):
        """Extracts raster values at point locations."""
        logger.info(
            f"Extracting raster values from {raster_path} for {len(points_gdf)} points..."
        )
        coords = [(p.x, p.y) for p in points_gdf.geometry]
        try:
            with rasterio.open(raster_path) as src:
                sampled_values = [val[0] for val in src.sample(coords)]
                points_gdf["predicted_speed"] = sampled_values
                # Handle NoData: WBT might use a large negative number, rasterio might read as NaN or fill_value
                nodata_val = src.nodatavals[0]
                if nodata_val is not None:
                    points_gdf["predicted_speed"] = points_gdf[
                        "predicted_speed"
                    ].replace(nodata_val, np.nan)
                # Also check for potential NaNs introduced by sampling outside raster extent
                num_nan = points_gdf["predicted_speed"].isnull().sum()
                if num_nan > 0:
                    logger.warning(
                        f"Found {num_nan} points with NoData/NaN predicted values (likely outside raster extent or nodata areas)."
                    )
            logger.info("Raster value extraction complete.")
            return points_gdf
        except Exception as e:
            logger.error(f"Error extracting raster values: {e}", exc_info=True)
            points_gdf["predicted_speed"] = np.nan  # Assign NaN if extraction fails
            return points_gdf

    def _calculate_rmse(
        self, gdf, actual_col="avg_speed", predicted_col="predicted_speed"
    ):
        """Calculates RMSE, ignoring NaN predictions."""
        valid_gdf = gdf.dropna(subset=[predicted_col])
        if valid_gdf.empty:
            logger.warning(
                f"Cannot calculate RMSE: No valid predicted values found in column '{predicted_col}'."
            )
            return np.nan
        if len(valid_gdf) < len(gdf):
            logger.info(
                f"Calculating RMSE using {len(valid_gdf)} points with valid predictions (out of {len(gdf)} total)."
            )

        actual = valid_gdf[actual_col]
        predicted = valid_gdf[predicted_col]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        logger.info(f"Calculated RMSE: {rmse:.4f}")
        return rmse

    def _convert_segments_to_points(self, segments_gdf):
        """Converts LineString segments GeoDataFrame to a Point GeoDataFrame."""
        logger.info("Converting activity split segments (LineStrings) to points...")
        # Ensure only relevant columns are passed to avoid issues if gdf has complex types
        cols_to_keep = ["average_speed", "geometry"]
        points_gdf = polyline_to_points(segments_gdf[cols_to_keep])
        if points_gdf.empty:
            logger.warning("No points generated from activity segments.")
            return gpd.GeoDataFrame() # Return empty GDF
        logger.info(f"Generated {len(points_gdf)} total points with 'average_speed'.")
        return points_gdf

    def _sample_points(self, points_gdf):
        """Samples a fraction of points from the GeoDataFrame."""
        sample_fraction = self.settings.processing.heatmap_sample_fraction
        sample_seed = self.settings.processing.heatmap_sample_seed
        logger.info(f"Sampling {sample_fraction*100:.0f}% of points for IDW input...")
        # Use settings from self.settings
        points_gdf_sampled = points_gdf.sample(
            frac=sample_fraction, random_state=sample_seed
        )
        if points_gdf_sampled.empty:
            logger.warning("Sampled points GeoDataFrame is empty.")
            return gpd.GeoDataFrame() # Return empty GDF
        logger.info(
            f"Using {len(points_gdf_sampled)} sampled points for interpolation."
        )
        return points_gdf_sampled

    def _filter_points_by_boundary(self, points_gdf):
        """Filters points to keep only those within the configured boundary."""
        points_gdf_final = points_gdf # Start with the input GDF
        try:
            # Use settings from self.settings
            fgdb_path = self.settings.paths.n50_gdb_path
            boundary_layer_name = self.settings.input_data.n50_land_cover_layer
            logger.info(
                f"Loading Oslo boundary layer '{boundary_layer_name}' from {fgdb_path}"
            )
            oslo_boundary_gdf = gpd.read_file(fgdb_path, layer=boundary_layer_name)
            logger.info(
                f"Loaded boundary layer with {len(oslo_boundary_gdf)} features. CRS: {oslo_boundary_gdf.crs}"
            )

            # Ensure boundary CRS matches points CRS
            target_crs = self.settings.processing.output_crs_epsg
            if oslo_boundary_gdf.crs.to_epsg() != target_crs:
                logger.warning(
                    f"Reprojecting boundary from {oslo_boundary_gdf.crs} to EPSG:{target_crs}"
                )
                oslo_boundary_gdf = oslo_boundary_gdf.to_crs(epsg=target_crs)

            # Dissolve into a single boundary polygon
            # Use unary_union for potentially overlapping polygons if dissolve fails or is slow
            oslo_boundary_single = oslo_boundary_gdf.unary_union
            logger.info("Dissolved boundary layer into a single polygon.")

            # Filter sampled points
            logger.info("Filtering points within the dissolved boundary...")
            points_within_oslo = points_gdf[
                points_gdf.within(oslo_boundary_single)
            ]
            logger.info(
                f"Filtered points within Oslo boundary: {len(points_within_oslo)} points remaining (out of {len(points_gdf)})."
            )
            points_gdf_final = points_within_oslo

        except ImportError:
             logger.warning("`pyogrio` not installed. Falling back to `fiona` for GDB reading. Performance may vary.")
             # Fallback or specific handling if pyogrio isn't available
             # This might involve slightly different reading logic if needed
             # For now, assume gpd.read_file handles it via fiona if pyogrio missing
             # Re-run the read operation within this block if specific fiona args are needed
             # Re-dissolve logic might also need adjustment depending on fiona output
             # For simplicity, we'll log and continue assuming basic read works.
             pass # Continue with the filtering logic below if read succeeded
        except Exception as e:
            logger.error(
                f"Error loading or processing Oslo boundary: {e}. Proceeding without filtering.",
                exc_info=True,
            )
            # points_gdf_final remains the input points_gdf

        if points_gdf_final.empty:
            logger.warning(
                "Final points GeoDataFrame (after potential filtering) is empty."
            )
            # Return empty GDF, let caller handle
            return gpd.GeoDataFrame()

        return points_gdf_final

    def _split_train_test(self, points_gdf):
        """Splits the GeoDataFrame into training and testing sets."""
        logger.info(
            "Splitting filtered points into training and testing sets (80/20)..."
        )
        try:
            # Rename target column for consistency before splitting
            # Use settings from self.settings
            points_gdf_renamed = points_gdf.rename(
                columns={"average_speed": "avg_speed"}
            )
            train_gdf, test_gdf = train_test_split(
                points_gdf_renamed,
                train_size=self.settings.processing.heatmap_train_test_split,
                random_state=42,  # For reproducibility
            )
            logger.info(
                f"Train set size: {len(train_gdf)}, Test set size: {len(test_gdf)}"
            )
            if train_gdf.empty or test_gdf.empty:
                logger.error(
                    "Train or test set is empty after splitting. Cannot proceed."
                )
                return None, None # Indicate failure
        except Exception as e:
            logger.error(f"Error during train/test split: {e}", exc_info=True)
            return None, None # Indicate failure

        return train_gdf, test_gdf

    def _prepare_wbt_input(self, train_gdf):
        """Prepares and saves the training data to a temporary shapefile for WBT."""
        speed_field_shp = "avg_speed" # Use the renamed field

        # Ensure speed field is numeric
        if not train_gdf.empty and not pd.api.types.is_numeric_dtype(
            train_gdf[speed_field_shp]
        ):
            logger.warning(
                f"'{speed_field_shp}' in training data is not numeric. Converting."
            )
            # Use .loc to avoid SettingWithCopyWarning if train_gdf is a slice
            train_gdf.loc[:, speed_field_shp] = pd.to_numeric(
                train_gdf[speed_field_shp], errors="coerce"
            ).fillna(0)

        # Use tempfile for automatic cleanup
        try:
            # Create a temporary directory that will be cleaned up automatically
            temp_dir = tempfile.TemporaryDirectory(prefix="heatmap_wbt_")
            input_shp_path = Path(temp_dir.name) / "activity_speed_points_train.shp"
            logger.info(f"Saving training points to temporary shapefile: {input_shp_path}")

            # Save the training points to the shapefile
            save_vector_data(
                train_gdf[[speed_field_shp, "geometry"]],
                input_shp_path,
                driver="ESRI Shapefile",
            )
            logger.info(f"Saved temporary training points shapefile: {input_shp_path}")
            # Return the path and the temporary directory object (so it stays alive)
            return input_shp_path, temp_dir, speed_field_shp

        except Exception as e:
            logger.error(f"Error preparing WBT input shapefile: {e}", exc_info=True)
            return None, None, None # Indicate failure

    def _verify_and_calculate_rmse(self, output_raster_path, train_gdf, test_gdf, speed_field_shp):
        """Verifies raster, assigns CRS, calculates RMSE, and saves results."""
        if not output_raster_path.is_file():
            logger.error(
                f"WBT IDW interpolation command completed, but output file was not created: {output_raster_path}"
            )
            return None, None # Indicate failure (no RMSE values)

        logger.info(
            f"WBT IDW interpolation completed. Output file found: {output_raster_path}"
        )
        # --- Assign Correct CRS ---
        try:
            target_crs_epsg = self.settings.processing.output_crs_epsg
            with rasterio.open(output_raster_path, "r+") as ds:
                if (
                    ds.crs is None
                    or not ds.crs.is_valid
                    or ds.crs.to_epsg() != target_crs_epsg
                ):
                    logger.warning(
                        f"Output raster CRS is missing or incorrect ({ds.crs}). Assigning EPSG:{target_crs_epsg}."
                    )
                    ds.crs = rasterio.crs.CRS.from_epsg(target_crs_epsg)
                else:
                    logger.info(
                        f"Output raster CRS already correctly set to {ds.crs}."
                    )
            logger.info(
                f"Successfully assigned CRS EPSG:{target_crs_epsg} to {output_raster_path}"
            )
        except Exception as crs_e:
            logger.error(
                f"Error assigning CRS to {output_raster_path}: {crs_e}",
                exc_info=True,
            )
            return None, None # Indicate failure

        # --- Calculate RMSE ---
        logger.info("--- Calculating RMSE ---")
        train_rmse, test_rmse = np.nan, np.nan # Initialize
        try:
            # Extract predicted values for training points
            train_gdf_pred = self._extract_raster_values(
                train_gdf.copy(), output_raster_path
            )
            train_rmse = self._calculate_rmse(
                train_gdf_pred,
                actual_col=speed_field_shp,
                predicted_col="predicted_speed",
            )
            logger.info(f"Train RMSE: {train_rmse:.4f}")

            # Extract predicted values for testing points
            test_gdf_pred = self._extract_raster_values(
                test_gdf.copy(), output_raster_path
            )
            test_rmse = self._calculate_rmse(
                test_gdf_pred,
                actual_col=speed_field_shp,
                predicted_col="predicted_speed",
            )
            logger.info(f"Test RMSE: {test_rmse:.4f}")
            logger.info("--- RMSE Calculation Complete ---")
        except Exception as rmse_e:
             logger.error(f"Error during RMSE calculation: {rmse_e}", exc_info=True)
             # Continue to save results, RMSE might be NaN

        # --- Save RMSE Results ---
        try:
            results_csv_path = (
                self.settings.paths.output_dir.parent
                / "heatmap_rmse_results.csv" # Save one level up from run-specific dir
            )
            file_exists = results_csv_path.is_file()
            # Get WBT params from settings
            idw_params = {
                "cell_size": self.settings.processing.heatmap_idw_cell_size,
                "weight": self.settings.processing.heatmap_idw_weight,
                "radius": self.settings.processing.heatmap_idw_radius,
                "min_points": self.settings.processing.heatmap_idw_min_points,
            }
            results_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_output_dir": self.settings.paths.output_dir.name,
                "train_rmse": (
                    f"{train_rmse:.4f}"
                    if not np.isnan(train_rmse)
                    else "NaN"
                ),
                "test_rmse": (
                    f"{test_rmse:.4f}" if not np.isnan(test_rmse) else "NaN"
                ),
                "cell_size": idw_params["cell_size"],
                "weight": idw_params["weight"],
                "radius": idw_params["radius"],
                "min_points": idw_params["min_points"],
                "train_points": len(train_gdf),
                "test_points": len(test_gdf),
            }
            with open(results_csv_path, "a", newline="") as csvfile:
                fieldnames = results_data.keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()  # Write header only if file is new
                writer.writerow(results_data)
            logger.info(f"RMSE results appended to: {results_csv_path}")
        except Exception as csv_e:
            logger.error(
                f"Error saving RMSE results to CSV: {csv_e}", exc_info=True
            )
            # Don't fail the whole process, just log the error

        return train_rmse, test_rmse # Return calculated RMSE values


    @dask.delayed
    def _build_average_speed_raster(self):
        """
        Builds an average speed raster using IDW interpolation on points derived
        from activity split segments and calculates train/test RMSE.
        """
        logger.info(
            "Building average speed raster using WBT IDW and calculating RMSE..."
        )
        if self.gdf is None:
            logger.warning(
                "Activity split data (LineStrings) not loaded, attempting load."
            )
            self.load_data()  # Try loading if not already done

        if self.gdf is None:
            logger.error(
                "Cannot build average speed raster: LineString data loading failed."
            )
            return None

        if self.wbt is None:
            logger.error("WhiteboxTools instance not available. Cannot perform IDW.")
            return None

        # --- Data Preparation Pipeline ---
        points_gdf = self._convert_segments_to_points(self.gdf)
        if points_gdf.empty: return None

        points_gdf_sampled = self._sample_points(points_gdf)
        if points_gdf_sampled.empty: return None

        points_gdf_filtered = self._filter_points_by_boundary(points_gdf_sampled)
        if points_gdf_filtered.empty: return None

        train_gdf, test_gdf = self._split_train_test(points_gdf_filtered) # Corrected variable name
        if train_gdf is None or test_gdf is None: return None # Check for split failure

        # Prepare WBT input within a temporary directory context
        input_shp_path, temp_dir_obj, speed_field_shp = self._prepare_wbt_input(train_gdf)
        if input_shp_path is None: return None # Check for preparation failure

        # --- Define Output Raster Path ---
        output_path_key = "average_speed_raster"
        output_raster_path = self._get_output_path(output_path_key)
        output_raster_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Run WBT IDW Interpolation ---
        idw_success = False
        try:
            logger.info(
                f"Running WBT IDW interpolation for field '{speed_field_shp}'..."
            )
            self.wbt.idw_interpolation(
                i=str(input_shp_path), # Use path from temp dir
                field=speed_field_shp,
                output=str(output_raster_path),
                cell_size=self.settings.processing.heatmap_idw_cell_size,
                weight=self.settings.processing.heatmap_idw_weight,
                radius=self.settings.processing.heatmap_idw_radius,
                min_points=self.settings.processing.heatmap_idw_min_points,
            )
            idw_success = True # Assume success if no exception

        except Exception as e:
            logger.error(f"Error during WBT IDW interpolation call: {e}", exc_info=True)
            # idw_success remains False

        finally:
            # --- Cleanup Temporary Directory ---
            # This happens automatically when temp_dir_obj goes out of scope
            # or explicitly via temp_dir_obj.cleanup() if needed sooner.
            # We rely on the 'with' context or function exit for cleanup.
            # Check if temp_dir_obj exists before trying to access its name or clean it up
            if 'temp_dir_obj' in locals() and temp_dir_obj:
                logger.info(f"Temporary directory {temp_dir_obj.name} will be cleaned up.")
                # Explicit cleanup can be done here if needed: temp_dir_obj.cleanup()
            else:
                 logger.warning("Temporary directory object not created, skipping cleanup log.")


        # --- Verification & RMSE Calculation (only if IDW ran) ---
        if idw_success:
            train_rmse, test_rmse = self._verify_and_calculate_rmse(
                output_raster_path, train_gdf, test_gdf, speed_field_shp
            )

            # Check if verification/RMSE step indicated failure (e.g., file not found or RMSE calc failed)
            if train_rmse is None and test_rmse is None:
                 logger.error("Verification or RMSE calculation failed after IDW.")
                 return None # Indicate overall failure

            # Store the path using the base class helper logic if successful
            self.output_paths[output_path_key] = output_raster_path
            logger.info(f"Successfully generated raster: {output_raster_path}")
            return str(output_raster_path)
        else:
            logger.error("IDW interpolation step failed. Cannot proceed.")
            return None # Indicate failure due to IDW error

    def build(self):
        """Builds the average speed raster from activity data."""
        # Ensure data is loaded first, even if build is called directly
        if self.gdf is None:  # self.gdf stores the LineString segments from load_data
            self.load_data()

        if self.gdf is None:  # Still None if loading failed
            logger.error("Cannot build Heatmap features: Data loading failed.")
            return []  # Return empty list consistent with segments.py

        # The task is now creating the raster file.
        task = self._build_average_speed_raster()
        results = dask.compute(task)  # Compute the single task
        successful_outputs = [r for r in results if r is not None]
        return successful_outputs


if __name__ == "__main__":
    # These imports are only needed for standalone testing
    import folium
    import rasterio
    import rasterio.warp
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import io
    import base64
    from pathlib import Path
    from dask.distributed import Client, LocalCluster
    from src.config import settings  # Assuming settings is an instance of AppConfig
    from whitebox import WhiteboxTools  # Ensure WBT is imported

    logger.info("--- Running heatmap.py Standalone Test ---")

    if settings:
        # --- Basic Setup ---  TODO refactor setup to another file to be reused later
        settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using Output Directory: {settings.paths.output_dir}")

        # Setup Dask client (optional but good practice if using dask.delayed)
        wbt = WhiteboxTools()  # Initialize WBT
        cluster = LocalCluster(
            n_workers=1, threads_per_worker=1
        )  # Adjust workers as needed
        client = Client(cluster)
        logger.info(f"Dask client started: {client.dashboard_link}")

        # --- Test Heatmap Feature ---
        try:
            logger.info("--- Testing Heatmap Feature ---")
            # Pass the initialized wbt instance
            heatmap_feature = Heatmap(settings, wbt)

            logger.info("1. Testing Heatmap Load Data...")
            heatmap_feature.load_data()
            if heatmap_feature.gdf is not None:
                logger.info(
                    f"Activity splits loaded successfully. Shape: {heatmap_feature.gdf.shape}"
                )
            else:
                logger.error("Heatmap GDF is None after loading.")

            logger.info("2. Testing Heatmap Build (Average Speed Raster)...")
            if heatmap_feature.gdf is not None:
                raster_paths = heatmap_feature.build()
                logger.info("Heatmap build process completed.")
                # --- Display Raster on Folium Map ---  # TODO refactor to own function
                if (
                    raster_paths
                ):  # Check again in case the warning was logged but path exists
                    logger.info("Generated Average Speed Raster:")
                    for path in raster_paths:
                        logger.info(f"  - {path}")
                    try:
                        raster_path_str = raster_paths[0]
                        raster_path = Path(raster_path_str)
                        logger.info(f"Attempting to display raster: {raster_path}")

                        with rasterio.open(raster_path) as src:
                            bounds = src.bounds
                            raster_crs = src.crs
                            logger.info(
                                f"Folium Display: Raster CRS read from file: {raster_crs}"
                            )  # Log the CRS being used for transformation
                            data = src.read(
                                1, masked=True
                            )  # Read first band as masked array

                            # Verify the CRS before transforming
                            if not raster_crs or not raster_crs.is_valid:
                                logger.error(
                                    f"Folium Display: Invalid or missing CRS ({raster_crs}) read from {raster_path}. Cannot transform bounds accurately."
                                )
                                # Skip map generation if CRS is bad
                                rgba_data = None  # Ensure we skip overlay
                            elif (
                                raster_crs.to_epsg()
                                != settings.processing.output_crs_epsg
                            ):
                                logger.warning(
                                    f"Folium Display: CRS read from raster ({raster_crs}) does not match expected EPSG:{settings.processing.output_crs_epsg}. Transformation might be incorrect."
                                )  # Use settings directly
                                # Proceed with caution, but log warning

                            # Transform bounds to WGS84 for Folium
                            logger.info(
                                f"Folium Display: Original bounds ({raster_crs}): {bounds}"
                            )
                            bounds_4326 = (
                                None  # Initialize to handle potential transform error
                            )
                            try:
                                bounds_4326 = rasterio.warp.transform_bounds(
                                    raster_crs, "EPSG:4326", *bounds
                                )  # left, bottom, right, top
                                logger.info(
                                    f"Folium Display: Transformed bounds (EPSG:4326): {bounds_4326}"
                                )
                            except Exception as warp_e:
                                logger.error(
                                    f"Folium Display: Error during bounds transformation: {warp_e}",
                                    exc_info=True,
                                )
                                # rgba_data will be checked later, no need to set it here

                            # Prepare data for image overlay: handle nodata, normalize, colormap
                            # This block executes regardless of the transform outcome initially.
                            cmap = plt.cm.viridis
                            valid_data = (
                                data.compressed()
                            )  # Get only valid (unmasked) data
                            rgba_data_calc = None  # Initialize calculated rgba data
                            if valid_data.size > 0:
                                norm = colors.Normalize(
                                    vmin=np.percentile(valid_data, 5),
                                    vmax=np.percentile(valid_data, 95),
                                )
                                rgba_data_calc = cmap(
                                    norm(data), bytes=True
                                )  # Apply colormap
                                if data.mask.any():
                                    rgba_data_calc[data.mask] = (
                                        0,
                                        0,
                                        0,
                                        0,
                                    )  # Make masked pixels transparent
                            else:
                                logger.warning(
                                    "Raster data contains no valid pixels. Cannot create image overlay."
                                )
                                # rgba_data_calc remains None

                            # Check if BOTH transformation succeeded AND we have valid pixel data
                            if rgba_data_calc is not None and bounds_4326 is not None:
                                rgba_data = rgba_data_calc  # Use the calculated data
                                map_bounds_folium = [
                                    [
                                        bounds_4326[1],
                                        bounds_4326[0],
                                    ],  # SouthWest (lat, lon)
                                    [
                                        bounds_4326[3],
                                        bounds_4326[2],
                                    ],  # NorthEast (lat, lon)
                                ]
                                center_lat = (bounds_4326[1] + bounds_4326[3]) / 2
                                center_lon = (bounds_4326[0] + bounds_4326[2]) / 2

                                # Save RGBA data to a PNG in memory
                                buf = io.BytesIO()
                                plt.imsave(
                                    buf, rgba_data_calc, format="png"
                                )  # Use calculated data
                                buf.seek(0)
                                png_base64 = base64.b64encode(buf.read()).decode(
                                    "utf-8"
                                )
                                img_uri = f"data:image/png;base64,{png_base64}"

                                # Create Folium map
                                m = folium.Map(
                                    location=[center_lat, center_lon],
                                    zoom_start=12,
                                    tiles="CartoDB positron",
                                )

                                # Add raster as ImageOverlay
                                img_overlay = folium.raster_layers.ImageOverlay(
                                    image=img_uri,
                                    bounds=map_bounds_folium,
                                    opacity=0.7,
                                    name="Average Speed Heatmap",
                                )
                                img_overlay.add_to(m)
                                folium.LayerControl().add_to(m)

                                # Save map
                                map_output_path = (
                                    settings.paths.output_dir
                                    / "average_speed_heatmap.html"
                                )
                                m.save(str(map_output_path))
                                logger.info(
                                    f"Folium map with raster overlay saved to: {map_output_path}"
                                )
                            else:
                                logger.warning(
                                    "Skipping Folium map generation due to lack of valid raster data."
                                )

                    except ImportError as imp_err:
                        logger.error(
                            f"Folium display skipped: Missing required library ({imp_err}). Install folium, matplotlib."
                        )
                    except Exception as map_e:
                        logger.error(
                            f"Error generating Folium map: {map_e}", exc_info=True
                        )

            else:
                logger.warning("Skipping Heatmap build test as data loading failed.")

            logger.info("--- Heatmap Feature Test Completed ---")

        except Exception as e:
            logger.error(f"Error during Heatmap test: {e}", exc_info=True)
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
        logger.error("Settings could not be loaded. Cannot run standalone test.")

    logger.info("--- Standalone Test Finished ---")
