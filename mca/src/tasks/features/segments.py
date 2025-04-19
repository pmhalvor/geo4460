import logging
import sys  # Added for main block testing
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import dask
from pathlib import Path
from tqdm import tqdm
import polyline
from shapely.geometry import LineString
from geokrige.methods import OrdinaryKriging
import rasterio
from rasterio.transform import from_origin
from dask.distributed import Client, LocalCluster  # Added for main block testing
from whitebox import WhiteboxTools  # Added for main block testing


# Local imports (adjust relative paths as needed)
try:
    # Assumes execution from the main workflow script in src/
    from .feature_base import FeatureBase
    from src.config import (
        AppConfig,
        settings as app_settings,
    )  # Import settings for main block
    from src.strava.explore import get_segment_details
    from src.utils import (
        load_vector_data,
        polyline_to_points,
        save_vector_data,
        save_raster_data,
    )
except ImportError:
    # Fallback for potential execution from different relative paths
    # This might happen if running tests or individual scripts directly
    try:
        from feature_base import FeatureBase

        # Need to import AppConfig and settings differently for direct execution
        # Assuming config.py is two levels up from features directory
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from config import AppConfig, settings as app_settings
        from strava.explore import get_segment_details
        from utils import (
            load_vector_data,
            polyline_to_points,
            save_vector_data,
            save_raster_data,
        )
    except ImportError as e:
        # If both fail, raise a more informative error
        raise ImportError(
            f"Could not resolve imports for segments.py. Ensure PYTHONPATH is set correctly or run from the project root. Original error: {e}"
        )


logger = logging.getLogger(__name__)


class Segments(FeatureBase):
    """Handles Strava segment data processing, including API fetching and caching."""

    def load_data(self):
        """
        Loads initial segment data, fetches details from Strava API (using CSV cache),
        decodes polylines, calculates age/metrics, and saves the preprocessed data.
        """
        logger.info("Loading and preprocessing Strava segments...")
        initial_gdf = load_vector_data(self.settings.paths.strava_segments_geojson)
        segment_id_field = self.settings.input_data.segment_id_field

        if segment_id_field not in initial_gdf.columns:
            raise ValueError(
                f"Segment ID field '{segment_id_field}' not found in {self.settings.paths.strava_segments_geojson}"
            )

        # --- API Fetching and Caching (CSV) ---
        segment_details_cache_df, cache_cols = self._fetch_and_cache_segment_details(
            initial_gdf
        )

        # --- Merge fetched/cached details with initial GeoDataFrame ---
        # Drop existing detail columns from initial_gdf if they exist, to avoid conflicts
        # Keep geometry from initial GDF
        cols_to_drop_from_initial = [
            col
            for col in cache_cols
            if col != segment_id_field and col in initial_gdf.columns
        ]
        initial_gdf_clean = initial_gdf.drop(
            columns=cols_to_drop_from_initial, errors="ignore"
        )

        # Perform the merge
        if not segment_details_cache_df.empty:
            # Ensure ID types match before merge
            try:
                initial_gdf_clean[segment_id_field] = initial_gdf_clean[
                    segment_id_field
                ].astype(segment_details_cache_df[segment_id_field].dtype)
            except Exception as e:
                logger.warning(
                    f"Could not align ID types for merge: {e}. Merge might fail or produce unexpected results."
                )

            # Select only cache columns for merging to avoid duplicating others
            details_to_merge = segment_details_cache_df[cache_cols]

            merged_gdf = pd.merge(
                initial_gdf_clean,
                details_to_merge,
                on=segment_id_field,
                how="left",  # Keep all segments from initial file, add details where available
            )
        else:
            logger.warning(
                "Segment details cache is empty. Proceeding without detailed attributes."
            )
            merged_gdf = initial_gdf_clean  # Use initial data if cache is empty
            # Add empty columns for required fields if they don't exist
            for col in cache_cols:
                if col != segment_id_field and col not in merged_gdf.columns:
                    merged_gdf[col] = np.nan

        # Check for segments that didn't get details (check for NaN in a required field like created_at)
        created_at_field = self.settings.input_data.segment_created_at_field
        if created_at_field in merged_gdf.columns:
            missing_details_count = merged_gdf[created_at_field].isna().sum()
            if missing_details_count > 0:
                logger.warning(
                    f"{missing_details_count} segments are missing details ({created_at_field} is null) after cache/API fetch."
                )
        else:
            logger.warning(
                f"'{created_at_field}' column not found after merge, cannot check for missing details."
            )

        # --- Decode Polylines and Create Geometry ---
        logger.info("Decoding polylines (if necessary)...")
        geometries = []
        polyline_field = (
            self.settings.input_data.segment_polyline_field
        )  # Field containing encoded polyline
        has_geometry_col = "geometry" in merged_gdf.columns

        for index, row in merged_gdf.iterrows():
            encoded_polyline = None
            # Check if geometry already exists and is valid
            if (
                has_geometry_col
                and row["geometry"] is not None
                and hasattr(row["geometry"], "geom_type")
            ):
                geometries.append(row["geometry"])
                continue  # Skip decoding if valid geometry exists

            # If no valid geometry, try decoding from 'map' or top-level field
            if (
                "map" in row
                and isinstance(row["map"], dict)
                and polyline_field in row["map"]
            ):
                encoded_polyline = row["map"][polyline_field]
            elif polyline_field in row and pd.notna(row[polyline_field]):
                encoded_polyline = row[polyline_field]

            if encoded_polyline:
                try:
                    decoded_coords = polyline.decode(encoded_polyline)
                    lon_lat_coords = [(lon, lat) for lat, lon in decoded_coords]
                    if len(lon_lat_coords) >= 2:
                        geometries.append(LineString(lon_lat_coords))
                    else:
                        geometries.append(None)
                        logger.warning(
                            f"Segment {row.get(segment_id_field, index)}: Decoded polyline has < 2 points."
                        )
                except Exception as e:
                    logger.warning(
                        f"Segment {row.get(segment_id_field, index)}: Failed to decode polyline '{encoded_polyline}': {e}"
                    )
                    geometries.append(None)
            else:
                # Only log warning if geometry wasn't already present
                if not (
                    has_geometry_col
                    and row["geometry"] is not None
                    and hasattr(row["geometry"], "geom_type")
                ):
                    logger.warning(
                        f"Segment {row.get(segment_id_field, index)}: Missing polyline data and no existing geometry."
                    )
                geometries.append(None)

        # Create the final GeoDataFrame
        cols_to_drop_final = ["map", polyline_field]
        if has_geometry_col:
            cols_to_drop_final.append("geometry")

        self.gdf = gpd.GeoDataFrame(
            merged_gdf.drop(columns=cols_to_drop_final, errors="ignore"),
            geometry=geometries,
        )
        # Set CRS - Assume WGS84 if geometry was decoded, otherwise inherit from initial_gdf if possible
        if not has_geometry_col or not initial_gdf.crs:
            self.gdf.crs = "EPSG:4326"
        else:
            self.gdf.crs = (
                initial_gdf.crs
            )  # Keep original CRS if geometry wasn't decoded

        # Drop rows with invalid/missing geometries or missing crucial details (like created_at)
        original_len = len(self.gdf)
        required_detail_cols = [created_at_field]  # Add other essential cols if needed
        cols_to_check = ["geometry"] + [
            col for col in required_detail_cols if col in self.gdf.columns
        ]
        self.gdf = self.gdf.dropna(subset=cols_to_check)

        self.gdf = self.gdf[self.gdf.geometry.is_valid & ~self.gdf.geometry.is_empty]
        if len(self.gdf) < original_len:
            logger.warning(
                f"Dropped {original_len - len(self.gdf)} segments due to invalid geometry or missing essential details."
            )

        if self.gdf.empty:
            logger.error(
                "No valid segments remaining after loading and detail fetching."
            )
            self.gdf = None  # Ensure gdf is None if empty
            return  # Stop processing if no valid segments

        self.gdf = self._reproject_if_needed(self.gdf)  # Reproject to target CRS

        # --- Preprocessing Steps (Age and Metrics) ---
        logger.info("Calculating segment age and popularity metrics...")
        athlete_count_field = self.settings.input_data.segment_athlete_count_field
        star_count_field = self.settings.input_data.segment_star_count_field

        # Ensure necessary columns exist after merge/cleanup
        required_cols_final = [created_at_field, athlete_count_field, star_count_field]
        missing_cols_final = [
            col for col in required_cols_final if col not in self.gdf.columns
        ]
        if missing_cols_final:
            # This might happen if cache was empty and API failed for all segments
            logger.error(
                f"Missing required columns for metric calculation: {missing_cols_final}. Cannot proceed."
            )
            self.gdf = None  # Mark as invalid
            return

        # Calculate Age
        age_col = self._calculate_age(created_at_field)

        # Calculate Popularity Metrics (using _per_age convention)
        if age_col:  # Proceed only if age calculation was successful
            self._calculate_popularity_metrics(
                age_col, athlete_count_field, star_count_field
            )
        else:
            logger.warning(
                "Skipping popularity metric calculation due to age calculation failure."
            )
            # Add NaN columns for metrics if they don't exist
            for (
                metric_config_name
            ) in self.settings.processing.segment_popularity_metrics:
                if metric_config_name not in self.gdf.columns:
                    self.gdf[metric_config_name] = np.nan

        logger.info("Strava segments loaded and preprocessed with API details.")
        self._save_intermediate_gdf(self.gdf, "prepared_segments_gpkg")

    def _fetch_and_cache_segment_details(self, initial_gdf):
        """Fetches and caches segment details from Strava API using CSV."""
        cache_path = self.settings.paths.segment_details_cache_csv
        segment_id_field = self.settings.input_data.segment_id_field
        athlete_count_field = self.settings.input_data.segment_athlete_count_field
        star_count_field = self.settings.input_data.segment_star_count_field
        created_at_field = self.settings.input_data.segment_created_at_field
        effort_count_field = self.settings.input_data.segment_effort_count_field
        distance_field = self.settings.input_data.segment_distance_field
        elevation_diff_field = self.settings.input_data.segment_elevation_diff_field

        cache_cols = [
            segment_id_field,
            athlete_count_field,
            star_count_field,
            created_at_field,
            effort_count_field,
            distance_field,
            elevation_diff_field,
        ]
        segment_details_cache_df = pd.DataFrame(columns=cache_cols)

        if cache_path.exists():
            try:
                segment_details_cache_df = pd.read_csv(cache_path)
                if not segment_details_cache_df.empty:
                    segment_details_cache_df[segment_id_field] = (
                        segment_details_cache_df[segment_id_field].astype(
                            initial_gdf[segment_id_field].dtype
                        )
                    )
                logger.info(
                    f"Loaded {len(segment_details_cache_df)} segment details from CSV cache: {cache_path}"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load or parse CSV cache file {cache_path}: {e}. Starting with empty cache."
                )
                segment_details_cache_df = pd.DataFrame(columns=cache_cols)

        cached_ids = set()
        if not segment_details_cache_df.empty:
            valid_cache_df = segment_details_cache_df.dropna(subset=[created_at_field])
            cached_ids = set(valid_cache_df[segment_id_field].unique())

        ids_to_fetch = (
            initial_gdf[~initial_gdf[segment_id_field].isin(cached_ids)][
                segment_id_field
            ]
            .unique()
            .tolist()
        )

        newly_fetched_details = []
        needs_saving = False
        if ids_to_fetch:
            logger.info(
                f"Fetching details for {len(ids_to_fetch)} segments not found or incomplete in cache..."
            )
            for segment_id in tqdm(ids_to_fetch, desc="Fetching Segment Details"):
                segment_input = {"id": segment_id}
                details = get_segment_details(segment_input)
                if details:
                    detail_subset = {
                        col: details.get(col) for col in cache_cols if col in details
                    }
                    detail_subset[segment_id_field] = segment_id
                    newly_fetched_details.append(detail_subset)
                    needs_saving = True
                else:
                    logger.warning(
                        f"Failed to fetch details for segment ID: {segment_id}"
                    )
                    placeholder = {col: np.nan for col in cache_cols}
                    placeholder[segment_id_field] = segment_id
                    newly_fetched_details.append(placeholder)
                    needs_saving = True  # Save failure marker

                time.sleep(self.settings.processing.strava_api_request_delay)

            if newly_fetched_details:
                new_details_df = pd.DataFrame(newly_fetched_details)
                if not segment_details_cache_df.empty:
                    new_details_df[segment_id_field] = new_details_df[
                        segment_id_field
                    ].astype(segment_details_cache_df[segment_id_field].dtype)
                segment_details_cache_df = pd.concat(
                    [segment_details_cache_df, new_details_df], ignore_index=True
                )
                segment_details_cache_df = segment_details_cache_df.drop_duplicates(
                    subset=[segment_id_field], keep="last"
                )

        if needs_saving:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                segment_details_cache_df.to_csv(cache_path, index=False)
                logger.info(f"Updated segment details cache CSV: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save updated cache CSV: {e}")
        else:
            logger.info("All segment details found in cache.")

        return segment_details_cache_df, cache_cols

    def _calculate_age(self, created_at_field):
        """Calculates segment age based on configuration."""
        age_col = None
        try:
            # Convert 'created_at' to datetime (should be done by API/cache ideally, but ensure)
            self.gdf[created_at_field] = pd.to_datetime(
                self.gdf[created_at_field], errors="coerce", utc=True
            )
            # Drop rows where conversion failed (e.g., if cache had bad data)
            original_len_date = len(self.gdf)
            self.gdf = self.gdf.dropna(subset=[created_at_field])
            if len(self.gdf) < original_len_date:
                logger.warning(
                    f"Dropped {original_len_date - len(self.gdf)} rows due to invalid date format in '{created_at_field}' after merge."
                )

            if self.gdf.empty:
                logger.error("No valid segments remaining after date validation.")
                self.gdf = None
                return None

            now = pd.Timestamp.now(tz="UTC")  # Use timezone-aware timestamp

            if self.settings.processing.segment_age_calculation_method == "days":
                self.gdf["age_days"] = (now - self.gdf[created_at_field]).dt.days
                age_col = "age_days"
                self.gdf.loc[self.gdf[age_col] <= 0, age_col] = 1  # Min age 1 day
            else:  # Default to years
                self.gdf["age_years"] = (
                    now - self.gdf[created_at_field]
                ).dt.days / 365.25
                age_col = "age_years"
                self.gdf.loc[self.gdf[age_col] <= 0, age_col] = (
                    1 / 365.25
                )  # Min age ~1 day in years
            logger.info(
                f"Calculated segment age using method: '{self.settings.processing.segment_age_calculation_method}' (column: {age_col})"
            )
            return age_col

        except Exception as e:
            logger.error(f"Error calculating segment age: {e}", exc_info=True)
            return None

    def _calculate_popularity_metrics(
        self, age_col, athlete_count_field, star_count_field
    ):
        """Calculates popularity metrics based on configuration."""
        logger.info("Calculating popularity metrics...")
        for metric_config_name in self.settings.processing.segment_popularity_metrics:
            metric_col_name = metric_config_name  # Use the name directly from config
            try:
                # Ensure count fields are numeric
                self.gdf[athlete_count_field] = pd.to_numeric(
                    self.gdf[athlete_count_field], errors="coerce"
                ).fillna(0)
                self.gdf[star_count_field] = pd.to_numeric(
                    self.gdf[star_count_field], errors="coerce"
                ).fillna(0)

                # Check if age calculation succeeded before calculating age-based metrics
                is_age_metric = "per_age" in metric_col_name
                if is_age_metric and age_col is None:
                    logger.warning(
                        f"Skipping age-based metric '{metric_col_name}' because age calculation failed."
                    )
                    self.gdf[metric_col_name] = np.nan  # Assign NaN if age is missing
                    continue

                if metric_col_name == "athletes_per_age":
                    self.gdf[metric_col_name] = (
                        self.gdf[athlete_count_field] / self.gdf[age_col]
                    )
                elif metric_col_name == "stars_per_age":
                    self.gdf[metric_col_name] = (
                        self.gdf[star_count_field] / self.gdf[age_col]
                    )
                elif metric_col_name == "stars_per_athlete":
                    # Avoid division by zero
                    self.gdf[metric_col_name] = np.where(
                        self.gdf[athlete_count_field] > 0,
                        self.gdf[star_count_field] / self.gdf[athlete_count_field],
                        0,
                    )
                else:
                    logger.warning(
                        f"Unsupported popularity metric configured: {metric_col_name}"
                    )
                    continue  # Skip calculation for unsupported metrics

                logger.info(f"Calculated popularity metric: {metric_col_name}")
            except Exception as e:
                logger.error(
                    f"Error calculating metric '{metric_col_name}': {e}", exc_info=True
                )
                self.gdf[metric_col_name] = np.nan  # Assign NaN on error

        # Fill any potential NaNs created during calculations
        metric_cols_to_fill = [
            m
            for m in self.settings.processing.segment_popularity_metrics
            if m in self.gdf.columns
        ]
        if metric_cols_to_fill:
            self.gdf[metric_cols_to_fill] = self.gdf[metric_cols_to_fill].fillna(
                0
            )  # Fill NaNs with 0

    @dask.delayed
    def _build_popularity_raster(self, metric: str):
        """Builds a popularity raster for a single metric using geokrige."""
        logger.info(f"Building popularity raster for metric: {metric} using geokrige")
        if self.gdf is None:
            # Attempt to load data if not already loaded (e.g., if build called directly)
            logger.warning("Segments GDF not loaded, attempting to load now...")
            self.load_data()
            if self.gdf is None:  # Still None after trying to load
                raise ValueError("Segments data could not be loaded.")

        # 1. Ensure metric column exists and is valid
        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(
                f"Metric column '{metric}' not found or contains only NaN. Skipping raster generation."
            )
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(
                f"Metric column '{metric}' is not numeric. Attempting conversion."
            )
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(
                0
            )  # Fill NaNs introduced by coercion

        # 2. Convert polylines to points and prepare data for Kriging
        cols_to_keep = ["geometry", metric]
        points_gdf = polyline_to_points(self.gdf[cols_to_keep].dropna(subset=[metric]))

        if points_gdf.empty:
            logger.warning(
                f"No valid points generated for metric '{metric}'. Skipping interpolation."
            )
            return None

        # Extract coordinates and values
        x_coords = points_gdf.geometry.x.to_numpy()
        y_coords = points_gdf.geometry.y.to_numpy()
        values = points_gdf[metric].to_numpy()

        # 3. Define output grid
        cell_size = self.settings.processing.output_cell_size
        minx, miny, maxx, maxy = points_gdf.total_bounds
        grid_x = np.arange(minx, maxx + cell_size, cell_size)
        grid_y = np.arange(miny, maxy + cell_size, cell_size)
        logger.info(
            f"Creating output grid: X({len(grid_x)} steps), Y({len(grid_y)} steps)"
        )

        # 4. Perform Ordinary Kriging
        try:
            logger.info(f"Running Ordinary Kriging for metric '{metric}'...")
            OK = OrdinaryKriging(
                xi=x_coords,
                yi=y_coords,
                zi=values,
                xk=grid_x,
                yk=grid_y,
                model=self.settings.processing.kriging_model,
            )
            OK.execute()
            zvalues = OK.Z  # Get interpolated values
            grid_shape = (len(grid_y), len(grid_x))
            # Ensure zvalues is reshaped correctly, handling potential flattening issues
            # Geokrige Z is typically flattened in C-order (row-major) matching np.meshgrid indexing='xy' default
            # However, rasterio expects (rows, cols) which corresponds to (y, x)
            # We need to reshape considering the grid definition (np.arange)
            # The grid_y corresponds to rows, grid_x to columns.
            # Reshape should be (len(grid_y), len(grid_x))
            if zvalues.size == grid_shape[0] * grid_shape[1]:
                interpolated_grid = zvalues.reshape(grid_shape)
            else:
                raise ValueError(
                    f"Kriging output size {zvalues.size} does not match grid shape {grid_shape}"
                )

            logger.info(f"Kriging execution complete for metric '{metric}'.")

        except Exception as e:
            logger.error(
                f"Error during geokrige execution for {metric}: {e}", exc_info=True
            )
            return None

        # 5. Save raster using rasterio profile
        try:
            # Create rasterio profile
            transform = from_origin(
                minx, maxy, cell_size, cell_size
            )  # Origin is top-left
            profile = {
                "driver": "GTiff",
                "height": interpolated_grid.shape[0],
                "width": interpolated_grid.shape[1],
                "count": 1,
                "dtype": str(
                    interpolated_grid.dtype
                ),  # Ensure dtype is string for profile
                "crs": self.gdf.crs,  # Use the CRS of the reprojected GDF
                "transform": transform,
                "nodata": -9999,  # Or choose an appropriate nodata value
            }

            # Use the _save_raster helper, passing the metric name
            self._save_raster(
                interpolated_grid,
                profile,
                "segment_popularity_raster_prefix",
                metric_name=metric,
            )
            # The actual path is stored in self.output_paths by _save_raster
            output_path = self.output_paths.get(
                f"segment_popularity_raster_prefix_{metric}"
            )
            return str(output_path) if output_path else None

        except Exception as e:
            logger.error(
                f"Error saving Kriging raster for {metric}: {e}", exc_info=True
            )
            return None

    def build(self):
        """Builds popularity rasters for all configured metrics."""
        if self.gdf is None:
            self.load_data()
        if self.gdf is None:  # Check again if load_data failed
            logger.error("Cannot build Segments features: Data loading failed.")
            return []

        tasks = []
        # Ensure metrics exist in the dataframe after loading/preprocessing
        available_metrics = [
            m
            for m in self.settings.processing.segment_popularity_metrics
            if m in self.gdf.columns
            and pd.api.types.is_numeric_dtype(self.gdf[m])
            and not self.gdf[m].isna().all()
        ]
        skipped_metrics = [
            m
            for m in self.settings.processing.segment_popularity_metrics
            if m not in available_metrics
        ]
        if skipped_metrics:
            logger.warning(
                f"Metrics configured but not found, not numeric, or all NaN in GDF: {skipped_metrics}. Skipping raster generation for these."
            )

        for metric in available_metrics:
            tasks.append(self._build_popularity_raster(metric))

        if not tasks:
            logger.warning("No valid metrics found to build popularity rasters.")
            return []

        # Compute all raster tasks in parallel
        logger.info(f"Computing {len(tasks)} popularity raster tasks...")
        results = dask.compute(*tasks)
        logger.info("Popularity raster computation finished.")
        # Filter out None results (errors during geokrige)
        successful_rasters = [r for r in results if r is not None]
        return successful_rasters  # Return list of paths to generated rasters


# --- Example Usage ---
if __name__ == "__main__":
    # Basic setup for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("--- Running segments.py Standalone Test ---")

    # Use settings loaded from config.py
    settings = app_settings
    # Ensure output directory exists for the test
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using Output Directory: {settings.paths.output_dir}")

    wbt = None  # WhiteboxTools not directly used in Segments class anymore, but keep for consistency if needed later
    # try:
    #     logger.info("Initializing WhiteboxTools for test...")
    #     wbt = WhiteboxTools()
    #     wbt.set_verbose_mode(settings.processing.wbt_verbose)
    #     logger.info("WhiteboxTools initialized.")
    # except Exception as e:
    #     logger.error(f"Failed to initialize WhiteboxTools: {e}.")
    #     # sys.exit(1) # Don't exit if WBT fails, as Segments doesn't use it directly now

    # Initialize Dask client for parallel processing
    try:
        cluster = LocalCluster()
        client = Client(cluster)
        logger.info(f"Dask client started: {client.dashboard_link}")
    except Exception as e:
        logger.error(f"Failed to start Dask client: {e}")
        client = None

    # --- Test Segments Feature ---
    try:
        logger.info("--- Testing Segments Feature ---")
        # Pass None for wbt if not initialized or needed
        segments_feature = Segments(settings, wbt=None)

        logger.info("1. Testing Segments Load Data...")
        segments_feature.load_data()
        if segments_feature.gdf is not None:
            logger.info(
                f"Segments loaded successfully. Shape: {segments_feature.gdf.shape}"
            )
            print("Sample preprocessed segments data (first 5 rows):")
            print(segments_feature.gdf.head())
            # Check if metric columns were added
            expected_metrics = settings.processing.segment_popularity_metrics
            actual_metrics = [
                m for m in expected_metrics if m in segments_feature.gdf.columns
            ]
            logger.info(f"Expected metrics: {expected_metrics}")
            logger.info(f"Actual metrics found in GDF: {actual_metrics}")
            print("\nMetrics sample:")
            print(segments_feature.gdf[actual_metrics].head())
        else:
            logger.error("Segments GDF is None after loading.")

        logger.info("2. Testing Segments Build (Popularity Rasters)...")
        # Ensure data is loaded before building
        if segments_feature.gdf is not None:
            raster_paths = segments_feature.build()
            logger.info(f"Segments build process completed.")
            if raster_paths:
                logger.info("Generated Popularity Rasters:")
                for path in raster_paths:
                    logger.info(f"  - {path}")
            else:
                logger.warning(
                    "No popularity rasters were generated (check logs for errors)."
                )
        else:
            logger.warning("Skipping Segments build test as data loading failed.")

        logger.info("--- Segments Feature Test Completed ---")

    except Exception as e:
        logger.error(f"Error during Segments test: {e}", exc_info=True)

    # Clean up Dask client
    if client:
        try:
            client.close()
            cluster.close()
            logger.info("Dask client and cluster closed.")
        except Exception as e:
            logger.warning(f"Error closing Dask client/cluster: {e}")

    logger.info("--- Standalone Test Finished ---")
