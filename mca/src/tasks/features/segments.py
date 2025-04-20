import logging
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import dask
from tqdm import tqdm
import polyline
from shapely.geometry import LineString
from geokrige.methods import OrdinaryKriging
import rasterio  # Added for reading/writing normalized raster
from rasterio.transform import from_origin
import matplotlib.cm as cm  # For color mapping
import matplotlib.colors as mcolors  # For color mapping

from src.tasks.features.feature_base import FeatureBase
from src.strava.explore import get_segment_details
from src.utils import (
    load_vector_data,
    polyline_to_points,
    save_vector_data,
)


# Basic setup for standalone testing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        cols_to_drop_from_initial = [
            col
            for col in cache_cols
            if col != segment_id_field and col in initial_gdf.columns
        ]
        initial_gdf_clean = initial_gdf.drop(
            columns=cols_to_drop_from_initial, errors="ignore"
        )

        if not segment_details_cache_df.empty:
            try:
                initial_gdf_clean[segment_id_field] = initial_gdf_clean[
                    segment_id_field
                ].astype(segment_details_cache_df[segment_id_field].dtype)
            except Exception as e:
                logger.warning(
                    f"Could not align ID types for merge: {e}. Merge might fail or produce unexpected results."
                )
            details_to_merge = segment_details_cache_df[cache_cols]
            merged_gdf = pd.merge(
                initial_gdf_clean, details_to_merge, on=segment_id_field, how="left"
            )
        else:
            logger.warning(
                "Segment details cache is empty. Proceeding without detailed attributes."
            )
            merged_gdf = initial_gdf_clean
            for col in cache_cols:
                if col != segment_id_field and col not in merged_gdf.columns:
                    merged_gdf[col] = np.nan

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
        polyline_field = self.settings.input_data.segment_polyline_field
        has_geometry_col = "geometry" in merged_gdf.columns

        for index, row in merged_gdf.iterrows():
            encoded_polyline = None
            if (
                has_geometry_col
                and row["geometry"] is not None
                and hasattr(row["geometry"], "geom_type")
            ):
                geometries.append(row["geometry"])
                continue
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
                if not (
                    has_geometry_col
                    and row["geometry"] is not None
                    and hasattr(row["geometry"], "geom_type")
                ):
                    logger.warning(
                        f"Segment {row.get(segment_id_field, index)}: Missing polyline data and no existing geometry."
                    )
                geometries.append(None)

        cols_to_drop_final = ["map", polyline_field]
        if has_geometry_col:
            cols_to_drop_final.append("geometry")
        self.gdf = gpd.GeoDataFrame(
            merged_gdf.drop(columns=cols_to_drop_final, errors="ignore"),
            geometry=geometries,
        )
        if not has_geometry_col or not initial_gdf.crs:
            self.gdf.crs = "EPSG:4326"
        else:
            self.gdf.crs = initial_gdf.crs

        original_len = len(self.gdf)
        required_detail_cols = [created_at_field]
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
            self.gdf = None
            return

        self.gdf = self._reproject_if_needed(self.gdf)

        # --- Preprocessing Steps (Age and Metrics) ---
        logger.info("Calculating segment age and popularity metrics...")
        athlete_count_field = self.settings.input_data.segment_athlete_count_field
        effort_count_field = self.settings.input_data.segment_effort_count_field
        star_count_field = self.settings.input_data.segment_star_count_field
        required_cols_final = [
            created_at_field,
            athlete_count_field,
            effort_count_field,
            star_count_field,
        ]
        missing_cols_final = [
            col for col in required_cols_final if col not in self.gdf.columns
        ]
        if missing_cols_final:
            logger.error(
                f"Missing required columns for metric calculation: {missing_cols_final}. Cannot proceed."
            )
            self.gdf = None
            return

        age_col = self._calculate_age(created_at_field)
        if age_col:
            self._calculate_popularity_metrics(
                age_col, athlete_count_field, effort_count_field, star_count_field
            )
        else:
            logger.warning(
                "Skipping popularity metric calculation due to age calculation failure."
            )
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
            i = 0  # Dev counter
            for segment_id in tqdm(ids_to_fetch, desc="Fetching Segment Details"):
                segment_input = {"id": segment_id}
                details = get_segment_details(segment_input)
                if details:
                    if isinstance(details, int) and details == 429:
                        logger.warning(
                            "Rate limit likely reached (status 429). Stopping detail fetching."
                        )
                        break
                    if isinstance(details, dict) and details.get("status_code") == 429:
                        logger.warning(
                            "Rate limit likely reached (status 429 in dict). Stopping detail fetching."
                        )
                        break
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
                    needs_saving = True
                time.sleep(self.settings.processing.strava_api_request_delay)

                # --- DEV ONLY ---
                # TODO Remove this block
                i += 1
                if i > 5:
                    logger.info("DEV ONLY: Skipping rest of detail fetching.")
                    break
                # ----------------

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
            self.gdf[created_at_field] = pd.to_datetime(
                self.gdf[created_at_field], errors="coerce", utc=True
            )
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
            now = pd.Timestamp.now(tz="UTC")
            if self.settings.processing.segment_age_calculation_method == "days":
                self.gdf["age_days"] = (now - self.gdf[created_at_field]).dt.days
                age_col = "age_days"
                self.gdf.loc[self.gdf[age_col] <= 0, age_col] = 1
            else:
                self.gdf["age_years"] = (
                    now - self.gdf[created_at_field]
                ).dt.days / 365.25
                age_col = "age_years"
                self.gdf.loc[self.gdf[age_col] <= 0, age_col] = 1 / 365.25
            logger.info(
                f"Calculated segment age using method: '{self.settings.processing.segment_age_calculation_method}' (column: {age_col})"
            )
            return age_col
        except Exception as e:
            logger.error(f"Error calculating segment age: {e}", exc_info=True)
            return None

    def _calculate_popularity_metrics(
        self, age_col, athlete_count_field, effort_count_field, star_count_field
    ):
        """Calculates popularity metrics based on configuration."""
        logger.info("Calculating popularity metrics...")
        for metric_config_name in self.settings.processing.segment_popularity_metrics:
            metric_col_name = metric_config_name
            try:
                self.gdf[athlete_count_field] = pd.to_numeric(
                    self.gdf[athlete_count_field], errors="coerce"
                ).fillna(0)
                self.gdf[star_count_field] = pd.to_numeric(
                    self.gdf[star_count_field], errors="coerce"
                ).fillna(0)
                is_age_metric = "per_age" in metric_col_name
                if is_age_metric and age_col is None:
                    logger.warning(
                        f"Skipping age-based metric '{metric_col_name}' because age calculation failed."
                    )
                    self.gdf[metric_col_name] = np.nan
                    continue
                if metric_col_name == "athletes_per_age":
                    self.gdf[metric_col_name] = (
                        self.gdf[athlete_count_field] / self.gdf[age_col]
                    )
                elif metric_col_name == "efforts_per_age":
                    self.gdf[metric_col_name] = (
                        self.gdf[effort_count_field] / self.gdf[age_col]
                    )
                elif metric_col_name == "stars_per_age":
                    self.gdf[metric_col_name] = (
                        self.gdf[star_count_field] / self.gdf[age_col]
                    )
                elif metric_col_name == "stars_per_athlete":
                    self.gdf[metric_col_name] = np.where(
                        self.gdf[athlete_count_field] > 0,
                        self.gdf[star_count_field] / self.gdf[athlete_count_field],
                        0,
                    )
                else:
                    logger.warning(
                        f"Unsupported popularity metric configured: {metric_col_name}"
                    )
                    continue
                logger.info(f"Calculated popularity metric: {metric_col_name}")
            except Exception as e:
                logger.error(
                    f"Error calculating metric '{metric_col_name}': {e}", exc_info=True
                )
                self.gdf[metric_col_name] = np.nan
        metric_cols_to_fill = [
            m
            for m in self.settings.processing.segment_popularity_metrics
            if m in self.gdf.columns
        ]
        if metric_cols_to_fill:
            self.gdf[metric_cols_to_fill] = self.gdf[metric_cols_to_fill].fillna(0)

    @dask.delayed
    def _build_popularity_raster(self, metric: str):
        """Builds a popularity raster using the configured interpolation method."""
        method = self.settings.processing.interpolation_method_polylines  # Check config
        logger.info(
            f"Building popularity raster for metric: {metric} using method: {method}"
        )

        if method == "kriging":
            return self._build_popularity_raster_kriging(metric)
        elif method == "idw":
            return self._build_popularity_raster_idw(metric)
        elif method == "nn":
            return self._build_popularity_raster_nn(metric)
        elif method == "tin":
            return self._build_popularity_raster_tin(metric)
        else:
            logger.error(
                f"Unsupported interpolation method '{method}' configured for segments (supported: idw, nn, tin, kriging)."
            )
            return None

    def _build_popularity_raster_idw(self, metric: str):
        """Builds a popularity raster using WhiteboxTools IDW."""
        logger.info(f"Interpolating {metric} using WBT IDW...")
        if self.gdf is None:
            raise ValueError("Segments GDF not loaded.")
        if self.wbt is None:
            raise ValueError("WhiteboxTools not initialized.")

        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping IDW.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(
                0
            )

        points_gdf = polyline_to_points(
            self.gdf[["geometry", metric]].dropna(subset=[metric])
        )
        if points_gdf.empty:
            logger.warning(f"No valid points for {metric}. Skipping IDW.")
            return None

        sanitized_metric_field = "".join(filter(str.isalnum, metric))[:10] or "metric"
        points_gdf_shp = points_gdf.rename(columns={metric: sanitized_metric_field})
        input_shp_path = (
            self.settings.paths.output_dir / f"temp_segment_points_{metric}.shp"
        )
        output_raster_path = self._get_output_path("segment_popularity_raster_prefix")
        output_raster_path = (
            output_raster_path.parent / f"{output_raster_path.stem}_{metric}.tif"
        )

        save_vector_data(points_gdf_shp, input_shp_path, driver="ESRI Shapefile")

        try:
            self.wbt.idw_interpolation(
                i=str(input_shp_path),
                field=sanitized_metric_field,
                output=str(output_raster_path),
                min_points=self.settings.processing.segment_popularity_idw_min_points,
                weight=self.settings.processing.segment_popularity_idw_power,
                radius=self.settings.processing.segment_popularity_idw_radius,
                cell_size=self.settings.processing.output_cell_size,
            )
            logger.info(f"Generated IDW popularity raster: {output_raster_path}")
            # Use the base class helper to save/store path
            self._save_raster(
                None, None, "segment_popularity_raster_prefix", metric_name=metric
            )  # Pass None for array/profile as WBT saves directly
            self.output_paths[f"segment_popularity_raster_prefix_{metric}"] = (
                output_raster_path  # Ensure path is stored
            )
            return str(output_raster_path)
        except Exception as e:
            logger.error(
                f"Error during WBT IDW interpolation for {metric}: {e}", exc_info=True
            )
            return None
        finally:
            # Clean up temporary shapefile components
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError as unlink_e:
                        logger.warning(
                            f"Could not delete temp file {temp_file}: {unlink_e}"
                        )

    def _build_popularity_raster_nn(self, metric: str):
        """Builds a popularity raster using WhiteboxTools Nearest Neighbour Gridding."""
        logger.info(f"Interpolating {metric} using WBT Nearest Neighbour...")
        if self.gdf is None:
            raise ValueError("Segments GDF not loaded.")
        if self.wbt is None:
            raise ValueError("WhiteboxTools not initialized.")

        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping NN.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(
                0
            )

        points_gdf = polyline_to_points(
            self.gdf[["geometry", metric]].dropna(subset=[metric])
        )
        if points_gdf.empty:
            logger.warning(f"No valid points for {metric}. Skipping NN.")
            return None

        sanitized_metric_field = "".join(filter(str.isalnum, metric))[:10] or "metric"
        points_gdf_shp = points_gdf.rename(columns={metric: sanitized_metric_field})
        input_shp_path = (
            self.settings.paths.output_dir / f"temp_segment_points_{metric}_nn.shp"
        )
        output_raster_path = self._get_output_path("segment_popularity_raster_prefix")
        output_raster_path = (
            output_raster_path.parent / f"{output_raster_path.stem}_{metric}_nn.tif"
        )  # Add suffix to avoid overwriting

        save_vector_data(points_gdf_shp, input_shp_path, driver="ESRI Shapefile")

        try:
            self.wbt.nearest_neighbour_gridding(
                i=str(input_shp_path),
                field=sanitized_metric_field,
                output=str(output_raster_path),
                cell_size=self.settings.processing.output_cell_size,
                max_dist=self.settings.processing.segment_popularity_nn_max_dist,
                # Add other NN specific parameters from settings if needed
            )
            logger.info(f"Generated NN popularity raster: {output_raster_path}")
            self._save_raster(
                None,
                None,
                "segment_popularity_raster_prefix",
                metric_name=f"{metric}_nn",
            )
            self.output_paths[f"segment_popularity_raster_prefix_{metric}_nn"] = (
                output_raster_path
            )
            return str(output_raster_path)
        except Exception as e:
            logger.error(
                f"Error during WBT NN interpolation for {metric}: {e}", exc_info=True
            )
            return None
        finally:
            # Clean up temporary shapefile components
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError as unlink_e:
                        logger.warning(
                            f"Could not delete temp file {temp_file}: {unlink_e}"
                        )

    def _build_popularity_raster_tin(self, metric: str):
        """Builds a popularity raster using WhiteboxTools TIN Gridding."""
        logger.info(f"Interpolating {metric} using WBT TIN Gridding...")
        if self.gdf is None:
            raise ValueError("Segments GDF not loaded.")
        if self.wbt is None:
            raise ValueError("WhiteboxTools not initialized.")

        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping TIN.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(
                0
            )

        points_gdf = polyline_to_points(
            self.gdf[["geometry", metric]].dropna(subset=[metric])
        )
        if points_gdf.empty:
            logger.warning(f"No valid points for {metric}. Skipping TIN.")
            return None

        sanitized_metric_field = "".join(filter(str.isalnum, metric))[:10] or "metric"
        points_gdf_shp = points_gdf.rename(columns={metric: sanitized_metric_field})
        input_shp_path = (
            self.settings.paths.output_dir / f"temp_segment_points_{metric}_tin.shp"
        )
        output_raster_path = self._get_output_path("segment_popularity_raster_prefix")
        output_raster_path = (
            output_raster_path.parent / f"{output_raster_path.stem}_{metric}_tin.tif"
        )  # Add suffix to avoid overwriting

        save_vector_data(points_gdf_shp, input_shp_path, driver="ESRI Shapefile")

        try:
            self.wbt.tin_gridding(
                i=str(input_shp_path),
                field=sanitized_metric_field,
                output=str(output_raster_path),
                resolution=self.settings.processing.output_cell_size,
                max_triangle_edge_length=self.settings.processing.segment_popularity_tin_max_triangle_edge_length,
                # Add other TIN specific parameters from settings if needed (e.g., interp_parameter)
            )
            logger.info(f"Generated TIN popularity raster: {output_raster_path}")
            self._save_raster(
                None,
                None,
                "segment_popularity_raster_prefix",
                metric_name=f"{metric}_tin",
            )
            self.output_paths[f"segment_popularity_raster_prefix_{metric}_tin"] = (
                output_raster_path
            )
            return str(output_raster_path)
        except Exception as e:
            logger.error(
                f"Error during WBT TIN interpolation for {metric}: {e}", exc_info=True
            )
            return None
        finally:
            # Clean up temporary shapefile components
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError as unlink_e:
                        logger.warning(
                            f"Could not delete temp file {temp_file}: {unlink_e}"
                        )

    def _build_popularity_raster_kriging(self, metric: str):
        """NOTE: Current kriging implementation is broken. Use IDW instead."""
        """Builds a popularity raster for a single metric using geokrige."""
        logger.warning(
            f"Attempting Kriging for metric: {metric}. NOTE: geokrige library might be unstable or have issues."
        )
        if self.gdf is None:
            raise ValueError("Segments GDF not loaded.")

        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping Kriging.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(
                0
            )

        points_gdf = polyline_to_points(
            self.gdf[["geometry", metric]].dropna(subset=[metric])
        )
        if points_gdf.empty:
            logger.warning(f"No valid points for {metric}. Skipping Kriging.")
            return None

        X_known = np.array(
            points_gdf.geometry.apply(lambda geom: [geom.x, geom.y]).tolist()
        )
        y_known = points_gdf[metric].to_numpy()

        cell_size = self.settings.processing.output_cell_size
        minx, miny, maxx, maxy = points_gdf.total_bounds
        grid_x_coords = np.arange(minx, maxx + cell_size, cell_size)
        grid_y_coords = np.arange(miny, maxy + cell_size, cell_size)
        grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x_coords, grid_y_coords)
        X_predict = np.vstack([grid_x_mesh.ravel(), grid_y_mesh.ravel()]).T
        logger.info(
            f"Creating output grid for prediction: {len(grid_x_coords)} x {len(grid_y_coords)} cells"
        )

        interpolated_grid = None
        try:
            logger.info(f"Running Ordinary Kriging for metric '{metric}'...")
            OK = OrdinaryKriging()
            OK.load(X=X_known, y=y_known)
            logger.info("Loaded data into Kriging model.")

            fit_successful = False
            try:
                n_bins = 15  # Fixed number of bins
                logger.info(f"Calculating experimental variogram with {n_bins} bins...")
                OK.variogram(bins=n_bins)
                logger.info("Fitting variogram model...")
                OK.fit(model=self.settings.processing.kriging_model)
                if hasattr(OK, "learned_params") and OK.learned_params is not None:
                    fit_successful = True
                    logger.info(
                        f"Variogram fitting successful for {metric}. Learned params: {OK.learned_params}"
                    )
                else:
                    logger.warning(
                        f"Variogram fitting did not produce learned parameters for {metric}."
                    )
            except ValueError as ve:
                if "Too many bins" in str(ve) or "None values occurrence" in str(ve):
                    logger.warning(
                        f"Variogram calculation failed for {metric} due to binning issue: {ve}. Try adjusting 'bins'."
                    )
                else:
                    logger.warning(
                        f"Could not automatically fit variogram for {metric}: {ve}. Prediction might use default/unfitted model."
                    )
            except Exception as fit_e:
                logger.warning(
                    f"Could not automatically fit variogram for {metric}: {fit_e}. Prediction might use default/unfitted model."
                )

            if fit_successful:
                logger.info("Predicting values on grid...")
                zvalues = OK.predict(X=X_predict)
                logger.info(f"Kriging prediction complete for metric '{metric}'.")
                grid_shape = (len(grid_y_coords), len(grid_x_coords))
                if zvalues.size == grid_shape[0] * grid_shape[1]:
                    interpolated_grid = zvalues.reshape(grid_shape)
                else:
                    raise ValueError(
                        f"Kriging prediction output size {zvalues.size} does not match grid shape {grid_shape}"
                    )
            else:
                logger.error(
                    f"Skipping prediction for {metric} because variogram fitting failed."
                )
                return None
        except Exception as e:
            logger.error(
                f"Error during geokrige execution for {metric}: {e}", exc_info=True
            )
            return None

        if interpolated_grid is not None:
            try:
                transform = from_origin(minx, maxy, cell_size, cell_size)
                profile = {
                    "driver": "GTiff",
                    "height": interpolated_grid.shape[0],
                    "width": interpolated_grid.shape[1],
                    "count": 1,
                    "dtype": str(interpolated_grid.dtype),
                    "crs": self.gdf.crs,
                    "transform": transform,
                    "nodata": -9999,
                }
                self._save_raster(
                    interpolated_grid,
                    profile,
                    "segment_popularity_raster_prefix",
                    metric_name=metric,
                )
                output_path = self.output_paths.get(
                    f"segment_popularity_raster_prefix_{metric}"
                )
                return str(output_path) if output_path else None
            except Exception as e:
                logger.error(
                    f"Error saving Kriging raster for {metric}: {e}", exc_info=True
                )
                return None
        else:
            return None

    @dask.delayed
    def _build_popularity_vector(self, metric: str):
        """Builds a popularity vector layer (buffered segments) for a single metric."""
        logger.info(f"Building popularity vector for metric: {metric}")
        if self.gdf is None:
            raise ValueError("Segments GDF not loaded.")

        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(
                f"Metric '{metric}' not found or all NaN. Skipping vector generation."
            )
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(
                f"Metric '{metric}' not numeric. Converting for vector generation."
            )
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(
                0
            )

        # Select relevant columns and drop rows with NaN metric values
        metric_gdf = self.gdf[
            [self.settings.input_data.segment_id_field, "geometry", metric]
        ].copy()
        metric_gdf = metric_gdf.dropna(subset=[metric])

        if metric_gdf.empty:
            logger.warning(
                f"No valid segments for metric '{metric}' after dropping NaN. Skipping vector generation."
            )
            return None

        # --- Normalize Metric (Min-Max 0-1) ---
        min_val = metric_gdf[metric].min()
        max_val = metric_gdf[metric].max()
        norm_col = f"{metric}_norm"
        if max_val == min_val:
            metric_gdf[norm_col] = 0.0  # Assign 0 if all values are the same
        else:
            metric_gdf[norm_col] = (metric_gdf[metric] - min_val) / (max_val - min_val)
        logger.info(
            f"Normalized metric '{metric}' to '{norm_col}' (Min={min_val}, Max={max_val})"
        )

        # --- Apply Buffer ---
        buffer_dist = self.settings.processing.segment_popularity_buffer_distance
        if buffer_dist <= 0:
            logger.warning(
                f"Buffer distance ({buffer_dist}m) is zero or negative. Skipping buffer."
            )
        else:
            try:
                logger.info(f"Applying buffer of {buffer_dist}m to segments...")
                # Ensure geometry column is active
                metric_gdf = metric_gdf.set_geometry("geometry")
                metric_gdf["geometry"] = metric_gdf.geometry.buffer(
                    buffer_dist, cap_style=2, join_style=2
                )  # Use flat cap, mitre join
                logger.info("Buffer applied successfully.")
            except Exception as e:
                logger.error(
                    f"Error applying buffer for metric {metric}: {e}", exc_info=True
                )
                return None  # Cannot proceed without buffer

        # --- Add Color Column (Simple Grayscale Hex) ---
        # Map normalized value (0-1) to grayscale hex (#000000 to #FFFFFF)
        def grayscale_hex(norm_value):
            if pd.isna(norm_value):
                return "#808080"  # Gray for NaN
            intensity = int(norm_value * 255)
            return f"#{intensity:02x}{intensity:02x}{intensity:02x}"

        metric_gdf["color"] = metric_gdf[norm_col].apply(grayscale_hex)
        logger.info("Added grayscale 'color' column based on normalized metric.")

        # --- Save Output ---
        output_path_key = "segment_popularity_vector_prefix"
        output_vector_path = self._get_output_path(
            output_path_key
        )  # Get base path from config name
        output_vector_path = (
            output_vector_path.parent / f"{output_vector_path.stem}_{metric}.gpkg"
        )

        try:
            save_vector_data(metric_gdf, output_vector_path, driver="GPKG")
            logger.info(f"Saved popularity vector: {output_vector_path}")
            # Store the path using the base class helper logic (even though it's vector)
            self.output_paths[f"{output_path_key}_{metric}"] = output_vector_path
            return str(output_vector_path)
        except Exception as e:
            logger.error(
                f"Error saving popularity vector for {metric}: {e}", exc_info=True
            )
            return None

    def build(self):
        """Builds popularity vectors (buffered segments) for all configured metrics."""
        # TODO: Add a config switch to choose between raster/vector output?
        # For now, defaults to vector output.
        if self.gdf is None:
            self.load_data()
        if self.gdf is None:  # Check again if load_data failed
            logger.error("Cannot build Segments features: Data loading failed.")
            return []

        tasks = []
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

        # --- Choose Output Type (Vector for now) ---
        build_vector = True  # Could be driven by config later
        # build_raster = False

        for metric in available_metrics:
            if build_vector:
                tasks.append(self._build_popularity_vector(metric))
            # elif build_raster:
            #     tasks.append(self._build_popularity_raster(metric)) # Keep raster code available

        if not tasks:
            logger.warning("No valid metrics found to build popularity outputs.")
            return []

        logger.info(f"Computing {len(tasks)} popularity vector tasks...")
        results = dask.compute(*tasks)
        logger.info("Popularity vector computation finished.")
        successful_outputs = [r for r in results if r is not None]
        return successful_outputs


if __name__ == "__main__":
    # These import only needed for standalone testing
    from dask.distributed import Client, LocalCluster
    from whitebox import WhiteboxTools

    from src.config import settings

    logger.info("--- Running segments.py Standalone Test ---")

    # --- Basic Setup ---
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using Output Directory: {settings.paths.output_dir}")

    wbt = WhiteboxTools()
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    logger.info(f"Dask client started: {client.dashboard_link}")

    # --- Test Segments Feature ---
    try:
        logger.info("--- Testing Segments Feature ---")
        segments_feature = Segments(settings, wbt=wbt)

        logger.info("1. Testing Segments Load Data...")
        segments_feature.load_data()
        if segments_feature.gdf is not None:
            logger.info(
                f"Segments loaded successfully. Shape: {segments_feature.gdf.shape}"
            )
            print("Sample preprocessed segments data (first 5 rows):")
            print(segments_feature.gdf.head())
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

        if segments_feature.gdf is not None:
            # settings.processing.interpolation_method_polylines = (
            #     "idw"  # Explicitly set to IDW for testing
            # )
            logger.info(
                f"Testing build with interpolation method: {settings.processing.interpolation_method_polylines}"
            )

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
