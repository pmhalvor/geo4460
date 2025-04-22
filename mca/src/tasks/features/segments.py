import logging
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import dask
from tqdm import tqdm
import polyline
from shapely.geometry import LineString, MultiLineString # Added MultiLineString
from geokrige.methods import OrdinaryKriging
import rasterio
from rasterio.transform import from_origin
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.tasks.features.feature_base import FeatureBase
from src.strava.explore import get_segment_details
from src.utils import (
    load_vector_data,
    polyline_to_points,
    save_vector_data,
    display_vectors_on_folium_map,
    reproject_gdf, # Explicitly import reproject_gdf
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
        Loads segment data, combines sources ensuring consistent target CRS,
        fetches details, decodes polylines (reprojecting them), validates,
        simplifies, calculates metrics, and saves the preprocessed data.
        """
        logger.info("Loading and preprocessing Strava segments...")
        segment_id_field = self.settings.input_data.segment_id_field
        target_crs = f"EPSG:{self.settings.processing.output_crs_epsg}"
        source_crs_4326 = "EPSG:4326" # Define source CRS for polylines

        # --- Load Original GeoJSON ---
        logger.info(f"Loading original segments from: {self.settings.paths.strava_segments_geojson}")
        geojson_gdf = load_vector_data(self.settings.paths.strava_segments_geojson)
        if geojson_gdf is not None:
            if segment_id_field not in geojson_gdf.columns:
                 logger.warning(f"Segment ID field '{segment_id_field}' not found in {self.settings.paths.strava_segments_geojson}. Skipping.")
                 geojson_gdf = None
            else:
                 # Ensure target CRS
                 geojson_gdf = reproject_gdf(geojson_gdf, target_crs)
                 logger.info(f"Loaded and ensured CRS {target_crs} for {len(geojson_gdf)} original segments.")
        else:
            logger.warning(f"Could not load original segments from {self.settings.paths.strava_segments_geojson}")

        # --- Load Newly Collected GeoPackage ---
        logger.info(f"Loading newly collected segments from: {self.settings.paths.collected_segments_gpkg}")
        gpkg_gdf = load_vector_data(self.settings.paths.collected_segments_gpkg)
        if gpkg_gdf is not None:
            if segment_id_field not in gpkg_gdf.columns:
                 logger.warning(f"Segment ID field '{segment_id_field}' not found in {self.settings.paths.collected_segments_gpkg}. Skipping.")
                 gpkg_gdf = None
            else:
                 # Ensure target CRS
                 gpkg_gdf = reproject_gdf(gpkg_gdf, target_crs)
                 logger.info(f"Loaded and ensured CRS {target_crs} for {len(gpkg_gdf)} newly collected segments.")
        else:
            logger.warning(f"Could not load collected segments from {self.settings.paths.collected_segments_gpkg}")

        # --- Combine Data (Both should now be in target_crs) ---
        if geojson_gdf is not None and gpkg_gdf is not None:
            logger.info("Combining original and collected segments...")
            # Ensure CRS match before concat (should already be the case)
            if not geojson_gdf.crs.equals(gpkg_gdf.crs):
                 logger.error(f"CRS mismatch before combining: {geojson_gdf.crs} vs {gpkg_gdf.crs}. Aborting.")
                 self.gdf = None
                 return
            combined_gdf = pd.concat([geojson_gdf, gpkg_gdf], ignore_index=True, sort=False)
            combined_gdf = combined_gdf.drop_duplicates(subset=[segment_id_field], keep='last')
            logger.info(f"Combined data has {len(combined_gdf)} unique segments (CRS: {combined_gdf.crs}).")
        elif geojson_gdf is not None:
            logger.info("Using only original segments.")
            combined_gdf = geojson_gdf
        elif gpkg_gdf is not None:
            logger.info("Using only newly collected segments.")
            combined_gdf = gpkg_gdf
        else:
            logger.error("No segment data could be loaded. Cannot proceed.")
            self.gdf = None
            return

        # --- API Fetching and Caching ---
        segment_details_cache_df, cache_cols = self._fetch_and_cache_segment_details(combined_gdf)

        # --- Merge fetched/cached details ---
        cols_to_drop_before_merge = [
            col for col in cache_cols if col != segment_id_field and col in combined_gdf.columns
        ]
        combined_gdf_clean = combined_gdf.drop(columns=cols_to_drop_before_merge, errors="ignore")

        if not segment_details_cache_df.empty:
             try:
                 combined_gdf_clean[segment_id_field] = combined_gdf_clean[segment_id_field].astype(segment_details_cache_df[segment_id_field].dtype)
             except Exception as e:
                 logger.warning(f"Could not align ID types for merge: {e}.")

             details_to_merge = segment_details_cache_df[cache_cols]
             merged_gdf = pd.merge(
                 combined_gdf_clean, details_to_merge, on=segment_id_field, how="left"
             )
             logger.info(f"Merged details from cache/API onto {len(merged_gdf)} segments.")
        else:
             logger.warning("Segment details cache is empty. Proceeding without detailed attributes.")
             merged_gdf = combined_gdf_clean
             for col in cache_cols:
                 if col != segment_id_field and col not in merged_gdf.columns:
                     merged_gdf[col] = np.nan

        # Check for missing essential details (like created_at)
        created_at_field = self.settings.input_data.segment_created_at_field
        if created_at_field in merged_gdf.columns:
             missing_details_count = merged_gdf[created_at_field].isna().sum()
             if missing_details_count > 0:
                 logger.warning(f"{missing_details_count} segments missing '{created_at_field}' after cache/API fetch.")
        else:
             logger.warning(f"'{created_at_field}' column not found after merge.")

        # --- Handle Geometry Creation/Update from Polylines ---
        logger.info("Checking for and processing missing geometries from polylines...")
        polyline_field = self.settings.input_data.segment_polyline_field
        # Identify rows needing geometry from polyline (geometry is null or maybe invalid placeholder)
        # We assume existing geometries from loaded files are already in target_crs
        needs_geom_mask = merged_gdf['geometry'].isnull() | ~merged_gdf.geometry.is_valid

        rows_to_decode = merged_gdf[needs_geom_mask]
        decoded_geoms_data = [] # List to hold {'index': index, 'geometry': geom_4326}

        if not rows_to_decode.empty:
            logger.info(f"Found {len(rows_to_decode)} segments potentially needing geometry decoded from polyline.")
            for index, row in rows_to_decode.iterrows():
                encoded_polyline = None
                if 'map' in row and isinstance(row['map'], dict) and polyline_field in row['map']:
                    encoded_polyline = row['map'][polyline_field]
                elif polyline_field in row and pd.notna(row[polyline_field]):
                    encoded_polyline = row[polyline_field]

                if encoded_polyline:
                    try:
                        decoded_coords = polyline.decode(encoded_polyline)
                        lon_lat_coords = [(lon, lat) for lat, lon in decoded_coords]
                        if len(lon_lat_coords) >= 2:
                            geom_4326 = LineString(lon_lat_coords)
                            decoded_geoms_data.append({'index': index, 'geometry': geom_4326})
                        else:
                            logger.warning(f"Segment index {index}: Decoded polyline has < 2 points.")
                    except Exception as e:
                        logger.warning(f"Segment index {index}: Failed to decode polyline '{encoded_polyline}': {e}")
                else:
                     # If geometry was null/invalid AND no polyline, log it
                     logger.debug(f"Segment index {index}: Missing geometry and no polyline found.")

            # Reproject decoded geometries if any were created
            if decoded_geoms_data:
                logger.info(f"Successfully decoded {len(decoded_geoms_data)} polylines. Reprojecting them to {target_crs}...")
                temp_decoded_gdf = gpd.GeoDataFrame(decoded_geoms_data, crs=source_crs_4326)
                temp_decoded_gdf = temp_decoded_gdf.set_index('index') # Use original index

                try:
                    temp_reprojected_gdf = reproject_gdf(temp_decoded_gdf, target_crs)
                    # Update the main GeoDataFrame using the index
                    # Use .loc for safe index-based assignment
                    merged_gdf.loc[temp_reprojected_gdf.index, 'geometry'] = temp_reprojected_gdf['geometry']
                    logger.info("Updated main GDF with reprojected geometries from polylines.")
                except Exception as reproj_e:
                    logger.error(f"Failed to reproject decoded polylines: {reproj_e}. Geometries for these segments remain unset.")
        else:
            logger.info("No segments required geometry decoding from polylines.")

        # --- Final GDF Creation and Cleanup ---
        # Drop helper columns before creating the final GDF
        cols_to_drop_final = ['map', polyline_field]
        self.gdf = gpd.GeoDataFrame(
            merged_gdf.drop(columns=cols_to_drop_final, errors='ignore'),
            crs=target_crs # Set the correct CRS from the start
        )

        # Cleanup: Drop rows with null geometry or missing essential details
        original_len = len(self.gdf)
        required_detail_cols = [created_at_field]
        cols_to_check = ['geometry'] + [col for col in required_detail_cols if col in self.gdf.columns]
        self.gdf = self.gdf.dropna(subset=cols_to_check)
        dropped_count = original_len - len(self.gdf)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} segments due to null geometry or missing essential details ({created_at_field}).")

        if self.gdf.empty:
            logger.error("No valid segments remaining after initial cleanup.")
            self.gdf = None
            return

        # --- Geometry Validation (in target_crs) ---
        logger.info(f"Validating {len(self.gdf)} geometries in CRS {self.gdf.crs}...")
        invalid_mask = ~self.gdf.geometry.is_valid
        num_invalid = invalid_mask.sum()

        if num_invalid > 0:
            logger.warning(f"Found {num_invalid} invalid geometries before simplification. Attempting buffer(0) fix...")
            try:
                # Apply buffer(0) only to invalid geometries
                self.gdf.loc[invalid_mask, 'geometry'] = self.gdf.loc[invalid_mask].geometry.buffer(0)
                # Re-check validity
                fixed_mask = self.gdf.loc[invalid_mask].geometry.is_valid
                num_fixed = fixed_mask.sum()
                num_still_invalid = num_invalid - num_fixed
                logger.info(f"Buffer(0) applied. Fixed: {num_fixed}, Still Invalid: {num_still_invalid}")
                if num_still_invalid > 0:
                     logger.warning(f"Indices still invalid after buffer(0): {self.gdf[invalid_mask][~fixed_mask].index.tolist()[:20]}...")
            except Exception as buffer_e:
                 logger.error(f"Error during buffer(0) fix: {buffer_e}")

            # Remove any geometries that became empty after the fix
            empty_mask = self.gdf.geometry.is_empty
            num_empty = empty_mask.sum()
            if num_empty > 0:
                logger.warning(f"{num_empty} geometries became empty after buffer(0) fix. Removing them.")
                self.gdf = self.gdf[~empty_mask].copy() # Use copy

        # Final check before simplification
        final_valid_count = self.gdf.geometry.is_valid.sum()
        final_total_count = len(self.gdf)
        logger.info(f"Validation complete. Valid geometries before simplification: {final_valid_count}/{final_total_count}")
        if final_valid_count < final_total_count:
             logger.error(f"{final_total_count - final_valid_count} invalid geometries remain before simplification. Check logs.")
             # Decide whether to proceed or stop? For now, proceed but log error.

        if self.gdf.empty:
            logger.error("No valid segments remaining after validation.")
            self.gdf = None
            return

        # --- Simplify Geometries (in target_crs) ---
        # Use a tolerance appropriate for the projected CRS (e.g., meters)
        # Add simplify_tolerance_projected to config (e.g., 0.5 meters)
        simplify_tolerance_proj = self.settings.processing.segment_collection_simplify_tolerance_projected
        if simplify_tolerance_proj > 0:
             logger.info(f"Simplifying geometries with tolerance {simplify_tolerance_proj} (units of {self.gdf.crs.axis_info[0].unit_name})...")
             try:
                #  original_count_simplify = len(self.gdf)
                 # Ensure geometry column is active
                 self.gdf = self.gdf.set_geometry("geometry")
                 self.gdf['geometry'] = self.gdf.geometry.simplify(simplify_tolerance_proj, preserve_topology=True)

                 # Remove any that became empty after simplifying
                 empty_after_simplify = self.gdf.geometry.is_empty
                 num_empty_simplify = empty_after_simplify.sum()
                 if num_empty_simplify > 0:
                      logger.warning(f"Removed {num_empty_simplify} geometries that became empty after simplification.")
                      self.gdf = self.gdf[~empty_after_simplify].copy()

                 logger.info(f"Simplification complete. Shape after empty removal: {self.gdf.shape}")

                 # Final validity check after simplification
                 num_invalid_after_simplify = (~self.gdf.geometry.is_valid).sum()
                 if num_invalid_after_simplify > 0:
                      logger.warning(f"{num_invalid_after_simplify} geometries became invalid AFTER simplification. Check tolerance or data.")
                 else:
                      logger.info("All geometries valid after simplification.")

             except Exception as simplify_e:
                 logger.error(f"Error during geometry simplification: {simplify_e}")
        else:
             logger.info("Skipping simplification as tolerance is zero or negative.")


        # --- Final Log Check ---
        if self.gdf is not None and not self.gdf.empty:
            final_null = self.gdf["geometry"].isna().sum()
            final_empty = self.gdf.geometry.is_empty.sum()
            final_valid = self.gdf.geometry.is_valid.sum()
            logger.info(f"End of load_data geometry processing (CRS: {self.gdf.crs}): Null={final_null}, Empty={final_empty}, Valid={final_valid}, Total={len(self.gdf)}")
            if final_valid < len(self.gdf):
                logger.error(f"{len(self.gdf) - final_valid} geometries are invalid at the end of load_data!")
        else:
             logger.warning("GDF is None or empty at the end of load_data.")


        # --- Preprocessing Steps (Age and Metrics) ---
        if self.gdf is None or self.gdf.empty:
             logger.error("Cannot calculate metrics: GDF is empty or None.")
             return

        logger.info("Calculating segment age and popularity metrics...")
        # ... (rest of the metric calculation code remains the same) ...
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

        logger.info("Strava segments loaded and preprocessed.")
        if self.gdf is not None and not self.gdf.empty:
             self._save_intermediate_gdf(self.gdf, "prepared_segments_gpkg")
        else:
             logger.warning("Final GDF is empty, skipping save of prepared_segments_gpkg.")


    def _fetch_and_cache_segment_details(self, initial_gdf):
        """Fetches and caches segment details from Strava API using CSV."""
        # This function remains largely the same, operating on the combined GDF
        # which already has geometries (potentially null for some)
        cache_path = self.settings.paths.segment_details_cache_csv
        segment_id_field = self.settings.input_data.segment_id_field
        athlete_count_field = self.settings.input_data.segment_athlete_count_field
        star_count_field = self.settings.input_data.segment_star_count_field
        created_at_field = self.settings.input_data.segment_created_at_field
        effort_count_field = self.settings.input_data.segment_effort_count_field
        distance_field = self.settings.input_data.segment_distance_field
        elevation_diff_field = self.settings.input_data.segment_elevation_diff_field # Corresponds to total_elevation_gain

        # Define the specific columns intended for the CSV cache and return
        desired_cache_cols = [
            segment_id_field,
            athlete_count_field,
            star_count_field,
            created_at_field,
            effort_count_field,
            distance_field,
            elevation_diff_field,
        ]
        # Ensure unique columns
        desired_cache_cols = list(dict.fromkeys(desired_cache_cols))


        segment_details_cache_df = pd.DataFrame(columns=desired_cache_cols) # Initialize with desired cols

        if cache_path.exists():
            try:
                # Load cache, potentially containing more columns than desired
                temp_cache_df = pd.read_csv(cache_path)
                # Attempt to infer ID type for comparison/merge
                id_dtype = initial_gdf[segment_id_field].dtype
                if not temp_cache_df.empty and segment_id_field in temp_cache_df.columns:
                    temp_cache_df[segment_id_field] = temp_cache_df[segment_id_field].astype(id_dtype)

                # Keep only desired columns that exist in the loaded cache
                cols_to_keep_from_cache = [col for col in desired_cache_cols if col in temp_cache_df.columns]
                if cols_to_keep_from_cache:
                    segment_details_cache_df = temp_cache_df[cols_to_keep_from_cache]
                    logger.info(f"Loaded {len(segment_details_cache_df)} segment details for columns {cols_to_keep_from_cache} from CSV cache: {cache_path}")
                else:
                    logger.warning(f"Cache file {cache_path} exists but contains none of the desired columns. Starting fresh.")
                    segment_details_cache_df = pd.DataFrame(columns=desired_cache_cols)

            except Exception as e:
                logger.warning(f"Could not load/parse CSV cache {cache_path}: {e}. Starting fresh.")
                segment_details_cache_df = pd.DataFrame(columns=desired_cache_cols)

        cached_ids = set()
        if not segment_details_cache_df.empty and created_at_field in segment_details_cache_df.columns:
             # Consider details cached only if essential fields like 'created_at' are present
             # Use the potentially subsetted segment_details_cache_df here
             valid_cache_df = segment_details_cache_df.dropna(subset=[created_at_field])
             cached_ids = set(valid_cache_df[segment_id_field].unique())
        else:
             logger.info("Cache is empty or missing essential columns for validation.")


        ids_to_fetch = initial_gdf[~initial_gdf[segment_id_field].isin(cached_ids)][segment_id_field].unique().tolist()

        newly_fetched_details = []
        needs_saving = False
        if ids_to_fetch:
            logger.info(f"Fetching details for {len(ids_to_fetch)} segments not found or incomplete in cache...")
            i = 0 # Dev counter
            for segment_id in tqdm(ids_to_fetch, desc="Fetching Segment Details"):
                segment_input = {"id": segment_id}
                details = get_segment_details(segment_input) # Assumes this returns a dict or error code
                if details:
                    if isinstance(details, int) and details == 429:
                        logger.warning("Rate limit likely reached (status 429). Stopping.")
                        break
                    if isinstance(details, dict) and details.get("status_code") == 429:
                        logger.warning("Rate limit likely reached (status 429 dict). Stopping.")
                        break

                    # Extract only the desired columns from the fetched details
                    detail_subset = {col: details.get(col) for col in desired_cache_cols if col in details}
                    # Ensure the segment ID is correctly included
                    detail_subset[segment_id_field] = segment_id
                    # We specifically DO NOT extract 'map' or 'polyline' here for caching

                    newly_fetched_details.append(detail_subset)
                    needs_saving = True
                else:
                    logger.warning(f"Failed to fetch details for segment ID: {segment_id}")
                    # Add placeholder to avoid re-fetching constantly? Or rely on merge 'how=left'?
                    # For now, don't add placeholder, rely on merge.

                time.sleep(self.settings.processing.strava_api_request_delay)
                # --- DEV ONLY ---
                i += 1
                if i > 5: 
                    logger.info("DEV ONLY: Skipping rest of detail fetching.")
                    break
                # ----------------

            if newly_fetched_details:
                new_details_df = pd.DataFrame(newly_fetched_details)
                # Ensure ID types match before concat
                if not segment_details_cache_df.empty:
                     new_details_df[segment_id_field] = new_details_df[segment_id_field].astype(segment_details_cache_df[segment_id_field].dtype)
                segment_details_cache_df = pd.concat([segment_details_cache_df, new_details_df], ignore_index=True)
                # Keep only the latest details for each segment
                segment_details_cache_df = segment_details_cache_df.drop_duplicates(subset=[segment_id_field], keep='last')

        if needs_saving:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Select only the desired cache columns that actually exist in the DataFrame before saving
                cols_to_save = [col for col in desired_cache_cols if col in segment_details_cache_df.columns]
                if cols_to_save:
                    segment_details_cache_df[cols_to_save].to_csv(cache_path, index=False)
                    logger.info(f"Updated segment details cache CSV: {cache_path} with columns: {cols_to_save}")
                else:
                    logger.warning("No columns suitable for saving to cache were found in the DataFrame.")
            except Exception as e:
                logger.error(f"Failed to save updated cache CSV: {e}")
        else:
            logger.info("No new segment details fetched or cache update needed.")

        # Return only the desired cache columns that actually exist in the DataFrame
        cols_to_return = [col for col in desired_cache_cols if col in segment_details_cache_df.columns]
        logger.debug(f"Returning segment details DataFrame with columns: {cols_to_return}")
        return segment_details_cache_df[cols_to_return], cols_to_return


    def _calculate_age(self, created_at_field):
        """Calculates segment age based on configuration."""
        # This function remains the same
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
        # This function remains the same
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

    # --- Build Methods (Raster/Vector) ---
    # These methods (_build_popularity_raster*, _build_popularity_vector, build)
    # should now work correctly as self.gdf is expected to be valid and in the target CRS.
    # They remain unchanged from the previous version.

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
        if self.gdf is None: raise ValueError("Segments GDF not loaded.")
        if self.wbt is None: raise ValueError("WhiteboxTools not initialized.")
        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping IDW.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(0)

        points_gdf = polyline_to_points(self.gdf[["geometry", metric]].dropna(subset=[metric]))
        if points_gdf.empty: logger.warning(f"No valid points for {metric}. Skipping IDW."); return None

        sanitized_metric_field = "".join(filter(str.isalnum, metric))[:10] or "metric"
        points_gdf_shp = points_gdf.rename(columns={metric: sanitized_metric_field})
        input_shp_path = self.settings.paths.output_dir / f"temp_segment_points_{metric}.shp"
        output_raster_path = self._get_output_path("segment_popularity_raster_prefix")
        output_raster_path = output_raster_path.parent / f"{output_raster_path.stem}_{metric}.tif"
        save_vector_data(points_gdf_shp, input_shp_path, driver="ESRI Shapefile")
        try:
            self.wbt.idw_interpolation(i=str(input_shp_path), field=sanitized_metric_field, output=str(output_raster_path),
                                       min_points=self.settings.processing.segment_popularity_idw_min_points,
                                       weight=self.settings.processing.segment_popularity_idw_power,
                                       radius=self.settings.processing.segment_popularity_idw_radius,
                                       cell_size=self.settings.processing.output_cell_size)
            logger.info(f"Generated IDW popularity raster: {output_raster_path}")
            self._save_raster(None, None, "segment_popularity_raster_prefix", metric_name=metric)
            self.output_paths[f"segment_popularity_raster_prefix_{metric}"] = output_raster_path
            return str(output_raster_path)
        except Exception as e: logger.error(f"Error during WBT IDW for {metric}: {e}", exc_info=True); return None
        finally:
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    try: temp_file.unlink()
                    except OSError as unlink_e: logger.warning(f"Could not delete temp file {temp_file}: {unlink_e}")

    def _build_popularity_raster_nn(self, metric: str):
        """Builds a popularity raster using WhiteboxTools Nearest Neighbour Gridding."""
        logger.info(f"Interpolating {metric} using WBT Nearest Neighbour...")
        if self.gdf is None: raise ValueError("Segments GDF not loaded.")
        if self.wbt is None: raise ValueError("WhiteboxTools not initialized.")
        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping NN.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(0)

        points_gdf = polyline_to_points(self.gdf[["geometry", metric]].dropna(subset=[metric]))
        if points_gdf.empty: logger.warning(f"No valid points for {metric}. Skipping NN."); return None

        sanitized_metric_field = "".join(filter(str.isalnum, metric))[:10] or "metric"
        points_gdf_shp = points_gdf.rename(columns={metric: sanitized_metric_field})
        input_shp_path = self.settings.paths.output_dir / f"temp_segment_points_{metric}_nn.shp"
        output_raster_path = self._get_output_path("segment_popularity_raster_prefix")
        output_raster_path = output_raster_path.parent / f"{output_raster_path.stem}_{metric}_nn.tif"
        save_vector_data(points_gdf_shp, input_shp_path, driver="ESRI Shapefile")
        try:
            self.wbt.nearest_neighbour_gridding(i=str(input_shp_path), field=sanitized_metric_field, output=str(output_raster_path),
                                                cell_size=self.settings.processing.output_cell_size,
                                                max_dist=self.settings.processing.segment_popularity_nn_max_dist)
            logger.info(f"Generated NN popularity raster: {output_raster_path}")
            self._save_raster(None, None, "segment_popularity_raster_prefix", metric_name=f"{metric}_nn")
            self.output_paths[f"segment_popularity_raster_prefix_{metric}_nn"] = output_raster_path
            return str(output_raster_path)
        except Exception as e: logger.error(f"Error during WBT NN for {metric}: {e}", exc_info=True); return None
        finally:
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    try: temp_file.unlink()
                    except OSError as unlink_e: logger.warning(f"Could not delete temp file {temp_file}: {unlink_e}")

    def _build_popularity_raster_tin(self, metric: str):
        """Builds a popularity raster using WhiteboxTools TIN Gridding."""
        logger.info(f"Interpolating {metric} using WBT TIN Gridding...")
        if self.gdf is None: raise ValueError("Segments GDF not loaded.")
        if self.wbt is None: raise ValueError("WhiteboxTools not initialized.")
        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping TIN.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(0)

        points_gdf = polyline_to_points(self.gdf[["geometry", metric]].dropna(subset=[metric]))
        if points_gdf.empty: logger.warning(f"No valid points for {metric}. Skipping TIN."); return None

        sanitized_metric_field = "".join(filter(str.isalnum, metric))[:10] or "metric"
        points_gdf_shp = points_gdf.rename(columns={metric: sanitized_metric_field})
        input_shp_path = self.settings.paths.output_dir / f"temp_segment_points_{metric}_tin.shp"
        output_raster_path = self._get_output_path("segment_popularity_raster_prefix")
        output_raster_path = output_raster_path.parent / f"{output_raster_path.stem}_{metric}_tin.tif"
        save_vector_data(points_gdf_shp, input_shp_path, driver="ESRI Shapefile")
        try:
            self.wbt.tin_gridding(i=str(input_shp_path), field=sanitized_metric_field, output=str(output_raster_path),
                                  resolution=self.settings.processing.output_cell_size,
                                  max_triangle_edge_length=self.settings.processing.segment_popularity_tin_max_triangle_edge_length)
            logger.info(f"Generated TIN popularity raster: {output_raster_path}")
            self._save_raster(None, None, "segment_popularity_raster_prefix", metric_name=f"{metric}_tin")
            self.output_paths[f"segment_popularity_raster_prefix_{metric}_tin"] = output_raster_path
            return str(output_raster_path)
        except Exception as e: logger.error(f"Error during WBT TIN for {metric}: {e}", exc_info=True); return None
        finally:
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    try: temp_file.unlink()
                    except OSError as unlink_e: logger.warning(f"Could not delete temp file {temp_file}: {unlink_e}")

    def _build_popularity_raster_kriging(self, metric: str):
        """NOTE: Current kriging implementation is broken. Use IDW instead."""
        logger.warning(f"Attempting Kriging for {metric}. NOTE: geokrige library might be unstable.")
        if self.gdf is None: raise ValueError("Segments GDF not loaded.")
        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping Kriging.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(0)

        points_gdf = polyline_to_points(self.gdf[["geometry", metric]].dropna(subset=[metric]))
        if points_gdf.empty: logger.warning(f"No valid points for {metric}. Skipping Kriging."); return None

        X_known = np.array(points_gdf.geometry.apply(lambda geom: [geom.x, geom.y]).tolist())
        y_known = points_gdf[metric].to_numpy()
        cell_size = self.settings.processing.output_cell_size
        minx, miny, maxx, maxy = points_gdf.total_bounds
        grid_x_coords = np.arange(minx, maxx + cell_size, cell_size)
        grid_y_coords = np.arange(miny, maxy + cell_size, cell_size)
        grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x_coords, grid_y_coords)
        X_predict = np.vstack([grid_x_mesh.ravel(), grid_y_mesh.ravel()]).T
        logger.info(f"Creating Kriging grid: {len(grid_x_coords)} x {len(grid_y_coords)} cells")

        interpolated_grid = None
        try:
            logger.info(f"Running Ordinary Kriging for '{metric}'...")
            OK = OrdinaryKriging(); OK.load(X=X_known, y=y_known)
            fit_successful = False
            try:
                n_bins = 15; logger.info(f"Calculating variogram ({n_bins} bins)...")
                OK.variogram(bins=n_bins); logger.info("Fitting variogram model...")
                OK.fit(model=self.settings.processing.kriging_model)
                if hasattr(OK, "learned_params") and OK.learned_params is not None:
                    fit_successful = True; logger.info(f"Variogram fit successful: {OK.learned_params}")
                else: logger.warning("Variogram fitting did not produce parameters.")
            except ValueError as ve: logger.warning(f"Variogram calculation failed: {ve}.")
            except Exception as fit_e: logger.warning(f"Could not fit variogram: {fit_e}.")

            if fit_successful:
                logger.info("Predicting values on grid..."); zvalues = OK.predict(X=X_predict)
                logger.info(f"Kriging prediction complete for '{metric}'.")
                grid_shape = (len(grid_y_coords), len(grid_x_coords))
                if zvalues.size == grid_shape[0] * grid_shape[1]: interpolated_grid = zvalues.reshape(grid_shape)
                else: raise ValueError(f"Kriging output size {zvalues.size} != grid shape {grid_shape}")
            else: logger.error(f"Skipping prediction for {metric} due to variogram fit failure."); return None
        except Exception as e: logger.error(f"Error during geokrige for {metric}: {e}", exc_info=True); return None

        if interpolated_grid is not None:
            try:
                transform = from_origin(minx, maxy, cell_size, cell_size)
                profile = {"driver": "GTiff", "height": interpolated_grid.shape[0], "width": interpolated_grid.shape[1],
                           "count": 1, "dtype": str(interpolated_grid.dtype), "crs": self.gdf.crs,
                           "transform": transform, "nodata": -9999}
                self._save_raster(interpolated_grid, profile, "segment_popularity_raster_prefix", metric_name=metric)
                output_path = self.output_paths.get(f"segment_popularity_raster_prefix_{metric}")
                return str(output_path) if output_path else None
            except Exception as e: logger.error(f"Error saving Kriging raster for {metric}: {e}", exc_info=True); return None
        else: return None

    @dask.delayed
    def _build_popularity_vector(self, metric: str):
        """Builds a popularity vector layer (buffered segments) for a single metric."""
        logger.info(f"Building popularity vector for metric: {metric}")
        if self.gdf is None: raise ValueError("Segments GDF not loaded.")
        if metric not in self.gdf.columns or self.gdf[metric].isna().all():
            logger.warning(f"Metric '{metric}' not found or all NaN. Skipping vector.")
            return None
        if not pd.api.types.is_numeric_dtype(self.gdf[metric]):
            logger.warning(f"Metric '{metric}' not numeric. Converting.")
            self.gdf[metric] = pd.to_numeric(self.gdf[metric], errors="coerce").fillna(0)

        # Ensure geometries are valid before proceeding (should be after load_data)
        if not self.gdf.geometry.is_valid.all():
            logger.warning(f"Found invalid geometries before building vector for {metric}. Skipping.")
            # Optionally, could try buffer(0) here again, but load_data should handle it.
            return None

        metric_gdf = self.gdf[[self.settings.input_data.segment_id_field, "geometry", metric]].copy()
        metric_gdf = metric_gdf.dropna(subset=[metric])
        if metric_gdf.empty: logger.warning(f"No valid segments for {metric} after NaN drop."); return None

        # Log-Normalize Metric
        log_metric = np.log(metric_gdf[metric] + 1); min_val = log_metric.min(); max_val = log_metric.max()
        norm_col = f"{metric}_norm"
        if max_val == min_val: metric_gdf[norm_col] = 0.0
        else: metric_gdf[norm_col] = (log_metric - min_val) / (max_val - min_val)
        logger.info(f"Normalized '{metric}' to '{norm_col}' (Log Min/Max: {min_val:.2f}/{max_val:.2f})")

        # Apply Distance Buffer (Optional - Controlled by config, but code removed for simplicity now)
        # buffer_dist = self.settings.processing.segment_popularity_buffer_distance
        # if buffer_dist > 0:
        #     logger.info(f"Applying buffer of {buffer_dist}m...")
        #     try:
        #         metric_gdf = metric_gdf.set_geometry("geometry")
        #         metric_gdf["geometry"] = metric_gdf.geometry.buffer(buffer_dist, cap_style=2, join_style=2)
        #         if not metric_gdf.geometry.is_valid.all(): logger.warning("Invalid geometries after buffering.")
        #     except Exception as e: logger.error(f"Error buffering {metric}: {e}", exc_info=True); return None
        # else: logger.info("Skipping distance buffer (distance <= 0).")


        # Add Color Column
        def grayscale_hex(norm_value):
            if pd.isna(norm_value): return "#808080"
            intensity = int(norm_value * 255); return f"#{intensity:02x}{intensity:02x}{intensity:02x}"
        metric_gdf["color"] = metric_gdf[norm_col].apply(grayscale_hex)
        logger.info("Added grayscale 'color' column.")

        # Save Output
        output_path_key = "segment_popularity_vector_prefix"
        output_vector_path = self._get_output_path(output_path_key)
        output_vector_path = output_vector_path.parent / f"{output_vector_path.stem}_{metric}.gpkg"
        try:
            save_vector_data(metric_gdf, output_vector_path, driver="GPKG")
            logger.info(f"Saved popularity vector: {output_vector_path}")
            self.output_paths[f"{output_path_key}_{metric}"] = output_vector_path
            # Add Folium Display Call
            try:
                map_output_path = output_vector_path.with_suffix(".html")
                display_vectors_on_folium_map(
                    gdf=metric_gdf, output_html_path_str=str(map_output_path), style_column=norm_col,
                    cmap_name='viridis', tooltip_columns=[self.settings.input_data.segment_id_field, metric, norm_col],
                    popup_columns=[self.settings.input_data.segment_id_field, metric, norm_col, 'color'],
                    line_weight=3 # Use line_weight for LineStrings
                )
                logger.info(f"Saved popularity vector Folium map: {map_output_path}")
            except Exception as map_e: logger.error(f"Error generating Folium map for {metric}: {map_e}", exc_info=True)
            return str(output_vector_path)
        except Exception as e: logger.error(f"Error saving vector for {metric}: {e}", exc_info=True); return None


    def build(self):
        """Builds popularity vectors (or rasters) for configured metrics."""
        if self.gdf is None: self.load_data()
        if self.gdf is None: logger.error("Cannot build Segments: Data loading failed."); return []

        tasks = []
        available_metrics = [m for m in self.settings.processing.segment_popularity_metrics
                             if m in self.gdf.columns and pd.api.types.is_numeric_dtype(self.gdf[m]) and not self.gdf[m].isna().all()]
        skipped_metrics = [m for m in self.settings.processing.segment_popularity_metrics if m not in available_metrics]
        if skipped_metrics: logger.warning(f"Metrics skipped (not found/numeric/all NaN): {skipped_metrics}")

        # --- Choose Output Type (Vector default) ---
        build_vector = True # TODO: Configurable?
        # build_raster = False

        for metric in available_metrics:
            if build_vector: tasks.append(self._build_popularity_vector(metric))
            # elif build_raster: tasks.append(self._build_popularity_raster(metric))

        if not tasks: logger.warning("No valid metrics found to build outputs."); return []

        logger.info(f"Computing {len(tasks)} popularity output tasks...")
        results = dask.compute(*tasks)
        logger.info("Popularity output computation finished.")
        successful_outputs = [r for r in results if r is not None]
        return successful_outputs


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    from whitebox import WhiteboxTools
    from src.config import settings

    logger.info("--- Running segments.py Standalone Test ---")
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output Directory: {settings.paths.output_dir}")

    wbt = WhiteboxTools()
    cluster = LocalCluster(n_workers=4, threads_per_worker=1) # Dask optional for testing
    client = Client(cluster)
    # client = None # Run sequentially for simpler debugging if needed
    # logger.info(f"Dask client started: {client.dashboard_link if client else 'Sequential execution'}")

    try:
        logger.info("--- Testing Segments Feature ---")
        segments_feature = Segments(settings, wbt=wbt)
        logger.info("1. Testing Load Data...")
        segments_feature.load_data()
        if segments_feature.gdf is not None:
            logger.info(f"Load Data successful. Shape: {segments_feature.gdf.shape}, CRS: {segments_feature.gdf.crs}")
            print("Sample preprocessed segments (first 5 rows):"); print(segments_feature.gdf.head())
            expected_metrics = settings.processing.segment_popularity_metrics
            actual_metrics = [m for m in expected_metrics if m in segments_feature.gdf.columns]
            logger.info(f"Expected metrics: {expected_metrics}, Found: {actual_metrics}")
            if actual_metrics: print("\nMetrics sample:"); print(segments_feature.gdf[actual_metrics].head())
        else: logger.error("Segments GDF is None after loading.")

        logger.info("2. Testing Build...")
        if segments_feature.gdf is not None:
            logger.info(f"Testing build with interpolation: {settings.processing.interpolation_method_polylines} (if raster), vector default")
            output_paths = segments_feature.build()
            logger.info("Build process completed.")
            if output_paths: logger.info(f"Generated Outputs: {output_paths}")
            else: logger.warning("No outputs generated by build.")
        else: logger.warning("Skipping build test as data loading failed.")
        logger.info("--- Segments Feature Test Completed ---")
    except Exception as e: logger.error(f"Error during Segments test: {e}", exc_info=True)
    finally:
        if client:
            try: 
                client.close()
                cluster.close()
                logger.info("Dask client/cluster closed.")
            except Exception as e: 
                logger.warning(f"Error closing Dask: {e}")
    logger.info("--- Standalone Test Finished ---")
