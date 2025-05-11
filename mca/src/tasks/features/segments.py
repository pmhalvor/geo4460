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
from pathlib import Path

from src.tasks.features.feature_base import FeatureBase
from src.strava.explore import get_segment_details
from src.utils import (
    load_vector_data,
    polyline_to_points,
    save_vector_data,
    display_vectors_on_folium_map,
    reproject_gdf, # Explicitly import reproject_gdf
)

logger = logging.getLogger(__name__)


class Segments(FeatureBase):
    """Handles Strava segment data processing, including API fetching and caching."""

    def _load_single_segment_source(self, path, id_field, target_crs):
        """Loads a single segment source file (GeoJSON or GPKG) and ensures target CRS."""
        logger.info(f"Loading segments from: {path}")
        if not path.exists():
            logger.warning(f"Path {path} does not exist. Skipping.")
            return None
        gdf = load_vector_data(path)
        if gdf is None:
            logger.warning(f"Could not load segments from {path}")
            return None
        if id_field not in gdf.columns:
            logger.warning(f"Segment ID field '{id_field}' not found in {path}. Skipping.")
            return None
        gdf = reproject_gdf(gdf, target_crs)
        logger.info(f"Loaded and ensured CRS {target_crs} for {len(gdf)} segments from {path}.")
        return gdf

    def _combine_segment_sources(self, gdf1, gdf2, id_field):
        """Combines two segment GeoDataFrames, ensuring CRS match and dropping duplicates."""
        if gdf1 is None and gdf2 is None:
            logger.error("No segment data sources could be loaded.")
            return None
        elif gdf1 is None:
            logger.info("Using only second segment source.")
            return gdf2
        elif gdf2 is None:
            logger.info("Using only first segment source.")
            return gdf1
        else:
            logger.info("Combining segment sources...")
            if not gdf1.crs.equals(gdf2.crs):
                logger.error(f"CRS mismatch before combining: {gdf1.crs} vs {gdf2.crs}. Aborting.")
                return None
            combined_gdf = pd.concat([gdf1, gdf2], ignore_index=True, sort=False)
            combined_gdf = combined_gdf.drop_duplicates(subset=[id_field], keep='last')
            logger.info(f"Combined data has {len(combined_gdf)} unique segments (CRS: {combined_gdf.crs}).")
            return combined_gdf

    def _merge_segment_details(self, combined_gdf, details_df, id_field, cache_cols):
        """Merges fetched/cached details onto the combined segment GDF."""
        if details_df is None or details_df.empty:
            logger.warning("Segment details cache is empty. Proceeding without detailed attributes.")
            merged_gdf = combined_gdf.copy()
            # Ensure expected columns exist, even if empty
            for col in cache_cols:
                if col != id_field and col not in merged_gdf.columns:
                    merged_gdf[col] = np.nan
            return merged_gdf

        cols_to_drop_before_merge = [
            col for col in cache_cols if col != id_field and col in combined_gdf.columns
        ]
        combined_gdf_clean = combined_gdf.drop(columns=cols_to_drop_before_merge, errors="ignore")

        try:
            # Align ID types for robust merging
            if id_field in combined_gdf_clean.columns and id_field in details_df.columns:
                combined_gdf_clean[id_field] = combined_gdf_clean[id_field].astype(details_df[id_field].dtype)
            else:
                 logger.warning(f"ID field '{id_field}' missing from one of the dataframes for merge.")
                 return combined_gdf_clean # Return without merging if IDs are missing

        except Exception as e:
            logger.warning(f"Could not align ID types for merge: {e}. Proceeding with merge attempt.")

        details_to_merge = details_df[cache_cols] # Use only the intended cache columns
        merged_gdf = pd.merge(
            combined_gdf_clean, details_to_merge, on=id_field, how="left"
        )
        logger.info(f"Merged details from cache/API onto {len(merged_gdf)} segments.")

        # Check for missing essential details after merge
        created_at_field = self.settings.input_data.segment_created_at_field
        if created_at_field in merged_gdf.columns:
            missing_details_count = merged_gdf[created_at_field].isna().sum()
            if missing_details_count > 0:
                logger.warning(f"{missing_details_count} segments missing '{created_at_field}' after cache/API merge.")
        elif created_at_field in cache_cols: # Only warn if it was expected
             logger.warning(f"'{created_at_field}' column not found after merge, though expected from cache.")

        return merged_gdf

    def _decode_and_reproject_polylines(self, gdf, polyline_field, target_crs, source_crs="EPSG:4326"):
        """Decodes polylines for segments missing valid geometry and reprojects them if needed."""
        logger.info("Checking for and processing missing geometries from polylines...")
        # Identify rows needing geometry from polyline
        needs_geom_mask = gdf['geometry'].isnull() | ~gdf.geometry.is_valid
        rows_to_decode = gdf[needs_geom_mask]
        decoded_geoms_data = [] # List to hold {'index': index, 'geometry': geom_4326}

        if rows_to_decode.empty:
            logger.info("No segments required geometry decoding from polylines.")
            return gdf # Return original GDF if no decoding needed

        logger.info(f"Found {len(rows_to_decode)} segments potentially needing geometry decoded from polyline.")
        for index, row in rows_to_decode.iterrows():
            encoded_polyline = None
            # Prioritize 'map.polyline' if available, then the direct 'polyline' field
            if 'map' in row and isinstance(row.get('map'), dict) and polyline_field in row['map']:
                encoded_polyline = row['map'][polyline_field]
            elif polyline_field in row and pd.notna(row[polyline_field]):
                encoded_polyline = row[polyline_field]
            else:
                logger.debug(f"Segment index {index}: No polyline found in 'map' or '{polyline_field}' field.")
                continue

            if encoded_polyline:
                try:
                    decoded_coords = polyline.decode(encoded_polyline)
                    # Strava polylines are (lat, lon), need (lon, lat) for LineString
                    lon_lat_coords = [(lon, lat) for lat, lon in decoded_coords]
                    if len(lon_lat_coords) >= 2:
                        geom_4326 = LineString(lon_lat_coords)
                        decoded_geoms_data.append({'index': index, 'geometry': geom_4326})
                    else:
                        logger.warning(f"Segment index {index}: Decoded polyline has < 2 points.")
                except Exception as e:
                    logger.warning(f"Segment index {index}: Failed to decode polyline '{encoded_polyline}': {e}")
            else:
                 logger.debug(f"Segment index {index}: Missing geometry and no polyline found.")

        # Reproject decoded geometries if any were created
        if decoded_geoms_data:
            logger.info(f"Successfully decoded {len(decoded_geoms_data)} polylines.")
            temp_decoded_gdf = gpd.GeoDataFrame(decoded_geoms_data, crs=source_crs)
            temp_decoded_gdf = temp_decoded_gdf.set_index('index') # Use original index

            # Only reproject if the target CRS is different from source_crs
            if source_crs != target_crs:
                logger.info(f"Reprojecting decoded polylines from {source_crs} to {target_crs}...")
                try:
                    temp_reprojected_gdf = reproject_gdf(temp_decoded_gdf, target_crs)
                    # Update the main GeoDataFrame using the index
                    gdf.loc[temp_reprojected_gdf.index, 'geometry'] = temp_reprojected_gdf['geometry']
                    logger.info("Updated main GDF with reprojected geometries from polylines.")
                except Exception as reproj_e:
                    logger.error(f"Failed to reproject decoded polylines: {reproj_e}. Geometries for these segments remain unset.")
            else:
                # No reprojection needed, use the decoded geometries directly
                logger.info(f"No reprojection needed as source ({source_crs}) and target ({target_crs}) CRS are the same.")
                gdf.loc[temp_decoded_gdf.index, 'geometry'] = temp_decoded_gdf['geometry']
                logger.info("Updated main GDF with decoded polyline geometries (no reprojection needed).")

        return gdf

    def _create_final_gdf(self, merged_gdf, target_crs, polyline_field):
        """Creates the final GeoDataFrame, dropping helper columns."""
        cols_to_drop_final = ['map', polyline_field] # Add other potential intermediate columns if needed
        final_gdf = gpd.GeoDataFrame(
            merged_gdf.drop(columns=cols_to_drop_final, errors='ignore'),
            crs=target_crs
        )
        return final_gdf

    def _cleanup_gdf(self, gdf, required_cols):
        """Drops rows with null geometry or missing essential details."""
        if gdf is None or gdf.empty:
            return gdf
        original_len = len(gdf)
        logger.info(f"Cleaning up GDF with {original_len} segments...")
        logger.info(f"Columns in GDF: {gdf.columns.tolist()}")
        cols_to_check = ['geometry'] + [col for col in required_cols if col in gdf.columns]
        logger.info(f"Checking for nulls in columns: {cols_to_check}")
        cleaned_gdf = gdf.dropna(subset=cols_to_check)
        dropped_count = original_len - len(cleaned_gdf)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} segments due to null geometry or missing essential details ({required_cols}).")
            if dropped_count == original_len:
                logger.error("All segments dropped during cleanup. Check input data.")
                raise ValueError("All segments dropped during cleanup.")
        if cleaned_gdf.empty:
            logger.error("No valid segments remaining after initial cleanup.")
            return None
        return cleaned_gdf

    def _validate_geometries(self, gdf):
        """Validates geometries, attempting buffer(0) fix for invalid ones."""
        if gdf is None or gdf.empty:
            return gdf
        logger.info(f"Validating {len(gdf)} geometries in CRS {gdf.crs}...")
        invalid_mask = ~gdf.geometry.is_valid
        num_invalid = invalid_mask.sum()

        if num_invalid > 0:
            logger.warning(f"Found {num_invalid} invalid geometries. Attempting buffer(0) fix...")
            try:
                # Apply buffer(0) only to invalid geometries
                gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask].geometry.buffer(0)
                # Re-check validity
                fixed_mask = gdf.loc[invalid_mask].geometry.is_valid
                num_fixed = fixed_mask.sum()
                num_still_invalid = num_invalid - num_fixed
                logger.info(f"Buffer(0) applied. Fixed: {num_fixed}, Still Invalid: {num_still_invalid}")
                if num_still_invalid > 0:
                     logger.warning(f"Indices still invalid after buffer(0): {gdf[invalid_mask][~fixed_mask].index.tolist()[:20]}...")
            except Exception as buffer_e:
                 logger.error(f"Error during buffer(0) fix: {buffer_e}")

            # Remove any geometries that became empty after the fix
            empty_mask = gdf.geometry.is_empty
            num_empty = empty_mask.sum()
            if num_empty > 0:
                logger.warning(f"{num_empty} geometries became empty after buffer(0) fix. Removing them.")
                gdf = gdf[~empty_mask].copy() # Use copy

        # Final check
        final_valid_count = gdf.geometry.is_valid.sum()
        final_total_count = len(gdf)
        logger.info(f"Validation complete. Valid geometries: {final_valid_count}/{final_total_count}")
        if final_valid_count < final_total_count:
             logger.error(f"{final_total_count - final_valid_count} invalid geometries remain. Check logs.")

        if gdf.empty:
            logger.error("No valid segments remaining after validation.")
            return None
        return gdf

    def _simplify_geometries(self, gdf, tolerance):
        """Simplifies geometries with a given tolerance."""
        if gdf is None or gdf.empty:
            return gdf
        if tolerance <= 0:
             logger.info("Skipping simplification as tolerance is zero or negative.")
             return gdf

        logger.info(f"Simplifying geometries with tolerance {tolerance} (units of {gdf.crs.axis_info[0].unit_name})...")
        try:
             # Ensure geometry column is active
             gdf = gdf.set_geometry("geometry")
             gdf['geometry'] = gdf.geometry.simplify(tolerance, preserve_topology=True)

             # Remove any that became empty after simplifying
             empty_after_simplify = gdf.geometry.is_empty
             num_empty_simplify = empty_after_simplify.sum()
             if num_empty_simplify > 0:
                  logger.warning(f"Removed {num_empty_simplify} geometries that became empty after simplification.")
                  gdf = gdf[~empty_after_simplify].copy()

             logger.info(f"Simplification complete. Shape after empty removal: {gdf.shape}")

             # Final validity check after simplification
             num_invalid_after_simplify = (~gdf.geometry.is_valid).sum()
             if num_invalid_after_simplify > 0:
                  logger.warning(f"{num_invalid_after_simplify} geometries became invalid AFTER simplification. Check tolerance or data.")
             else:
                  logger.info("All geometries valid after simplification.")

        except Exception as simplify_e:
             logger.error(f"Error during geometry simplification: {simplify_e}")

        # Final log check
        if gdf is not None and not gdf.empty:
            final_null = gdf["geometry"].isna().sum()
            final_empty = gdf.geometry.is_empty.sum()
            final_valid = gdf.geometry.is_valid.sum()
            logger.info(f"End of simplify_geometries: Null={final_null}, Empty={final_empty}, Valid={final_valid}, Total={len(gdf)}")
            if final_valid < len(gdf):
                logger.error(f"{len(gdf) - final_valid} geometries are invalid after simplification!")
        else:
             logger.warning("GDF is None or empty after simplification.")

        return gdf

    def load_data(self):
        """
        Loads segment data, combines sources, fetches details, processes geometry,
        calculates metrics, and saves the preprocessed data.
        """
        logger.info("--- Starting Segment Loading and Preprocessing ---")
        segment_id_field = self.settings.input_data.segment_id_field
        target_crs = f"EPSG:{self.settings.processing.map_crs_epsg}"
        polyline_field = self.settings.input_data.segment_polyline_field
        created_at_field = self.settings.input_data.segment_created_at_field

        # --- 1. Load and Combine Sources ---
        geojson_gdf = self._load_single_segment_source(
            self.settings.paths.strava_segments_geojson, segment_id_field, target_crs
        )
        gpkg1_gdf = self._load_single_segment_source(
            self.settings.paths.collected_segments_gpkg, segment_id_field, target_crs
        )
        # Load segments collected based on simple roads
        gpkg2_gdf = self._load_single_segment_source(
            self.settings.paths.collected_segments_from_simple_roads_gpkg, segment_id_field, target_crs
        )
        # Combine all three sources
        combined_gdf = self._combine_segment_sources(geojson_gdf, gpkg1_gdf, segment_id_field)
        combined_gdf = self._combine_segment_sources(combined_gdf, gpkg2_gdf, segment_id_field)

        if combined_gdf is None:
            self.gdf = None
            return # Stop if no data could be combined

        # --- 2. Fetch/Cache and Merge Details ---
        segment_details_cache_df, cache_cols = self._fetch_and_cache_segment_details(combined_gdf)
        merged_gdf = self._merge_segment_details(combined_gdf, segment_details_cache_df, segment_id_field, cache_cols)

        # --- 3. Process Geometry (Decode, Reproject, Validate, Simplify) ---
        gdf_with_polylines = self._decode_and_reproject_polylines(merged_gdf, polyline_field, target_crs)
        final_gdf_structure = self._create_final_gdf(gdf_with_polylines, target_crs, polyline_field)
        cleaned_gdf = self._cleanup_gdf(final_gdf_structure, [created_at_field]) # Cleanup based on essential cols
        validated_gdf = self._validate_geometries(cleaned_gdf)
        # simplified_gdf = self._simplify_geometries(
        #     validated_gdf, self.settings.processing.segment_collection_simplify_tolerance_projected
        # ) NOTE remove simplification to avoid straight lines

        # self.gdf = simplified_gdf # Assign the fully processed GDF to self.gdf
        self.gdf = validated_gdf.copy() # Assign the fully processed GDF to self.gdf

        # --- 4. Calculate Metrics ---
        if self.gdf is None or self.gdf.empty:
             logger.error("Cannot calculate metrics: GDF is empty or None after geometry processing.")
             return

        logger.info("Calculating segment age and popularity metrics...")
        athlete_count_field = self.settings.input_data.segment_athlete_count_field
        effort_count_field = self.settings.input_data.segment_effort_count_field
        star_count_field = self.settings.input_data.segment_star_count_field
        required_cols_final = [
            created_at_field, athlete_count_field, effort_count_field, star_count_field,
        ]
        # Check if required columns exist *after* all processing
        missing_cols_final = [col for col in required_cols_final if col not in self.gdf.columns]
        if missing_cols_final:
            logger.error(f"Missing required columns for metric calculation: {missing_cols_final}. Cannot proceed.")
            self.gdf = None # Invalidate GDF if essential metrics are missing
            return

        age_col = self._calculate_age(created_at_field) # Assumes self.gdf is updated
        if age_col:
            self._calculate_popularity_metrics(
                age_col, athlete_count_field, effort_count_field, star_count_field
            ) # Assumes self.gdf is updated
        else:
            logger.warning("Skipping popularity metric calculation due to age calculation failure.")
            # Ensure metric columns exist even if calculation failed
            for metric_config_name in self.settings.processing.segment_popularity_metrics:
                if metric_config_name not in self.gdf.columns:
                    self.gdf[metric_config_name] = np.nan

        # --- 5. Save Final Preprocessed Data ---
        logger.info("Strava segments loaded and preprocessed.")
        if self.gdf is not None and not self.gdf.empty:
             self._save_intermediate_gdf(self.gdf, "prepared_segments_gpkg")
        else:
             logger.warning("Final GDF is empty, skipping save of prepared_segments_gpkg.")
        logger.info("--- Segment Loading and Preprocessing Finished ---")

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
                id_dtype = type(initial_gdf[segment_id_field].iloc[0])
                if not temp_cache_df.empty and segment_id_field in temp_cache_df.columns:
                    logger.debug(f"Converting {segment_id_field} to {id_dtype} for cache comparison.")
                    temp_cache_df[segment_id_field] = temp_cache_df[segment_id_field].astype(id_dtype)
                    logger.debug(f"Converted {segment_id_field} to {type(temp_cache_df[segment_id_field].iloc[0])} for cache comparison.")

                # Keep only desired columns that exist in the loaded cache
                cols_to_keep_from_cache = [col for col in desired_cache_cols if col in temp_cache_df.columns]
                logger.info(f"Columns to keep from cache: {cols_to_keep_from_cache}")

                # Drop duplicates 
                temp_cache_df = temp_cache_df.drop_duplicates(subset=[segment_id_field], keep='last')

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
             cached_ids = set(valid_cache_df[segment_id_field].astype(int).astype(str).unique())  # astype conversion avoid extra ''
        else:
             logger.info("Cache is empty or missing essential columns for validation.")

        # Debugging segment details fetch repeats
        logger.debug(f"Initial gdf has columns: {initial_gdf.columns.tolist()}")
        logger.debug(f"Initial id column type: {type(initial_gdf[segment_id_field].iloc[0])}")
        logger.debug(f"Details cache has columns: {segment_details_cache_df.columns.tolist()}")
        logger.debug(f"Details cache id column type: {type(segment_details_cache_df[segment_id_field].iloc[0])}")
        logger.debug(f"Filtering out {len(cached_ids)} cached segment IDs from initial gdf.")
        logger.debug(f"Cached IDs type: {type(list(cached_ids)[0])}")
        logger.debug(f"Initial gdf IDs: {initial_gdf[segment_id_field].unique()} (type: {type(initial_gdf[segment_id_field].iloc[0])})")

        # ensure all segment_id_field values of same type
        initial_gdf.loc[:, segment_id_field] = initial_gdf[segment_id_field].astype(int).astype(str)
        initial_gdf = initial_gdf.copy().drop_duplicates(subset=[segment_id_field], keep='last')

        ids_to_fetch = set(initial_gdf[segment_id_field].unique().tolist()) - set(cached_ids)
        ids_to_fetch = list(ids_to_fetch)
        logger.debug(f"IDs to fetch: {ids_to_fetch} (type: {type(ids_to_fetch[0])})") if len(ids_to_fetch) else None

        newly_fetched_ids = set()
        newly_fetched_details = []
        needs_saving = False
        if ids_to_fetch:
            logger.info(f"Fetching details for {len(ids_to_fetch)} segments without details in cache...")
            i = 0 # Dev counter
            for segment_id in tqdm(ids_to_fetch, desc="Fetching Segment Details"):
                segment_input = {"id": segment_id}
                details = get_segment_details(segment_input) # Assumes this returns a dict or error code
                if details:
                    if isinstance(details, int) and details == 429:
                        logger.warning("Rate limit reached (status 429). Stopping.")
                        break
                    if isinstance(details, dict) and details.get("status_code") == 429:
                        logger.warning("Rate limit reached (status 429 dict). Stopping.")
                        break

                    # Extract only the desired columns from the fetched details
                    detail_subset = {col: details.get(col) for col in desired_cache_cols if col in details}
                    # Ensure the segment ID is correctly included
                    detail_subset[segment_id_field] = segment_id
                    # We specifically DO NOT extract 'map' or 'polyline' here for caching

                    newly_fetched_details.append(detail_subset)
                    newly_fetched_ids.add(segment_id)
                    needs_saving = True
                else:
                    logger.warning(f"Failed to fetch details for segment ID: {segment_id}")
                    # Add placeholder to avoid re-fetching constantly? Or rely on merge 'how=left'?
                    # For now, don't add placeholder, rely on merge.

                time.sleep(self.settings.processing.strava_api_request_delay)
                
                # Respect API rate limits
                i += 1
                if i > self.settings.processing.segment_details_max_api_calls: 
                    logger.info("Reached segment_details_max_api_calls. Skipping rest of detail fetching.")
                    break

            if newly_fetched_details:
                logger.info(f"Fetched details for segment ids: \n{newly_fetched_ids}.")
                new_details_df = pd.DataFrame(newly_fetched_details)
                # Ensure ID types match before concat
                new_details_df = pd.DataFrame(newly_fetched_details)
                # Ensure ID types match before concat, using the initial GDF as reference if cache is empty
                id_dtype_ref = initial_gdf[segment_id_field].dtype if not segment_details_cache_df.empty else new_details_df[segment_id_field].dtype
                if not segment_details_cache_df.empty:
                    segment_details_cache_df[segment_id_field] = segment_details_cache_df[segment_id_field].astype(id_dtype_ref)
                new_details_df[segment_id_field] = new_details_df[segment_id_field].astype(id_dtype_ref)

                # Combine existing cache with ONLY the newly fetched details
                logger.info(f"Combining {len(segment_details_cache_df)} cached details with {len(new_details_df)} newly fetched details.")
                # Ensure columns align before concat, adding missing columns with NaN
                all_cols = list(dict.fromkeys(segment_details_cache_df.columns.tolist() + new_details_df.columns.tolist()))
                segment_details_cache_df = segment_details_cache_df.reindex(columns=all_cols)
                new_details_df = new_details_df.reindex(columns=all_cols)

                combined_df = pd.concat([segment_details_cache_df, new_details_df], ignore_index=True)

                # Keep only the latest details for each segment ID (handles potential overlaps/duplicates)
                # Using 'last' ensures newly fetched data takes precedence if an ID somehow existed in both.
                final_cache_df = combined_df.drop_duplicates(subset=[segment_id_field], keep='last')
                logger.info(f"Combined cache size before saving: {len(final_cache_df)} segments.")
                segment_details_cache_df = final_cache_df # Update the main variable
            else:
                 logger.info("No new details were fetched.")

        if needs_saving:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Select only the desired cache columns that actually exist in the DataFrame before saving
                # Use the updated segment_details_cache_df which now contains combined data
                cols_to_save = [col for col in desired_cache_cols if col in segment_details_cache_df.columns]
                if cols_to_save:
                    logger.info(f"Saving updated cache with columns: {cols_to_save}")
                    segment_details_cache_df[cols_to_save].to_csv(cache_path, index=False)
                    logger.info(f"Updated segment details cache CSV: {cache_path} with {len(segment_details_cache_df)} segments.")
                else:
                    logger.warning("No columns suitable for saving to cache were found in the DataFrame.")
            except Exception as e:
                logger.error(f"Failed to save updated cache CSV: {e}")
        else:
            logger.info("No new segment details fetched or cache update needed.")

        # Return only the desired cache columns that actually exist in the DataFrame
        # Use the potentially updated segment_details_cache_df
        cols_to_return = [col for col in desired_cache_cols if col in segment_details_cache_df.columns]
        logger.info(f"Returning segment details DataFrame with columns: {cols_to_return}")
        if not cols_to_return:
             logger.warning("Returning empty DataFrame as no desired columns exist.")
             return pd.DataFrame(columns=desired_cache_cols), desired_cache_cols # Return empty DF with expected columns
        return segment_details_cache_df[cols_to_return], cols_to_return

    def _calculate_age(self, created_at_field):
        """Calculates segment age based on configuration."""
        # This function remains the same
        age_col = None

        try:
            self.gdf[created_at_field] = pd.to_datetime(
                self.gdf[created_at_field], errors="coerce", utc=True
            )
            logging.info(f"Converted '{created_at_field}' to datetime.")
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


    @dask.delayed
    def _build_popularity_vector(self, metric: str):
        """Builds a popularity vector layer (buffered segments) for a single metric."""
        logger.info(f"Building popularity vector for metric: {metric}")
        if self.gdf is None: 
            raise ValueError("Segments GDF not loaded.")
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
        if metric_gdf.empty: 
            logger.warning(f"No valid segments for {metric} after NaN drop."); return None

        # Log-Normalize Metric
        log_metric = np.log(metric_gdf[metric] + 1); min_val = log_metric.min(); max_val = log_metric.max()
        norm_col = f"{metric}_norm"
        if max_val == min_val: 
            metric_gdf[norm_col] = 0.0
        else: 
            metric_gdf[norm_col] = (log_metric - min_val) / (max_val - min_val)
        logger.info(f"Normalized '{metric}' to '{norm_col}' (Log Min/Max: {min_val:.2f}/{max_val:.2f})")


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
        except Exception as e: 
            logger.error(f"Error saving vector for {metric}: {e}", exc_info=True);
            return None
        
        # Return the output path when successfully saved
        return str(output_vector_path)

    def build(self):
        """Builds popularity vectors (or rasters) for configured metrics."""
        if self.gdf is None: 
            self.load_data()
        if self.gdf is None: 
            logger.error("Cannot build Segments: Data loading failed."); return []

        tasks = []
        available_metrics = [m for m in self.settings.processing.segment_popularity_metrics
                             if m in self.gdf.columns and pd.api.types.is_numeric_dtype(self.gdf[m]) and not self.gdf[m].isna().all()]
        skipped_metrics = [m for m in self.settings.processing.segment_popularity_metrics if m not in available_metrics]
        if skipped_metrics: logger.warning(f"Metrics skipped (not found/numeric/all NaN): {skipped_metrics}")

        # --- Choose Output Type (Vector default) ---
        build_vector = True # TODO: Configurable?
        # build_raster = False

        for metric in available_metrics:
            if build_vector: 
                tasks.append(self._build_popularity_vector(metric))
            else:
                raise NotImplementedError("Raster building properly not implemented yet.")
            # elif build_raster: tasks.append(self._build_popularity_raster(metric))

        if not tasks:
            logger.warning("No valid metrics found to build outputs.")
            # Return an empty list or a placeholder indicating no tasks
            # Returning the empty list of tasks is consistent
            return tasks

        logger.info(f"Returning {len(tasks)} delayed popularity output tasks for Dask computation.")
            
        # Return the list of delayed objects directly, without computing here
        return tasks

    def create_multi_layer_visualization(self, output_paths=[]):
        """
        Creates a single multi-layer visualization of all segment popularity metrics.
        This combines all the individual vector layers into one folium map.
        """
        logger.info("Creating multi-layer visualization for all segment popularity metrics...")
        
        if not output_paths:
            logger.warning("No output paths available for visualization.")
            return None
        
        # Define output HTML path
        output_html_path = self.settings.paths.output_dir / "segment_popularity_multi_layer_map.html"
        
        try:
            # Define layers for multi-layer visualization
            layers = []
            
            # Add each metric as a separate layer
            metric_to_show = Path(output_paths[0]).stem.split('vector_')[-1]
            for vector_path in output_paths:
                # ensure vector_path is Path() object
                if isinstance(vector_path, str):
                    vector_path = Path(vector_path)

                # Extract metric name from file path
                metric_name = vector_path.stem.split('vector_')[-1]
                
                # Load the GeoDataFrame to get column names
                try:
                    gdf = gpd.read_file(vector_path)
                    norm_col = f"{metric_name}_norm" if f"{metric_name}_norm" in gdf.columns else None
                    
                    if not norm_col:
                        logger.warning(f"Normalized column not found for {metric_name}, using original metric")
                        norm_col = metric_name
                    
                    # Define columns for tooltip and popup
                    logger.info(f"GeoDataFrame at {vector_path} has columns: {gdf.columns.tolist()}")
                    id_field = self.settings.input_data.segment_id_field
                    tooltip_cols = [col for col in [id_field, metric_name, norm_col] if col in gdf.columns]
                    popup_cols = [col for col in [id_field, metric_name, norm_col, 'color'] if col in gdf.columns]
                    
                    # Add layer configuration
                    layers.append({
                        'path': vector_path,
                        'name': f'{metric_name}',
                        'type': 'vector',
                        'vector': {
                            'style_column': norm_col,
                            'cmap': 'viridis',
                            'line_weight': 3,  # Thicker lines for better visibility
                            'tooltip_cols': tooltip_cols,
                            'popup_cols': popup_cols,
                            'show': True if metric_name == metric_to_show else False  # Show only first layer by default
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Error loading GeoDataFrame for {vector_path}: {e}")
                    # Still add the layer with basic configuration
                    layers.append({
                        'path': vector_path,
                        'name': f'{metric_name}',
                        'type': 'vector',
                        'vector': {
                            'cmap': 'viridis',
                            'line_weight': 3,
                            'show': True if metric_name == metric_to_show else False
                        }
                    })
            
            # Use the display_multi_layer_on_folium_map utility
            from src.utils import display_multi_layer_on_folium_map
            display_multi_layer_on_folium_map(
                layers=layers,
                output_html_path_str=str(output_html_path),
                map_zoom=12,  # Adjust zoom level as needed
                map_tiles='CartoDB positron'  # Clean background map
            )
            
            logger.info(f"Created multi-layer visualization with {len(layers)} layers: {output_html_path}")
            return str(output_html_path)
            
        except Exception as e:
            logger.error(f"Error creating multi-layer visualization: {e}", exc_info=True)
            return None


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster
    from whitebox import WhiteboxTools
    from src.config import settings


    # Basic setup for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
            # print("Sample preprocessed segments (first 5 rows):"); print(segments_feature.gdf.head())
            expected_metrics = settings.processing.segment_popularity_metrics
            actual_metrics = [m for m in expected_metrics if m in segments_feature.gdf.columns]
            logger.info(f"Expected metrics: {expected_metrics}, Found: {actual_metrics}")
            # if actual_metrics: 
            #     print("\nMetrics sample:")
            #     print(segments_feature.gdf[actual_metrics].head())
        else: 
            logger.error("Segments GDF is None after loading.")

        logger.info("2. Testing Build...")
        if segments_feature.gdf is not None:
            logger.info(f"Testing build with interpolation: {settings.processing.interpolation_method_polylines} (if raster), vector default")
            # Build returns a list of delayed tasks now
            delayed_tasks = segments_feature.build()
            if delayed_tasks:
                logger.info(f"Received {len(delayed_tasks)} delayed tasks from build(). Computing...")
                # Compute the tasks here for the standalone test
                computed_results = dask.compute(*delayed_tasks)
                logger.info("Build computation completed.")
                # Filter out None results if any tasks failed internally
                output_paths = [r for r in computed_results if r is not None]
                if output_paths:
                    logger.info(f"Generated Outputs: {output_paths}")
                    # Create the multi-layer visualization after all tasks are completed
                    if settings.processing.display_segments:
                        logger.info("Creating multi-layer visualization of all segment popularity metrics...")
                        visualization_path = segments_feature.create_multi_layer_visualization(output_paths=output_paths)
                        if visualization_path:
                            logger.info(f"Created multi-layer visualization: {visualization_path}")
                        else:
                            logger.warning("Failed to create multi-layer visualization")
                else:
                    logger.warning("No valid outputs generated after computing build tasks.")
            else:
                logger.warning("Build() returned no tasks.")
        else:
            logger.warning("Skipping build test as data loading failed.")
        logger.info("--- Segments Feature Test Completed ---")
    except Exception as e: 
        logger.error(f"Error during Segments test: {e}", exc_info=True)
    finally:
        if client:
            try: 
                client.retire_workers()
                client.close()
                cluster.shutdown_on_close = True
                cluster.close()
                logger.info("Dask client/cluster closed.")
            except Exception as e: 
                logger.warning(f"Error closing Dask: {e}")
    logger.info("--- Standalone Test Finished ---")
