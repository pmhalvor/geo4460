import dask
import geopandas as gpd
import json
import logging
import numpy as np
import pandas as pd
import rasterio
import rasterio.crs
import rasterio.mask
import tempfile

from pathlib import Path
from pydantic import BaseModel
from shapely.geometry import Point
from whitebox import WhiteboxTools

from src.tasks.features.feature_base import FeatureBase
from src.utils import (
    save_vector_data,
    load_vector_data,
    display_raster_on_folium_map, # For main block testing
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Traffic(FeatureBase):
    """
    Processes hourly traffic data, joins it with station locations,
    calculates average traffic volume for different time periods (morning,
    daytime, evening), interpolates these averages onto rasters using IDW,
    and masks the rasters to the Oslo boundary.
    """

    def __init__(self, settings: BaseModel, wbt: WhiteboxTools):
        super().__init__(settings, wbt)
        self.gdf = None  # Stores the loaded and merged traffic data points
        self.oslo_boundary = None # Stores the dissolved Oslo boundary geometry

    def _load_station_coords(self) -> pd.DataFrame | None:
        """Loads station coordinates from the stations_all_roads.json file."""
        station_json_path = (
            self.settings.paths.traffic_stations_dir / "stations_all_roads.json"
        ) # TODO move this to config
        logger.info(f"Loading station coordinates from {station_json_path}...")
        if not station_json_path.is_file():
            logger.error(f"Station coordinates file not found: {station_json_path}")
            return None

        try:
            with open(station_json_path, "r", encoding="utf-8") as f:
                station_data = json.load(f)

            stations = []
            # Navigate the JSON structure provided in the feedback
            for point in station_data.get("data", {}).get(
                "trafficRegistrationPoints", []
            ):
                station_id = point.get("id")
                lat = point.get("location", {}).get("coordinates", {}).get("latLon", {}).get("lat")
                lon = point.get("location", {}).get("coordinates", {}).get("latLon", {}).get("lon")
                name = point.get("name")

                if station_id and lat is not None and lon is not None:
                    stations.append(
                        {
                            "station_id": station_id,
                            "name": name,
                            "latitude": lat,
                            "longitude": lon,
                        }
                    )
                else:
                    logger.warning(f"Skipping station due to missing data: {point}")

            if not stations:
                logger.error("No valid station data extracted from JSON.")
                return None

            station_df = pd.DataFrame(stations)
            logger.info(f"Loaded coordinates for {len(station_df)} stations.")
            return station_df

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {station_json_path}", exc_info=True)
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error loading station coordinates: {e}", exc_info=True
            )
            return None

    def load_data(self):
        """
        Loads hourly traffic data from CSV, loads station coordinates from JSON,
        merges them, parses timestamps, and creates a GeoDataFrame.
        """
        logger.info("Loading and processing traffic data...")

        # Define input paths from settings
        # TODO: Make the specific traffic CSV configurable? For now, hardcode the requested one.
        # TODO move to config
        traffic_csv_path = self.settings.paths.data_dir / "traffic" / "911210786_hour_20240501T0000_20240601T0000.csv"

        if not traffic_csv_path.is_file():
            logger.error(f"Traffic CSV file not found: {traffic_csv_path}")
            self.gdf = None
            return

        # 1. Load Station Coordinates
        station_coords_df = self._load_station_coords()
        if station_coords_df is None:
            logger.error("Failed to load station coordinates. Cannot proceed.")
            self.gdf = None
            return

        # 2. Load Traffic CSV Data
        logger.info(f"Loading traffic data from CSV: {traffic_csv_path}")
        try:
            # Specify separator and attempt encoding
            try:
                traffic_df = pd.read_csv(traffic_csv_path, sep=";", encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed, trying latin-1...")
                traffic_df = pd.read_csv(traffic_csv_path, sep=";", encoding="latin-1")

            logger.info(f"Loaded {len(traffic_df)} rows from traffic CSV.")

            # Rename columns for clarity and consistency (adjust based on header)
            # Using names from the header output:
            traffic_df = traffic_df.rename(
                columns={
                    "Trafikkregistreringspunkt": "station_id",
                    "Dato": "date_str",
                    "Fra tidspunkt": "time_str",
                    "Fra": "datetime_str",
                    "Trafikkmengde": "volume",
                    # Add other renames if needed
                }
            )
            # Select relevant columns early
            relevant_cols = ["station_id", "date_str", "datetime_str", "time_str", "volume"]
            missing_cols = [col for col in relevant_cols if col not in traffic_df.columns]
            if missing_cols:
                logger.error(f"Missing expected columns in traffic CSV: {missing_cols}")
                self.gdf = None
                return
            traffic_df = traffic_df[relevant_cols]

            # Convert volume to numeric, coercing errors
            traffic_df["volume"] = pd.to_numeric(traffic_df["volume"], errors="coerce")
            # Handle potential NaNs from coercion or original data
            original_len = len(traffic_df)
            traffic_df = traffic_df.dropna(subset=relevant_cols)
            if len(traffic_df) < original_len:
                logger.warning(f"Dropped {original_len - len(traffic_df)} rows with missing volume/id/date/time/datetime.")

            if traffic_df.empty:
                logger.error("No valid traffic data rows remaining after cleaning.")
                self.gdf = None
                return

            # 3. Merge Traffic Data with Station Coordinates
            logger.info("Merging traffic data with station coordinates...")
            merged_df = pd.merge(
                traffic_df,
                station_coords_df,
                on="station_id",
                how="inner", # Keep only traffic records with matching station coords
            )
            if merged_df.empty:
                logger.error("No matching stations found between traffic data and coordinates file.")
                self.gdf = None
                return
            logger.info(f"{len(merged_df)} traffic records matched with station coordinates.")

            # 4. Parse Timestamps
            logger.info("Parsing timestamps...")
            try:
                logger.info(f"{merged_df['datetime_str'].iloc[0]}")
                # Combine date and time strings, handle potential format variations if needed
                # Format: 2024-05-01T01:00+02:00
                merged_df["timestamp"] = pd.to_datetime(
                    merged_df["datetime_str"], utc=True, format="%Y-%m-%dT%H:%M%z", errors="coerce"
                )
                # Drop rows where timestamp parsing failed
                merged_df = merged_df.dropna(subset=["timestamp"])
                if merged_df.empty:
                    logger.error("No valid timestamps after parsing.")
                    self.gdf = None
                    return
                logger.info("Timestamp parsing successful.")
            except Exception as e:
                logger.error(f"Error parsing timestamps: {e}", exc_info=True)
                self.gdf = None
                return

            # 5. Create GeoDataFrame
            logger.info("Creating GeoDataFrame...")
            geometry = [
                Point(xy) for xy in zip(merged_df["longitude"], merged_df["latitude"])
            ]
            self.gdf = gpd.GeoDataFrame(
                merged_df,
                geometry=geometry,
                crs="EPSG:4326",  # Coordinates are Lat/Lon (WGS84)
            )
            logger.info(f"Initial GeoDataFrame created with {len(self.gdf)} points. CRS: {self.gdf.crs}")

            # 6. Reproject to Target CRS
            self.gdf = self._reproject_if_needed(self.gdf)
            logger.info(f"Reprojected GeoDataFrame CRS: {self.gdf.crs}")

            # 7. Save Intermediate File
            self._save_intermediate_gdf(self.gdf, "prepared_traffic_points_gpkg")
            logger.info("Traffic data loaded, merged, and preprocessed.")

        except FileNotFoundError:
            logger.error(f"File not found during processing: {traffic_csv_path}")
            self.gdf = None
        except Exception as e:
            logger.error(
                f"Unexpected error processing traffic data: {e}", exc_info=True
            )
            self.gdf = None

    def _group_and_average_by_time_period(self) -> dict[str, gpd.GeoDataFrame]:
        """
        Groups the loaded GeoDataFrame by station and time period, calculating
        the average traffic volume for each group.

        Returns:
            A dictionary mapping time period names ('morning', 'daytime', 'evening')
            to GeoDataFrames containing station points and their average volume
            for that period.
        """
        logger.info("Grouping traffic data by time period and calculating averages...")
        if self.gdf is None or self.gdf.empty:
            logger.error("GeoDataFrame is not loaded or is empty. Cannot group.")
            return {}

        # Ensure timestamp column exists
        if "timestamp" not in self.gdf.columns:
            logger.error("Timestamp column missing from GeoDataFrame.")
            return {}

        # Define time periods
        bins = [0, 8, 16, 24]
        labels = ["morning", "daytime", "evening"] # 00-08, 08-16, 16-24
        self.gdf["time_period"] = pd.cut(
            self.gdf["timestamp"].dt.hour,
            bins=bins,
            labels=labels,
            right=False, # [0, 8), [8, 16), [16, 24)
            include_lowest=True,
        )

        # Group by station and time period, calculate mean volume
        # Keep geometry - use first() as geometry should be the same for all points of the same station
        grouped = self.gdf.groupby(["station_id", "time_period"], observed=False).agg(
            avg_volume=("volume", "mean"),
            geometry=("geometry", "first"), # Assumes geometry is consistent per station_id
            # Add other fields if needed, e.g., station name
            name=("name", "first"),
        )
        grouped = grouped.reset_index() # Make station_id and time_period columns again

        # Convert back to GeoDataFrame (grouping might return DataFrame)
        grouped_gdf = gpd.GeoDataFrame(grouped, geometry="geometry", crs=self.gdf.crs)

        if grouped_gdf.empty:
            logger.error("Grouping resulted in an empty GeoDataFrame.")
            return {}

        logger.info(f"Calculated average volumes for {len(grouped_gdf)} station/time period combinations.")

        # Split into separate GDFs per time period
        period_gdfs = {}
        for period in labels:
            period_gdf = grouped_gdf[grouped_gdf["time_period"] == period].copy()
            if not period_gdf.empty:
                logger.info(f"Created GeoDataFrame for '{period}' with {len(period_gdf)} stations.")
                # Rename avg_volume for interpolation clarity if needed, e.g., to 'value'
                period_gdf = period_gdf.rename(columns={"avg_volume": "value"})
                period_gdfs[period] = period_gdf
            else:
                logger.warning(f"No data found for time period: {period}")

        return period_gdfs

    def _load_oslo_boundary(self) -> gpd.GeoSeries | None:
        """Loads and prepares the Oslo boundary polygon."""
        if self.oslo_boundary is not None:
            logger.debug("Using cached Oslo boundary.")
            return self.oslo_boundary

        logger.info("Loading and preparing Oslo boundary...")
        try:
            fgdb_path = self.settings.paths.n50_gdb_path
            boundary_layer_name = self.settings.input_data.n50_land_cover_layer
            oslo_boundary_gdf = load_vector_data(fgdb_path, layer=boundary_layer_name)

            if oslo_boundary_gdf is None or oslo_boundary_gdf.empty:
                logger.error("Failed to load Oslo boundary layer.")
                return None

            target_crs_epsg = self.settings.processing.output_crs_epsg
            if oslo_boundary_gdf.crs.to_epsg() != target_crs_epsg:
                logger.info(f"Reprojecting boundary to EPSG:{target_crs_epsg}")
                oslo_boundary_gdf = oslo_boundary_gdf.to_crs(epsg=target_crs_epsg)

            # Dissolve into a single polygon
            logger.info("Dissolving boundary layer...")
            # Use unary_union which is generally safer for complex/overlapping polygons
            dissolved_boundary = oslo_boundary_gdf.unary_union
            self.oslo_boundary = gpd.GeoSeries([dissolved_boundary], crs=oslo_boundary_gdf.crs)
            logger.info("Oslo boundary loaded and dissolved.")
            return self.oslo_boundary

        except Exception as e:
            logger.error(f"Error loading or processing Oslo boundary: {e}", exc_info=True)
            return None

    @dask.delayed
    def _interpolate_and_mask_raster(self, period_gdf: gpd.GeoDataFrame, period_name: str, boundary_gs: gpd.GeoSeries):
        """
        Interpolates traffic data for a given time period using IDW,
        masks the result with the Oslo boundary, and saves the final raster.

        Args:
            period_gdf: GeoDataFrame with station points and 'value' column for the period.
            period_name: Name of the time period (e.g., 'morning').
            boundary_gs: GeoSeries containing the dissolved Oslo boundary polygon.

        Returns:
            Path to the final masked raster file, or None if failed.
        """
        logger.info(f"--- Processing Time Period: {period_name} ---")
        if period_gdf.empty:
            logger.warning(f"Input GeoDataFrame for '{period_name}' is empty. Skipping.")
            return None
        if self.wbt is None:
            logger.error("WhiteboxTools instance not available. Cannot interpolate.")
            return None
        if boundary_gs is None or boundary_gs.empty:
            logger.error("Oslo boundary not available. Cannot mask.")
            return None

        # --- Prepare WBT Input ---
        input_shp_path = None
        temp_dir_obj = None
        value_field_shp = "value" # Field containing average volume

        try:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"traffic_{period_name}_wbt_")
            input_shp_path = Path(temp_dir_obj.name) / f"traffic_points_{period_name}.shp"
            logger.info(f"Saving {period_name} points to temporary shapefile: {input_shp_path}")

            # Ensure value field is numeric
            if not pd.api.types.is_numeric_dtype(period_gdf[value_field_shp]):
                logger.warning(f"'{value_field_shp}' in {period_name} data is not numeric. Converting.")
                period_gdf.loc[:, value_field_shp] = pd.to_numeric(
                    period_gdf[value_field_shp], errors="coerce"
                ).fillna(0) # Fill NaNs resulting from conversion with 0? Or drop?

            save_vector_data(
                period_gdf[[value_field_shp, "geometry"]],
                input_shp_path,
                driver="ESRI Shapefile",
            )
            logger.info(f"Saved temporary shapefile for {period_name}: {input_shp_path}")

        except Exception as e:
            logger.error(f"Error preparing WBT input for {period_name}: {e}", exc_info=True)
            if temp_dir_obj:
                temp_dir_obj.cleanup()
            return None # Indicate failure

        # --- Define Output Raster Paths ---
        # Use specific filenames from config based on period name
        output_filename_attr = f"traffic_density_raster_{period_name}"
        try:
            raw_output_raster_path = self._get_output_path(output_filename_attr)
            # Create an intermediate path for the raw IDW output before masking
            raw_output_raster_path_unmasked = raw_output_raster_path.with_name(
                f"{raw_output_raster_path.stem}_unmasked{raw_output_raster_path.suffix}"
            )
            masked_output_raster_path = raw_output_raster_path # Final masked output path
            masked_output_raster_path.parent.mkdir(parents=True, exist_ok=True)
        except AttributeError:
             logger.error(f"Output filename attribute '{output_filename_attr}' not found in config.output_files.")
             if temp_dir_obj:
                 temp_dir_obj.cleanup()
             return None


        # --- Run WBT IDW Interpolation ---
        idw_success = False
        try:
            logger.info(f"Running WBT IDW interpolation for {period_name}...")
            # Use general processing settings for cell size, specific for power?
            # Reuse heatmap radius/min_points for now, or add specific traffic ones to config
            self.wbt.idw_interpolation(
                i=str(input_shp_path),
                field=value_field_shp,
                output=str(raw_output_raster_path_unmasked), # Save to unmasked path first
                cell_size=self.settings.processing.output_cell_size, # Use general cell size
                weight=self.settings.processing.traffic_interpolation_power, # Use traffic power
                # Remove radius and min_points to use WBT defaults, as 150 min_points is too high for 81 inputs
                # radius=self.settings.processing.heatmap_idw_radius,
                # min_points=self.settings.processing.heatmap_idw_min_points,
                min_points=20, # Set a lower default for testing
            )
            idw_success = True
            logger.info(f"WBT IDW interpolation completed for {period_name} using default radius/min_points.")

        except Exception as e:
            logger.error(f"Error during WBT IDW interpolation for {period_name}: {e}", exc_info=True)
            # idw_success remains False

        finally:
            # Cleanup temporary directory
            if temp_dir_obj:
                temp_dir_obj.cleanup()
                logger.info(f"Cleaned up temporary directory for {period_name}.")

        # --- Assign CRS, Mask, and Save Final Raster ---
        if idw_success:
            try:
                logger.info(f"Assigning CRS and masking raster for {period_name}...")
                target_crs_epsg = self.settings.processing.output_crs_epsg

                with rasterio.open(raw_output_raster_path_unmasked) as src:
                    # Ensure boundary CRS matches raster CRS (should match target_crs_epsg)
                    if boundary_gs.crs.to_epsg() != target_crs_epsg:
                         logger.warning(f"Reprojecting boundary CRS for masking {period_name}.")
                         boundary_gs = boundary_gs.to_crs(epsg=target_crs_epsg)

                    # Check and assign CRS to the raw raster *before* masking
                    profile = src.profile
                    if profile['crs'] is None or profile['crs'].to_epsg() != target_crs_epsg:
                        logger.warning(f"Assigning EPSG:{target_crs_epsg} to raw {period_name} raster.")
                        profile.update(crs=rasterio.CRS.from_epsg(target_crs_epsg))
                        # Need to write this back temporarily or handle in memory?
                        # Let's try updating profile for the mask operation directly.
                    else:
                         logger.info(f"Raw {period_name} raster CRS is correct ({profile['crs']}).")


                    # Mask the raster
                    out_image, out_transform = rasterio.mask.mask(
                        src,
                        boundary_gs.geometry, # Pass the geometry objects
                        crop=True, # Crop to the extent of the boundary
                        nodata=profile.get('nodata', -9999), # Use existing nodata or set one
                        filled=True # Fill areas outside the mask with nodata
                    )
                    out_meta = src.meta.copy()

                # Update metadata for the masked raster
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "crs": rasterio.CRS.from_epsg(target_crs_epsg), # Ensure final CRS is set
                        "nodata": out_meta.get('nodata', -9999) # Ensure nodata is in meta
                    }
                )

                # Write the masked raster
                with rasterio.open(masked_output_raster_path, "w", **out_meta) as dest:
                    dest.write(out_image)

                logger.info(f"Successfully masked and saved final raster: {masked_output_raster_path}")

                # Clean up the unmasked intermediate file
                try:
                    raw_output_raster_path_unmasked.unlink()
                    logger.debug(f"Removed intermediate unmasked file: {raw_output_raster_path_unmasked}")
                except OSError as unlink_e:
                    logger.warning(f"Could not remove intermediate unmasked file: {unlink_e}")
                # --- End Clean up ---

                # Store and return the final path
                self.output_paths[output_filename_attr] = masked_output_raster_path
                return str(masked_output_raster_path)

            except Exception as mask_e:
                logger.error(f"Error assigning CRS or masking raster for {period_name}: {mask_e}", exc_info=True)
                # Attempt to clean up raw file if it exists
                if raw_output_raster_path_unmasked.exists():
                    try:
                        raw_output_raster_path_unmasked.unlink()
                    except OSError: pass
                return None # Indicate failure
        else:
            logger.error(f"IDW interpolation failed for {period_name}. Cannot mask.")
            return None # Indicate failure

    def build(self):
        """
        Builds the traffic density rasters for morning, daytime, and evening,
        masked to the Oslo boundary.
        """
        # 1. Load and prepare data
        if self.gdf is None:
            self.load_data()
        if self.gdf is None or self.gdf.empty:
            logger.error("Cannot build Traffic features: Data loading failed or resulted in empty GDF.")
            return []

        # 2. Group data by time period
        period_gdfs = self._group_and_average_by_time_period()
        if not period_gdfs:
            logger.error("Cannot build Traffic features: Grouping by time period failed.")
            return []

        # 3. Load Oslo boundary
        oslo_boundary = self._load_oslo_boundary()
        if oslo_boundary is None:
            logger.error("Cannot build Traffic features: Failed to load Oslo boundary.")
            return []

        # 4. Create delayed tasks for interpolation and masking for each period
        tasks = []
        for period_name, period_gdf in period_gdfs.items():
            task = self._interpolate_and_mask_raster(period_gdf, period_name, oslo_boundary)
            tasks.append(task)

        # 5. Return the list of delayed tasks
        if not tasks:
            logger.warning("No valid tasks created for traffic density rasters.")
            return [] # Return empty list if no tasks

        logger.info(f"Returning {len(tasks)} delayed tasks for traffic density raster computation.")
        return tasks


if __name__ == "__main__":
    # Imports needed for standalone testing/execution
    from dask.distributed import Client, LocalCluster
    from src.config import settings
    from whitebox import WhiteboxTools

    logger.info("--- Running traffic.py Standalone Test ---")

    if settings:
        # --- Basic Setup ---
        settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using Output Directory: {settings.paths.output_dir}")

        # Setup Dask client
        wbt = WhiteboxTools()
        # Set verbose mode based on config
        # wbt.verbose = settings.processing.wbt_verbose
        logger.info(f"WhiteboxTools verbose mode: {wbt.verbose}")

        cluster = LocalCluster(n_workers=1, threads_per_worker=1) # Adjust as needed
        client = Client(cluster)
        logger.info(f"Dask client started: {client.dashboard_link}")

        # --- Test Traffic Feature ---
        try:
            logger.info("--- Testing Traffic Feature ---")
            traffic_feature = Traffic(settings, wbt)

            # 1. Test Load Data (implicitly called by build if needed, but can test separately)
            # logger.info("1. Testing Traffic Load Data...")
            # traffic_feature.load_data()
            # if traffic_feature.gdf is not None:
            #     logger.info(f"Traffic points loaded successfully. Shape: {traffic_feature.gdf.shape}")
            # else:
            #     logger.error("Traffic GDF is None after loading.")

            # 2. Test Build (Loads, Groups, Interpolates, Masks)
            logger.info("2. Testing Traffic Build (Density Rasters)...")
            # Build now returns a list of delayed tasks
            delayed_tasks = traffic_feature.build()

            # 3. Compute tasks and Display Rasters on Folium Map
            if delayed_tasks:
                logger.info(f"Received {len(delayed_tasks)} delayed tasks from build(). Computing...")
                # Compute the list of tasks
                computed_results = dask.compute(*delayed_tasks)
                logger.info("Build computation completed.")

                # Filter out None results and process paths
                raster_paths = [r for r in computed_results if r is not None]

                if raster_paths:
                    logger.info("Generated Traffic Density Rasters:")
                    for path_str in raster_paths:
                        logger.info(f"  - {path_str}")
                        try:
                            map_output_path = settings.paths.output_dir / f"{Path(path_str).stem}_map.html"
                            display_raster_on_folium_map(
                                raster_path_str=path_str,
                                output_html_path_str=str(map_output_path),
                                target_crs_epsg=settings.processing.output_crs_epsg,
                                cmap_name='viridis', # Use a suitable colormap for density
                                # layer_name=Path(path_str).stem # Name layer based on filename
                            )
                            logger.info(f"Map saved to: {map_output_path}")
                        except Exception as display_e:
                            logger.error(f"Error displaying raster {path_str} on map: {display_e}", exc_info=True)
                else:
                    logger.warning("No valid raster paths generated after computing build tasks. Skipping map display.")
            else:
                logger.warning("Build() returned no tasks.")

            logger.info("--- Traffic Feature Test Completed ---")

        except Exception as e:
            logger.error(f"Error during Traffic test: {e}", exc_info=True)
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
