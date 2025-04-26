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

            # 6. Save Intermediate File (store in 4326 for visualization compatibility)
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

    def _prepare_wbt_input(self, period_gdf, period_name):
        """
        Prepares a shapefile for WBT interpolation input from period GeoDataFrame.
        
        Args:
            period_gdf: GeoDataFrame with station points and 'value' column for the period.
            period_name: Name of the time period (e.g., 'morning').
            
        Returns:
            tuple: (input_shp_path, temp_dir_obj, value_field_shp) or (None, temp_dir_obj, None) on failure.
        """
        input_shp_path = None
        temp_dir_obj = None
        value_field_shp = "value"  # Field containing average volume
        
        try:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"traffic_{period_name}_wbt_")
            input_shp_path = Path(temp_dir_obj.name) / f"traffic_points_{period_name}.shp"
            logger.info(f"Saving {period_name} points to temporary shapefile: {input_shp_path}")
            
            # Convert to EPSG:25833 for WBT processing
            wbt_period_gdf = self._prepare_for_wbt(period_gdf.copy())
            logger.info(f"Prepared points for WBT: converted from {period_gdf.crs} to {wbt_period_gdf.crs}")
            
            # Ensure value field is numeric
            if not pd.api.types.is_numeric_dtype(wbt_period_gdf[value_field_shp]):
                logger.warning(f"'{value_field_shp}' in {period_name} data is not numeric. Converting.")
                wbt_period_gdf.loc[:, value_field_shp] = pd.to_numeric(
                    wbt_period_gdf[value_field_shp], errors="coerce"
                ).fillna(0)
            
            save_vector_data(
                wbt_period_gdf[[value_field_shp, "geometry"]],
                input_shp_path,
                driver="ESRI Shapefile",
            )
            logger.info(f"Saved temporary shapefile for {period_name}: {input_shp_path}")
            
            return input_shp_path, temp_dir_obj, value_field_shp
            
        except Exception as e:
            logger.error(f"Error preparing WBT input for {period_name}: {e}", exc_info=True)
            return None, temp_dir_obj, None
    
    def _get_output_paths(self, period_name):
        """
        Creates output paths for raw and masked rasters.
        
        Args:
            period_name: Name of the time period (e.g., 'morning').
            
        Returns:
            tuple: (raw_output_raster_path_unmasked, masked_output_raster_path) or (None, None) on failure.
        """
        output_filename_attr = f"traffic_density_raster_{period_name}"
        
        try:
            raw_output_raster_path = self._get_output_path(output_filename_attr)
            # Create an intermediate path for the raw IDW output before masking
            raw_output_raster_path_unmasked = raw_output_raster_path.with_name(
                f"{raw_output_raster_path.stem}_unmasked{raw_output_raster_path.suffix}"
            )
            masked_output_raster_path = raw_output_raster_path  # Final masked output path
            masked_output_raster_path.parent.mkdir(parents=True, exist_ok=True)
            
            return raw_output_raster_path_unmasked, masked_output_raster_path, output_filename_attr
        
        except AttributeError:
            logger.error(f"Output filename attribute '{output_filename_attr}' not found in config.output_files.")
            return None, None, None
    
    def _run_idw_interpolation(self, input_shp_path, value_field_shp, output_path):
        """
        Runs WhiteboxTools IDW interpolation.
        
        Args:
            input_shp_path: Path to the input shapefile.
            value_field_shp: Field containing values to interpolate.
            output_path: Path for the output raster.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if self.wbt is None:
            logger.error("WhiteboxTools instance not available. Cannot interpolate.")
            return False
            
        try:
            logger.info("Running WBT IDW interpolation...")
            self.wbt.idw_interpolation(
                i=str(input_shp_path),
                field=value_field_shp,
                output=str(output_path),
                cell_size=self.settings.processing.output_cell_size,
                weight=self.settings.processing.traffic_interpolation_power,
                min_points=20,  # Set a lower default for testing #TODO move to config
            )
            logger.info("WBT IDW interpolation completed using default radius/min_points.")
            return True
            
        except Exception as e:
            logger.error(f"Error during WBT IDW interpolation: {e}", exc_info=True)
            return False
    
    def _mask_raster_with_boundary(self, raster_path, boundary_gs, output_path):
        """
        Masks a raster using a boundary polygon and saves the result.
        
        Args:
            raster_path: Path to the input raster.
            boundary_gs: GeoSeries containing the boundary polygon.
            output_path: Path to save the masked raster.
            
        Returns:
            Path: Path to the masked raster, or None if failed.
        """
        try:
            # 1. First, assign the correct CRS to the WBT output (25833)
            with rasterio.open(raster_path, "r+") as src:
                if src.crs is None or src.crs.to_epsg() != 25833:
                    logger.info(f"Assigning CRS EPSG:25833 to WBT output raster")
                    src.crs = rasterio.CRS.from_epsg(25833)
            
            # 2. Make sure boundary is in the same CRS for masking
            boundary_for_mask = boundary_gs.copy()
            if boundary_for_mask.crs.to_epsg() != 25833:
                logger.info(f"Reprojecting boundary to EPSG:25833 for masking")
                boundary_for_mask = boundary_for_mask.to_crs(epsg=25833)
            
            # 3. Mask the raster with the boundary
            with rasterio.open(raster_path) as src:
                out_image, out_transform = rasterio.mask.mask(
                    src,
                    boundary_for_mask.geometry,
                    crop=True,
                    nodata=src.nodata if src.nodata is not None else -9999,
                    filled=True
                )
                out_meta = src.meta.copy()
            
            # 4. Save the masked raster (still in 25833)
            masked_25833_path = output_path.with_name(f"{output_path.stem}_25833{output_path.suffix}")
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": rasterio.CRS.from_epsg(25833),
                "nodata": out_meta.get('nodata', -9999)
            })
            
            with rasterio.open(masked_25833_path, "w", **out_meta) as dest:
                dest.write(out_image)
            logger.info(f"Saved masked raster in EPSG:25833: {masked_25833_path}")
            
            return masked_25833_path
            
        except Exception as e:
            logger.error(f"Error masking raster: {e}", exc_info=True)
            return None
    
    def _reproject_raster_to_4326(self, masked_25833_path, output_path):
        """
        Reprojects a masked raster from EPSG:25833 to EPSG:4326 for visualization.
        
        Args:
            masked_25833_path: Path to the masked raster in EPSG:25833.
            output_path: Path to save the reprojected raster.
            
        Returns:
            Path: Path to the reprojected raster, or None if failed.
        """
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        
        try:
            with rasterio.open(masked_25833_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, 4326, src.width, src.height, *src.bounds
                )
                
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": height,
                    "width": width,
                    "transform": transform,
                    "crs": rasterio.CRS.from_epsg(4326)
                })
                
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dest, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=rasterio.CRS.from_epsg(4326),
                            resampling=Resampling.bilinear
                        )
            
            logger.info(f"Successfully reprojected masked raster to EPSG:4326: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error reprojecting raster: {e}", exc_info=True)
            return None
    
    def _cleanup_intermediate_files(self, raw_unmasked_path, masked_25833_path):
        """
        Cleans up intermediate files created during processing.
        
        Args:
            raw_unmasked_path: Path to the raw unmasked raster.
            masked_25833_path: Path to the masked raster in EPSG:25833.
        """
        try:
            if raw_unmasked_path and raw_unmasked_path.exists():
                raw_unmasked_path.unlink()
            if masked_25833_path and masked_25833_path.exists():
                masked_25833_path.unlink()
            logger.debug("Removed intermediate files")
        except OSError as unlink_e:
            logger.warning(f"Could not remove intermediate files: {unlink_e}")

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
        input_shp_path, temp_dir_obj, value_field_shp = self._prepare_wbt_input(period_gdf, period_name)
        if input_shp_path is None:
            logger.error(f"Failed to prepare WBT input for {period_name}")
            if temp_dir_obj:
                temp_dir_obj.cleanup()
            return None

        # --- Define Output Raster Paths ---
        raw_output_path, masked_output_path, output_filename_attr = self._get_output_paths(period_name)
        if raw_output_path is None or masked_output_path is None:
            logger.error(f"Failed to create output paths for {period_name}")
            if temp_dir_obj:
                temp_dir_obj.cleanup()
            return None

        try:
            # --- Run WBT IDW Interpolation ---
            idw_success = self._run_idw_interpolation(
                input_shp_path, 
                value_field_shp, 
                raw_output_path
            )
            
            # Clean up temporary directory
            if temp_dir_obj:
                temp_dir_obj.cleanup()
                logger.info(f"Cleaned up temporary directory for {period_name}.")

            if not idw_success:
                logger.error(f"IDW interpolation failed for {period_name}. Cannot mask.")
                return None

            # --- Mask and Reproject Raster ---
            # 1. Mask with boundary
            masked_25833_path = self._mask_raster_with_boundary(
                raw_output_path, 
                boundary_gs, 
                masked_output_path
            )
            
            if masked_25833_path is None:
                logger.error(f"Failed to mask raster for {period_name}")
                return None
                
            # 2. Reproject to EPSG:4326 for visualization
            final_path = self._reproject_raster_to_4326(
                masked_25833_path, 
                masked_output_path
            )
            
            if final_path is None:
                logger.error(f"Failed to reproject raster for {period_name}")
                return None

            # 3. Clean up intermediate files
            self._cleanup_intermediate_files(raw_output_path, masked_25833_path)
            
            # Store and return the final path
            self.output_paths[output_filename_attr] = final_path
            return str(final_path)
            
        except Exception as e:
            logger.error(f"Error post-processing raster for {period_name}: {e}", exc_info=True)
            # Clean up any intermediate files that might exist
            self._cleanup_intermediate_files(raw_output_path, masked_output_path.with_name(f"{masked_output_path.stem}_25833{masked_output_path.suffix}"))
            return None

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
    from src.utils import display_multi_layer_on_folium_map # Import the multi-layer display function

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
                    
                    # Create a multi-layer map with all time periods
                    try:
                        # Create a list of layer configurations for the multi-layer map
                        layers = []
                        period_names = ["morning", "daytime", "evening"]
                        
                        # Match each raster path with its corresponding time period
                        for path_str in raster_paths:
                            path = Path(path_str)
                            # Determine the time period from the filename
                            period = next((period for period in period_names if period in path.stem), None)
                            
                            if period:
                                # Create layer configuration
                                layer = {
                                    'path': path_str,
                                    'name': f"Traffic - {period.capitalize()}", 
                                    'type': 'raster',
                                    'raster': {
                                        'cmap': 'viridis',  # You can use different colormaps for each period
                                        'opacity': 0.7,
                                        'nodata_transparent': True,
                                        'show': True  # All layers visible by default
                                    }
                                }
                                layers.append(layer)
                        
                        if layers:
                            # Create the combined map
                            multi_map_path = settings.paths.output_dir / "traffic_all_periods_map.html"
                            display_multi_layer_on_folium_map(
                                layers=layers,
                                output_html_path_str=str(multi_map_path),
                                map_zoom=12,
                                map_tiles='CartoDB positron'
                            )
                            logger.info(f"Combined map with all time periods saved to: {multi_map_path}")
                        else:
                            logger.warning("No valid layers created for multi-layer map.")
                            
                    except Exception as multi_map_e:
                        logger.error(f"Error creating multi-layer map: {multi_map_e}", exc_info=True)
                        
                        # Fallback: display individual maps if combined map fails
                        logger.info("Falling back to creating individual maps...")
                        for path_str in raster_paths:
                            try:
                                map_output_path = settings.paths.output_dir / f"{Path(path_str).stem}_map.html"
                                display_raster_on_folium_map(
                                    raster_path_str=path_str,
                                    output_html_path_str=str(map_output_path),
                                    target_crs_epsg=settings.processing.output_crs_epsg,
                                    cmap_name='viridis',
                                )
                                logger.info(f"Individual map saved to: {map_output_path}")
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
