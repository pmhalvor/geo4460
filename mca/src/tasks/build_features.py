import logging
import dask
import time
import pandas as pd
import geopandas as gpd
from abc import ABC, abstractmethod
from pathlib import Path
from dask.distributed import Client
from whitebox import WhiteboxTools

# Local imports
from src.config import AppConfig
from src.utils import (
    load_vector_data,
    save_vector_data,
    reproject_gdf,
    polyline_to_points,
    save_raster_data,  # Assuming we might save intermediate rasters
    load_raster_data,  # Added import for CostDistance class
)

logger = logging.getLogger(__name__)

# --- Docstring from original file (for reference) ---
"""
Workflow Steps relevant to this module:

1. Load data: convert data from `data/` to gdf w/ polylines or points
    1. Segments: gdf, get_metric(metric, id="all"), get(id), len, …
    2. Heatmap: gdf, get_activity(id), len, …
    3. Traffic: gdf, get_metric(metric, id="all", vehicle=["all", "bike", "car"]), get(id), len, …
    4. Lanes: gdf, get_classification(id), get(id), len, … # Part of Roads
    5. Elevation: contour lines, get_metric(metric, id="all"), get(id), len, …
    6. Roads: gdf, get_classification()

2. Generate feature layers:
    1. Roads:
        1. Roads polylines (needed for CDW analysis)
        2. Bike lane polylines (w/ lane classification if possible)
        3. Roads w/o bike lanes (final layer to be used downtream)
    2. Segment popularity rasters/lines
        1. Aggregate column data into relevant metrics:
            1. Athletes/age
            2. Stars/age
            3. Stars/athletes
        2. Aggregate metrics over all polylines (average)
        3. Create raster from aggregated metric polylines
    3. Average speed raster (from personal heat map - Strava Activities)
        1. Start w/ activities gdf including polylines w/ speed points
        2. Build speed points layer
        3. Create raster from speed points (doppler shift expected on two way roads up hill)
    4. Traffic buffers (for better segment intersection)
        1. Traffic stations as points w/ flux metrics
        2. Create buffers around traffic stations
        3. Create raster from traffic buffers (Interpolation)
    5. Elevation & slope rasters:
        1. Contour lines as points (from N50)
        3. Create elevation raster (DEM) from contour points
        4. Create slope raster from elevation raster
    6. Cost function raster
        1. Combine slope, speed (optional), road restrictions
"""


# --- Base Class ---
class FeatureBase(ABC):
    """Abstract base class for feature generation."""

    def __init__(self, settings: AppConfig, wbt: WhiteboxTools):
        self.settings = settings
        self.wbt = wbt
        self.gdf: gpd.GeoDataFrame | None = None  # Loaded and preprocessed data
        self.output_paths: dict = {}  # To store paths of generated outputs

    @abstractmethod
    def load_data(self):
        """Load and preprocess data specific to the feature."""
        pass

    @abstractmethod
    def build(self):
        """Build the feature layer(s) (potentially using Dask)."""
        pass

    def _get_output_path(self, key: str) -> Path:
        """Helper to get a full output path from settings."""
        filename = getattr(self.settings.output_files, key)
        return self.settings.paths.output_dir / filename

    def _save_intermediate_gdf(self, gdf: gpd.GeoDataFrame, output_key: str):
        """Saves an intermediate GeoDataFrame."""
        path = self._get_output_path(output_key)
        save_vector_data(gdf, path, driver="GPKG")  # Use GeoPackage for intermediates
        self.output_paths[output_key] = path
        logger.info(f"Saved intermediate vector data: {path}")

    def _save_raster(self, array, profile, output_key: str):
        """Saves a raster file."""
        path = self._get_output_path(output_key)
        save_raster_data(array, profile, path)
        self.output_paths[output_key] = path
        logger.info(f"Saved raster data: {path}")

    def _reproject_if_needed(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Reprojects GeoDataFrame to target CRS if not already matching."""
        target_crs = f"EPSG:{self.settings.processing.output_crs_epsg}"
        if gdf.crs is None:
            logger.warning("Input GDF has no CRS, assuming EPSG:4326 for reprojection.")
            gdf.crs = "EPSG:4326"  # Common default, adjust if needed
        if gdf.crs != target_crs:
            return reproject_gdf(gdf, target_crs)
        return gdf


# --- Feature Subclasses ---


class Segments(FeatureBase):
    """Handles Strava segment data processing."""

    def load_data(self):
        logger.info("Loading Strava segments...")
        self.gdf = load_vector_data(self.settings.paths.strava_segments_geojson)
        self.gdf = self._reproject_if_needed(self.gdf)
        # TODO: Add preprocessing steps (calculate age, metrics)
        # Example: Calculate age in days
        # self.gdf[self.settings.input_data.segment_created_at_field] = pd.to_datetime(self.gdf[self.settings.input_data.segment_created_at_field])
        # self.gdf['age_days'] = (pd.Timestamp.now(tz='UTC') - self.gdf[self.settings.input_data.segment_created_at_field]).dt.days
        # Example: Calculate metrics
        # for metric in self.settings.processing.segment_popularity_metrics:
        #     if metric == "athletes_per_day":
        #         # self.gdf[metric] = self.gdf[self.settings.input_data.segment_athlete_count_field] / self.gdf['age_days']
        #         pass # Implement calculation
        #     # ... other metrics
        logger.info("Strava segments loaded and reprojected.")
        self._save_intermediate_gdf(self.gdf, "prepared_segments_gpkg")

    @dask.delayed
    def _build_popularity_raster(self, metric: str):
        """Builds a popularity raster for a single metric."""
        logger.info(f"Building popularity raster for metric: {metric}")
        if self.gdf is None:
            raise ValueError("Segments data not loaded.")

        # 1. Ensure metric column exists (needs preprocessing in load_data)
        if metric not in self.gdf.columns:
            raise ValueError(
                f"Metric '{metric}' not found in segments GDF. Ensure preprocessing is done."
            )

        # 2. Convert polylines to points
        points_gdf = polyline_to_points(self.gdf)

        # 3. Interpolate points to raster using WBT
        input_shp_path = (
            self.settings.paths.output_dir / f"temp_segment_points_{metric}.shp"
        )
        output_raster_path = self._get_output_path("segment_popularity_raster_prefix")
        output_raster_path = (
            output_raster_path.parent / f"{output_raster_path.stem}_{metric}.tif"
        )

        # Save points to temporary Shapefile for WBT
        # WBT tools often prefer Shapefiles
        save_vector_data(points_gdf, input_shp_path, driver="ESRI Shapefile")

        try:
            # TODO: Choose appropriate WBT interpolation tool (IDW, Kriging, etc.)
            # Using IDW as an example
            self.wbt.idw_interpolation(
                i=str(input_shp_path),
                field=metric,
                output=str(output_raster_path),
                weight=self.settings.processing.traffic_interpolation_power,  # Reuse traffic power?
                radius=self.settings.processing.traffic_buffer_distance,  # Reuse traffic buffer?
                cell_size=self.settings.processing.output_cell_size,
                # base= Optional base raster path
            )
            logger.info(f"Generated popularity raster: {output_raster_path}")
            self.output_paths[f"popularity_raster_{metric}"] = output_raster_path
        except Exception as e:
            logger.error(f"Error during WBT interpolation for {metric}: {e}")
            # Optionally re-raise or return an error indicator
            return None
        finally:
            # Clean up temporary shapefile components
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    temp_file.unlink()

        return str(output_raster_path)  # Return path to the generated raster

    def build(self):
        """Builds popularity rasters for all configured metrics."""
        if self.gdf is None:
            self.load_data()

        tasks = []
        for metric in self.settings.processing.segment_popularity_metrics:
            tasks.append(self._build_popularity_raster(metric))

        # Compute all raster tasks in parallel
        logger.info(f"Computing {len(tasks)} popularity raster tasks...")
        results = dask.compute(*tasks)
        logger.info("Popularity raster computation finished.")
        # Filter out None results (errors during WBT)
        successful_rasters = [r for r in results if r is not None]
        return successful_rasters  # Return list of paths to generated rasters


class Heatmap(FeatureBase):
    """Handles Strava activity heatmap data processing."""

    def load_data(self):
        logger.warning("Heatmap (Strava Activity) loading not implemented yet.")
        # TODO: Implement loading of GPX/TCX files from settings.paths.strava_activities_dir
        # Needs libraries like gpxpy or similar
        # Combine into a single GeoDataFrame with speed, time, elevation attributes
        # self.gdf = ...
        # self.gdf = self._reproject_if_needed(self.gdf)
        # self._save_intermediate_gdf(self.gdf, "prepared_activities_gpkg")
        self.gdf = None  # Placeholder

    @dask.delayed
    def _build_average_speed_raster(self):
        logger.info("Building average speed raster...")
        if self.gdf is None:
            logger.warning("Heatmap data not loaded, skipping speed raster generation.")
            return None

        # TODO: Implement logic similar to Segments._build_popularity_raster
        # 1. Convert activity lines/points to points GDF with speed attribute
        #    (May need aggregation if multiple activities overlap a cell)
        # 2. Save points to temporary Shapefile
        # 3. Use WBT interpolation (e.g., IDW) with speed field
        # 4. Save raster using _save_raster
        # 5. Clean up temp files
        output_raster_path = self._get_output_path("average_speed_raster")
        logger.warning("Average speed raster generation logic not implemented.")
        return str(output_raster_path)  # Placeholder return

    def build(self):
        if self.gdf is None:
            self.load_data()
        if self.gdf is None:  # Still None if loading failed/not implemented
            return None

        task = self._build_average_speed_raster()
        result = dask.compute(task)[0]  # Compute the single task
        return result


class Roads(FeatureBase):
    """Handles N50 road and bike lane data."""

    def load_data(self):
        logger.info("Loading N50 road and bike lane data...")
        if (
            not self.settings.paths.n50_gdb_path
            or not self.settings.paths.n50_gdb_path.exists()
        ):
            logger.warning(
                "N50 GDB path not configured or not found. Skipping Roads loading."
            )
            self.gdf_roads = None
            self.gdf_lanes = None
            return

        try:
            # Load roads
            self.gdf_roads = load_vector_data(
                self.settings.paths.n50_gdb_path,
                layer=self.settings.input_data.n50_roads_layer,
            )
            self.gdf_roads = self._reproject_if_needed(self.gdf_roads)
            self._save_intermediate_gdf(self.gdf_roads, "prepared_roads_gpkg")

            # Load bike lanes
            self.gdf_lanes = load_vector_data(
                self.settings.paths.n50_gdb_path,
                layer=self.settings.input_data.n50_bike_lanes_layer,
            )
            self.gdf_lanes = self._reproject_if_needed(self.gdf_lanes)
            self._save_intermediate_gdf(self.gdf_lanes, "prepared_bike_lanes_gpkg")

            # Calculate roads without lanes
            # TODO: Refine this logic - spatial difference might be complex/slow
            # Consider attribute-based filtering first if possible, or buffering lanes
            logger.info("Calculating roads without bike lanes...")
            if self.gdf_roads is not None and self.gdf_lanes is not None:
                # Ensure consistent geometry types if needed before difference
                # Buffer lanes slightly for more robust difference?
                buffered_lanes = self.gdf_lanes.buffer(0.1)  # Small buffer
                gdf_roads_no_lanes = gpd.overlay(
                    self.gdf_roads,
                    gpd.GeoDataFrame(geometry=buffered_lanes, crs=self.gdf_lanes.crs),
                    how="difference",
                )
                self._save_intermediate_gdf(
                    gdf_roads_no_lanes, "prepared_roads_no_lanes_gpkg"
                )
                self.output_paths["roads_no_lanes"] = self._get_output_path(
                    "prepared_roads_no_lanes_gpkg"
                )
            else:
                logger.warning(
                    "Could not calculate roads without lanes due to missing inputs."
                )

            logger.info("N50 Roads/Lanes loaded and preprocessed.")

        except Exception as e:
            logger.error(f"Error loading N50 data: {e}", exc_info=True)
            self.gdf_roads = None
            self.gdf_lanes = None

    def build(self):
        """Build tasks related to roads (e.g., rasterization if needed)."""
        if getattr(self, "gdf_roads", None) is None:  # Check if loaded
            self.load_data()

        # Currently, load_data saves the prepared vector files.
        # This build step could rasterize them if needed for overlays.
        # Example: Rasterize roads_no_lanes
        roads_no_lanes_path = self.output_paths.get("roads_no_lanes")
        if roads_no_lanes_path and roads_no_lanes_path.exists():
            # TODO: Implement rasterization using WBT vector_lines_to_raster
            logger.warning("Road rasterization not implemented.")
            pass
        else:
            logger.warning("Prepared roads_no_lanes file not found, cannot rasterize.")

        # Return paths to the *vector* files generated in load_data for now
        return {
            "roads": self.output_paths.get("prepared_roads_gpkg"),
            "lanes": self.output_paths.get("prepared_bike_lanes_gpkg"),
            "roads_no_lanes": roads_no_lanes_path,
        }


class Traffic(FeatureBase):
    """Handles traffic count data."""

    def load_data(self):
        logger.info("Loading traffic station data...")
        # Load station locations (assuming JSON files in traffic_stations_dir)
        station_files = list(
            self.settings.paths.traffic_stations_dir.glob("stations_*.json")
        )
        if not station_files:
            logger.warning(
                "No traffic station JSON files found. Skipping traffic loading."
            )
            self.gdf = None
            return

        all_stations = []
        for f in station_files:
            try:
                station_gdf = gpd.read_file(f)
                # TODO: Extract relevant fields (id, name, geometry) based on JSON structure
                # Assuming GeoJSON format for simplicity
                all_stations.append(station_gdf)
            except Exception as e:
                logger.warning(f"Could not load or parse station file {f}: {e}")

        if not all_stations:
            logger.warning("Failed to load any station data.")
            self.gdf = None
            return

        stations_gdf = pd.concat(all_stations, ignore_index=True)
        stations_gdf = self._reproject_if_needed(stations_gdf)

        # Load traffic counts CSV
        try:
            counts_df = pd.read_csv(self.settings.paths.traffic_bikes_csv)
            # TODO: Process counts_df (aggregate per station, filter dates, etc.)
            # Example: Aggregate total bike volume per station ID
            # station_agg_counts = counts_df.groupby(self.settings.input_data.traffic_station_id_field)[self.settings.input_data.traffic_bike_volume_field].sum().reset_index()

            # Merge counts with station locations
            # self.gdf = pd.merge(stations_gdf, station_agg_counts, on=self.settings.input_data.traffic_station_id_field, how='left')
            # self.gdf[self.settings.input_data.traffic_bike_volume_field].fillna(0, inplace=True) # Handle stations with no counts

            # Placeholder until merge logic is defined
            self.gdf = stations_gdf
            logger.warning(
                "Traffic count merging and aggregation not fully implemented."
            )

            self._save_intermediate_gdf(self.gdf, "prepared_traffic_points_gpkg")
            logger.info("Traffic data loaded and preprocessed.")

        except FileNotFoundError:
            logger.error(
                f"Traffic counts CSV not found: {self.settings.paths.traffic_bikes_csv}"
            )
            self.gdf = None
        except Exception as e:
            logger.error(f"Error processing traffic data: {e}", exc_info=True)
            self.gdf = None

    @dask.delayed
    def _build_traffic_raster(self):
        """Interpolates traffic points to create a density raster."""
        logger.info("Building traffic density raster...")
        if (
            self.gdf is None
            or self.settings.input_data.traffic_bike_volume_field
            not in self.gdf.columns
        ):
            logger.warning(
                "Traffic data/volume field not available, skipping raster generation."
            )
            return None

        input_shp_path = self.settings.paths.output_dir / "temp_traffic_points.shp"
        output_raster_path = self._get_output_path("traffic_density_raster")

        # Save points to temporary Shapefile for WBT
        save_vector_data(self.gdf, input_shp_path, driver="ESRI Shapefile")

        try:
            # Use WBT IDW interpolation
            self.wbt.idw_interpolation(
                i=str(input_shp_path),
                field=self.settings.input_data.traffic_bike_volume_field,
                output=str(output_raster_path),
                weight=self.settings.processing.traffic_interpolation_power,
                radius=self.settings.processing.traffic_buffer_distance,  # Use buffer distance as search radius
                cell_size=self.settings.processing.output_cell_size,
            )
            logger.info(f"Generated traffic density raster: {output_raster_path}")
            self.output_paths["traffic_density_raster"] = output_raster_path
        except Exception as e:
            logger.error(f"Error during WBT interpolation for traffic: {e}")
            return None
        finally:
            # Clean up temporary shapefile components
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    temp_file.unlink()

        return str(output_raster_path)

    def build(self):
        if self.gdf is None:
            self.load_data()
        if self.gdf is None:
            return None

        task = self._build_traffic_raster()
        result = dask.compute(task)[0]
        return result


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
            self.gdf = self._reproject_if_needed(self.gdf)
            # TODO: Ensure elevation field exists and has correct name/type
            # elevation_field = self.settings.input_layers.contour_elevation_field # From GIS5 config
            # Need similar field in MCA config
            logger.warning("Elevation field name/validation not implemented.")

            self._save_intermediate_gdf(self.gdf, "prepared_contours_gpkg")
            logger.info("N50 Contours loaded and preprocessed.")
        except Exception as e:
            logger.error(f"Error loading N50 contour data: {e}", exc_info=True)
            self.gdf = None

    @dask.delayed
    def _build_dem_and_slope(self):
        """Generates DEM and Slope rasters from contours."""
        logger.info("Building DEM and Slope rasters...")
        if self.gdf is None:
            logger.warning("Contour data not available, skipping DEM/Slope generation.")
            return None, None

        contour_shp_path = self.settings.paths.output_dir / "temp_contours.shp"
        dem_path = self._get_output_path("elevation_dem_raster")
        slope_path = self._get_output_path("slope_raster")

        # Save contours to temporary Shapefile
        save_vector_data(self.gdf, contour_shp_path, driver="ESRI Shapefile")

        dem_generated = False
        try:
            # Interpolate contours to DEM using WBT
            # TODO: Choose appropriate tool (e.g., TopoToRaster variants if available/licensed, or simpler interpolation)
            # Using Natural Neighbor as an example - requires points
            logger.info("Converting contours to points for interpolation...")
            points_gdf = polyline_to_points(self.gdf)
            points_shp_path = self.settings.paths.output_dir / "temp_contour_points.shp"
            save_vector_data(points_gdf, points_shp_path, driver="ESRI Shapefile")

            # TODO: Get elevation field name from settings
            elevation_field = "hoyde"  # Placeholder from GIS5 config
            logger.warning(f"Using placeholder elevation field: {elevation_field}")

            self.wbt.natural_neighbor_interpolation(
                i=str(points_shp_path),
                field=elevation_field,
                output=str(dem_path),
                cell_size=self.settings.processing.output_cell_size,
            )
            logger.info(f"Generated DEM raster: {dem_path}")
            self.output_paths["elevation_dem_raster"] = dem_path
            dem_generated = True

            # Calculate Slope from DEM
            self.wbt.slope(
                dem=str(dem_path),
                output=str(slope_path),
                zfactor=1.0,
                units=self.settings.processing.slope_units,
            )
            logger.info(f"Generated Slope raster: {slope_path}")
            self.output_paths["slope_raster"] = slope_path

        except Exception as e:
            logger.error(f"Error during WBT DEM/Slope generation: {e}")
            return None, None  # Return None for both if error occurs
        finally:
            # Clean up temporary shapefiles
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_c = contour_shp_path.with_suffix(suffix)
                temp_p = points_shp_path.with_suffix(suffix)
                if temp_c.exists():
                    temp_c.unlink()
                if temp_p.exists():
                    temp_p.unlink()

        return str(dem_path) if dem_generated else None, (
            str(slope_path) if dem_generated else None
        )

    def build(self):
        if self.gdf is None:
            self.load_data()
        if self.gdf is None:
            return None, None

        task = self._build_dem_and_slope()
        dem_result, slope_result = dask.compute(task)[0]
        return dem_result, slope_result


class CostDistance(FeatureBase):
    """Calculates a cost surface based on slope, speed, and road restrictions."""

    # This class needs access to the outputs of other features (Slope, Roads, maybe Heatmap)
    # It might be better structured as a function called *after* other features are built,
    # or it needs references to the other feature objects.

    def __init__(
        self,
        settings: AppConfig,
        wbt: WhiteboxTools,
        slope_raster_path: Path,
        roads_raster_path: Path = None,
        speed_raster_path: Path = None,
    ):
        super().__init__(settings, wbt)
        self.slope_raster_path = slope_raster_path
        self.roads_raster_path = roads_raster_path  # Optional rasterized roads
        self.speed_raster_path = speed_raster_path  # Optional speed raster

    def load_data(self):
        # No specific data loading here, relies on inputs passed to __init__
        logger.info("CostDistance initialized with input raster paths.")
        if not self.slope_raster_path or not self.slope_raster_path.exists():
            raise FileNotFoundError("Slope raster path is required for CostDistance.")

    @dask.delayed
    def _build_cost_raster(self):
        """Builds the cost raster by combining inputs."""
        logger.info("Building cost function raster...")
        output_path = self._get_output_path("cost_function_raster")

        # TODO: Implement cost calculation using WBT raster calculator or Python logic
        # Example logic (needs refinement and actual WBT calls):
        # 1. Load slope raster data (use utils.load_raster_data)
        # 2. Apply slope weight: cost = slope * settings.processing.cost_slope_weight
        # 3. If speed raster exists:
        #    - Load speed raster data
        #    - Apply speed weight: cost = cost + (speed * settings.processing.cost_speed_weight)
        # 4. If roads raster exists (as restriction):
        #    - Load roads raster (where roads=1, non-roads=0 or nodata)
        #    - Set cost to a very high value or nodata where roads raster is not 1
        # 5. Save the final cost raster using _save_raster

        logger.warning("Cost raster generation logic not implemented.")
        # Placeholder: Copy slope raster as cost raster for now
        if self.slope_raster_path:
            try:
                slope_data, profile = load_raster_data(self.slope_raster_path)
                self._save_raster(slope_data, profile, "cost_function_raster")
                logger.info(f"Placeholder: Copied slope raster to {output_path}")
                return str(output_path)
            except Exception as e:
                logger.error(f"Failed to create placeholder cost raster: {e}")
                return None
        return None

    def build(self):
        self.load_data()  # Basic check
        task = self._build_cost_raster()
        result = dask.compute(task)[0]
        return result


# --- Main Task Function ---


def build_features_task(settings: AppConfig, wbt: WhiteboxTools):
    """
    Main task function to build all features.

    Args:
        settings (AppConfig): Application configuration object.
        wbt (WhiteboxTools): Initialized WhiteboxTools object.

    Returns:
        dict: A dictionary containing the generated feature objects or result paths.
              Keys might include 'segments', 'heatmap', 'traffic', 'roads', 'elevation', 'cost_distance'.
    """
    logger.info("--- Start Feature Building Task ---")

    # Initialize Dask client (consider managing client lifecycle outside this function if calling multiple tasks)
    # client = Client() # Removed - manage client in workflow.py if needed across tasks

    # Initialize Feature Objects
    segments = Segments(settings, wbt)
    heatmap = Heatmap(settings, wbt)
    traffic = Traffic(settings, wbt)
    roads = Roads(settings, wbt)
    elevation = Elevation(settings, wbt)

    # Build features - results contain paths or data needed for subsequent steps
    # Run load_data explicitly first (or ensure build calls it)
    logger.info("Loading data for all features...")
    segments.load_data()
    heatmap.load_data()
    traffic.load_data()
    roads.load_data()
    elevation.load_data()
    logger.info("Data loading complete.")

    # Build features using Dask where applicable
    logger.info("Building feature layers...")
    segment_raster_paths = segments.build()  # Returns list of paths
    speed_raster_path = heatmap.build()  # Returns path or None
    traffic_raster_path = traffic.build()  # Returns path or None
    road_vector_paths = roads.build()  # Returns dict of paths
    dem_path, slope_path = elevation.build()  # Returns two paths or None

    # Cost distance depends on slope, potentially roads/speed
    cost_distance = None
    if slope_path:
        # TODO: Decide if roads/speed rasters are needed/generated for cost function
        cost_distance = CostDistance(settings, wbt, slope_raster_path=Path(slope_path))
        cost_raster_path = cost_distance.build()
    else:
        logger.warning(
            "Skipping Cost Distance calculation as slope raster was not generated."
        )
        cost_raster_path = None

    # Consolidate results
    results = {
        "segments": segments,  # Keep object for potential later use
        "segment_rasters": segment_raster_paths,
        "heatmap": heatmap,
        "speed_raster": speed_raster_path,
        "traffic": traffic,
        "traffic_raster": traffic_raster_path,
        "roads": roads,
        "road_vectors": road_vector_paths,
        "elevation": elevation,
        "dem_raster": dem_path,
        "slope_raster": slope_path,
        "cost_distance": cost_distance,
        "cost_raster": cost_raster_path,
        # Add paths to prepared vector data as well
        "prepared_segments": segments.output_paths.get("prepared_segments_gpkg"),
        "prepared_activities": heatmap.output_paths.get("prepared_activities_gpkg"),
        "prepared_traffic": traffic.output_paths.get("prepared_traffic_points_gpkg"),
        "prepared_roads": roads.output_paths.get("prepared_roads_gpkg"),
        "prepared_lanes": roads.output_paths.get("prepared_bike_lanes_gpkg"),
        "prepared_roads_no_lanes": roads.output_paths.get(
            "prepared_roads_no_lanes_gpkg"
        ),
        "prepared_contours": elevation.output_paths.get("prepared_contours_gpkg"),
    }

    # client.close() # Removed - manage client in workflow.py

    logger.info("--- Feature Building Task Completed ---")
    # Log generated paths for clarity
    logger.info("Generated outputs:")
    for key, value in results.items():
        if isinstance(value, Path) or isinstance(value, str) and Path(value).exists():
            logger.info(f"  - {key}: {value}")
        elif isinstance(value, list) and value:
            logger.info(f"  - {key}: {value}")  # Log list of paths
        elif isinstance(value, dict) and value:
            logger.info(f"  - {key}: {value}")  # Log dict of paths

    return results
