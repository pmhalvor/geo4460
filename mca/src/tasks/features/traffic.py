import logging
import dask
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)  # Define logger

# Local imports (adjust relative paths as needed)
try:
    from .feature_base import FeatureBase
    from src.config import AppConfig
    from src.utils import save_vector_data  # WBT interpolation needs shapefile
except ImportError:
    logger.warning(
        "Could not import from src.* or .feature_base directly, attempting relative import..."
    )
    from feature_base import FeatureBase
    from ...config import AppConfig
    from ...utils import save_vector_data

logger = logging.getLogger(__name__)


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
                # Assuming GeoJSON format for station files
                station_gdf = gpd.read_file(f)
                # TODO: Extract relevant fields if needed (id, name, geometry)
                all_stations.append(station_gdf)
            except Exception as e:
                logger.warning(f"Could not load or parse station file {f}: {e}")

        if not all_stations:
            logger.warning("Failed to load any station data.")
            self.gdf = None
            return

        stations_gdf = pd.concat(all_stations, ignore_index=True)
        # Ensure geometry column exists and drop rows without geometry
        if "geometry" not in stations_gdf.columns:
            logger.error("No 'geometry' column found in concatenated station data.")
            self.gdf = None
            return
        stations_gdf = stations_gdf.dropna(subset=["geometry"])
        stations_gdf = stations_gdf[~stations_gdf.geometry.is_empty]

        if stations_gdf.empty:
            logger.warning("No valid station geometries found after loading.")
            self.gdf = None
            return

        # Set CRS if missing (assume WGS84 for GeoJSON)
        if stations_gdf.crs is None:
            logger.warning("Station GeoJSON has no CRS defined, assuming EPSG:4326.")
            stations_gdf.crs = "EPSG:4326"

        stations_gdf = self._reproject_if_needed(stations_gdf)

        # Load traffic counts CSV
        try:
            counts_df = pd.read_csv(self.settings.paths.traffic_bikes_csv)
            # TODO: Process counts_df (aggregate per station, filter dates, etc.)
            # Example: Aggregate total bike volume per station ID
            station_id_field = self.settings.input_data.traffic_station_id_field
            bike_vol_field = self.settings.input_data.traffic_bike_volume_field

            if (
                station_id_field not in counts_df.columns
                or bike_vol_field not in counts_df.columns
            ):
                logger.warning(
                    f"Required columns ('{station_id_field}', '{bike_vol_field}') not found in traffic CSV. Skipping merge."
                )
                self.gdf = stations_gdf  # Keep only station locations
            else:
                # Ensure ID types match for merging
                if station_id_field not in stations_gdf.columns:
                    logger.warning(
                        f"Station ID field '{station_id_field}' not found in station GeoJSONs. Cannot merge counts."
                    )
                    self.gdf = stations_gdf
                else:
                    try:
                        stations_gdf[station_id_field] = stations_gdf[
                            station_id_field
                        ].astype(counts_df[station_id_field].dtype)
                    except Exception as e:
                        logger.warning(
                            f"Could not align ID types for traffic merge: {e}. Skipping merge."
                        )
                        self.gdf = stations_gdf
                    else:
                        # Perform aggregation (Example: sum)
                        logger.info(
                            f"Aggregating traffic counts by '{station_id_field}'..."
                        )
                        station_agg_counts = (
                            counts_df.groupby(station_id_field)[bike_vol_field]
                            .sum()
                            .reset_index()
                        )

                        # Merge counts with station locations
                        logger.info("Merging traffic counts with station locations...")
                        self.gdf = pd.merge(
                            stations_gdf,
                            station_agg_counts,
                            on=station_id_field,
                            how="left",
                        )
                        self.gdf[bike_vol_field].fillna(
                            0, inplace=True
                        )  # Handle stations with no counts
                        logger.info("Traffic count merging complete.")

            self._save_intermediate_gdf(self.gdf, "prepared_traffic_points_gpkg")
            logger.info("Traffic data loaded and preprocessed.")

        except FileNotFoundError:
            logger.error(
                f"Traffic counts CSV not found: {self.settings.paths.traffic_bikes_csv}. Proceeding with station locations only."
            )
            self.gdf = stations_gdf  # Keep station locations even if counts are missing
            # Add placeholder column if merge didn't happen
            if (
                self.settings.input_data.traffic_bike_volume_field
                not in self.gdf.columns
            ):
                self.gdf[self.settings.input_data.traffic_bike_volume_field] = 0
            self._save_intermediate_gdf(self.gdf, "prepared_traffic_points_gpkg")

        except Exception as e:
            logger.error(f"Error processing traffic data: {e}", exc_info=True)
            self.gdf = None

    @dask.delayed
    def _build_traffic_raster(self):
        """Interpolates traffic points to create a density raster."""
        logger.info("Building traffic density raster...")
        bike_vol_field = self.settings.input_data.traffic_bike_volume_field
        if (
            self.gdf is None
            or bike_vol_field not in self.gdf.columns
            or self.gdf[bike_vol_field].isna().all()  # Check if all values are NaN
        ):
            logger.warning(
                f"Traffic data/volume field ('{bike_vol_field}') not available or all NaN, skipping raster generation."
            )
            return None

        # Drop rows with NaN values in the interpolation field
        gdf_filtered = self.gdf.dropna(subset=[bike_vol_field])
        if gdf_filtered.empty:
            logger.warning(
                f"No valid data points remain after dropping NaN in '{bike_vol_field}'. Skipping interpolation."
            )
            return None

        input_shp_path = self.settings.paths.output_dir / "temp_traffic_points.shp"
        output_raster_path = self._get_output_path("traffic_density_raster")

        # Save points to temporary Shapefile for WBT
        # Sanitize field name for Shapefile
        sanitized_field = "".join(filter(str.isalnum, bike_vol_field))[:10]
        if not sanitized_field:
            sanitized_field = "value"
        gdf_shp = gdf_filtered.rename(columns={bike_vol_field: sanitized_field})
        save_vector_data(gdf_shp, input_shp_path, driver="ESRI Shapefile")

        try:
            # Use WBT IDW interpolation
            logger.info(
                f"Running IDW interpolation for traffic (field: '{sanitized_field}')..."
            )
            self.wbt.idw_interpolation(
                i=str(input_shp_path),
                field=sanitized_field,
                output=str(output_raster_path),
                weight=self.settings.processing.traffic_interpolation_power,
                radius=self.settings.processing.traffic_buffer_distance,  # Use buffer distance as search radius
                cell_size=self.settings.processing.output_cell_size,
            )
            logger.info(f"Generated traffic density raster: {output_raster_path}")
            self.output_paths["traffic_density_raster"] = output_raster_path
            return str(output_raster_path)
        except Exception as e:
            logger.error(f"Error during WBT interpolation for traffic: {e}")
            return None
        finally:
            # Clean up temporary shapefile components
            for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                temp_file = input_shp_path.with_suffix(suffix)
                if temp_file.exists():
                    temp_file.unlink()

    def build(self):
        """Builds the traffic density raster."""
        if self.gdf is None:
            self.load_data()
        if self.gdf is None:
            logger.error("Cannot build Traffic features: Data loading failed.")
            return None

        task = self._build_traffic_raster()
        result = dask.compute(task)[0]
        return result
