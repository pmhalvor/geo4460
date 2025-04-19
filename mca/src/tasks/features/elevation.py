import logging
import dask
import geopandas as gpd
import pandas as pd  # Import pandas

logger = logging.getLogger(__name__)  # Define logger

# Local imports (adjust relative paths as needed)
try:
    from .feature_base import FeatureBase
    from src.config import AppConfig
    from src.utils import load_vector_data, polyline_to_points, save_vector_data
except ImportError:
    logger.warning(
        "Could not import from src.* or .feature_base directly, attempting relative import..."
    )
    from feature_base import FeatureBase
    from ...config import AppConfig
    from ...utils import load_vector_data, polyline_to_points, save_vector_data

logger = logging.getLogger(__name__)


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
            # Ensure geometry exists and is valid
            if "geometry" not in self.gdf.columns:
                logger.error("No 'geometry' column found in contour data.")
                self.gdf = None
                return
            self.gdf = self.gdf.dropna(subset=["geometry"])
            self.gdf = self.gdf[~self.gdf.geometry.is_empty]
            if self.gdf.empty:
                logger.warning("No valid contour geometries found after loading.")
                self.gdf = None
                return

            # TODO: Ensure elevation field exists and has correct name/type
            # Need to add 'n50_contour_elevation_field' to InputDataConfig
            elevation_field = getattr(
                self.settings.input_data, "n50_contour_elevation_field", None
            )
            if not elevation_field or elevation_field not in self.gdf.columns:
                # Attempt to find a likely candidate if not configured or missing
                potential_fields = [
                    "hoyde",
                    "HOYDE",
                    "CONTOUR",
                    "Z_Value",
                    "Elevation",
                    "ELEV",
                ]
                found_field = None
                for field in potential_fields:
                    if field in self.gdf.columns:
                        found_field = field
                        logger.warning(
                            f"Elevation field not configured or found ('{elevation_field}'). Using fallback field: '{found_field}'"
                        )
                        break
                if not found_field:
                    logger.error(
                        f"Could not find a suitable elevation field in contour data. Checked: {potential_fields}"
                    )
                    self.gdf = None
                    return
                self.elevation_field_name = found_field  # Store the found field name
            else:
                self.elevation_field_name = (
                    elevation_field  # Store the configured field name
                )

            # Ensure elevation field is numeric
            self.gdf[self.elevation_field_name] = pd.to_numeric(
                self.gdf[self.elevation_field_name], errors="coerce"
            )
            original_len = len(self.gdf)
            self.gdf = self.gdf.dropna(subset=[self.elevation_field_name])
            if len(self.gdf) < original_len:
                logger.warning(
                    f"Dropped {original_len - len(self.gdf)} contours due to non-numeric elevation values."
                )

            if self.gdf.empty:
                logger.error("No valid contours remaining after elevation validation.")
                self.gdf = None
                return

            self.gdf = self._reproject_if_needed(self.gdf)
            self._save_intermediate_gdf(self.gdf, "prepared_contours_gpkg")
            logger.info("N50 Contours loaded and preprocessed.")
        except Exception as e:
            logger.error(f"Error loading N50 contour data: {e}", exc_info=True)
            self.gdf = None

    @dask.delayed
    def _build_dem_and_slope(self):
        """Generates DEM and Slope rasters from contours."""
        logger.info("Building DEM and Slope rasters...")
        if self.gdf is None or not hasattr(self, "elevation_field_name"):
            logger.warning(
                "Contour data or elevation field not available, skipping DEM/Slope generation."
            )
            return None, None

        contour_shp_path = self.settings.paths.output_dir / "temp_contours.shp"
        dem_path = self._get_output_path("elevation_dem_raster")
        slope_path = self._get_output_path("slope_raster")

        # Save contours to temporary Shapefile
        # Need to sanitize elevation field name for shapefile
        sanitized_elev_field = "".join(filter(str.isalnum, self.elevation_field_name))[
            :10
        ]
        if not sanitized_elev_field:
            sanitized_elev_field = "elev"
        gdf_shp = self.gdf.rename(
            columns={self.elevation_field_name: sanitized_elev_field}
        )
        save_vector_data(gdf_shp, contour_shp_path, driver="ESRI Shapefile")

        dem_generated = False
        try:
            # Interpolate contours to DEM using WBT
            # Using Natural Neighbor as an example - requires points
            logger.info("Converting contours to points for interpolation...")
            points_gdf = polyline_to_points(
                gdf_shp[[sanitized_elev_field, "geometry"]]
            )  # Pass only needed columns
            points_shp_path = self.settings.paths.output_dir / "temp_contour_points.shp"
            save_vector_data(points_gdf, points_shp_path, driver="ESRI Shapefile")

            logger.info(
                f"Running Natural Neighbor interpolation (field: '{sanitized_elev_field}')..."
            )
            self.wbt.natural_neighbor_interpolation(
                i=str(points_shp_path),
                field=sanitized_elev_field,
                output=str(dem_path),
                cell_size=self.settings.processing.output_cell_size,
            )
            logger.info(f"Generated DEM raster: {dem_path}")
            self.output_paths["elevation_dem_raster"] = dem_path
            dem_generated = True

            # Calculate Slope from DEM
            logger.info("Calculating slope from DEM...")
            self.wbt.slope(
                dem=str(dem_path),
                output=str(slope_path),
                zfactor=1.0,
                units=self.settings.processing.slope_units,
            )
            logger.info(f"Generated Slope raster: {slope_path}")
            self.output_paths["slope_raster"] = slope_path

        except Exception as e:
            logger.error(f"Error during WBT DEM/Slope generation: {e}", exc_info=True)
            # Ensure paths are not added if generation failed
            if "elevation_dem_raster" in self.output_paths:
                del self.output_paths["elevation_dem_raster"]
            if "slope_raster" in self.output_paths:
                del self.output_paths["slope_raster"]
            return None, None
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
        """Builds DEM and Slope rasters."""
        if self.gdf is None:
            self.load_data()
        if self.gdf is None:
            logger.error("Cannot build Elevation features: Data loading failed.")
            return None, None

        task = self._build_dem_and_slope()
        dem_result, slope_result = dask.compute(task)[0]
        return dem_result, slope_result
