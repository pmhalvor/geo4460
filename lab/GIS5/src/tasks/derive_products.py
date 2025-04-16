# lab/GIS5/src/tasks/derive_products.py
from pathlib import Path
from typing import Optional, Dict, List
from whitebox import WhiteboxTools
from pydantic import BaseModel
import logging
import geopandas as gpd
from shapely.geometry import LineString

# Configure logging
# Consider changing level to logging.DEBUG for more verbose output if needed
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DerivedProductGenerator:
    """
    Generates derived raster and vector products from DEMs using WhiteboxTools.

    Attributes:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        processed_dems: Dictionary storing names and paths of processed DEMs.
        common_crs: The common Coordinate Reference System derived from input data.
        common_extent: The common bounding box (minx, miny, maxx, maxy) of input data.
    """

    def __init__(
        self,
        settings: BaseModel,
        wbt: WhiteboxTools,
        common_crs: Optional[str] = None,
        common_extent: Optional[tuple[float, float, float, float]] = None,
    ):
        """
        Initializes the DerivedProductGenerator.

        Args:
            settings: The application configuration object.
            wbt: Initialized WhiteboxTools object.
            common_crs: The common CRS string (e.g., 'EPSG:XXXX') or object for outputs.
            common_extent: Tuple representing the common bounding box (minx, miny, maxx, maxy).
        """
        self.settings = settings
        self.wbt = wbt
        self.common_crs = common_crs  # Store the common CRS
        self.common_extent = common_extent  # Store the common extent
        self.processed_dems: Dict[str, Path] = {}
        self.wbt.set_verbose_mode(self.settings.processing.wbt_verbose)
        logger.info("DerivedProductGenerator initialized.")
        if not common_crs:
            logger.warning("Common CRS not provided. Transect creation might fail.")
        if not common_extent:
            logger.warning(
                "Common Extent not provided. Dynamic transect creation will fail."
            )

    def _get_output_path(self, key: str) -> Path:
        """Helper to get the full output path from settings."""
        return self.settings.output_files.get_full_path(
            key, self.settings.paths.output_dir
        )

    def add_dem(self, name: str, dem_path: Path):
        """
        Adds a DEM to the internal registry if it exists.

        Args:
            name: A unique name to identify the DEM (e.g., 'interpolated', 'topo').
            dem_path: Path to the DEM raster file.

        Raises:
            FileNotFoundError: If the DEM file does not exist.
        """
        if not isinstance(dem_path, Path):  # Ensure it's a Path object
            dem_path = Path(dem_path)
        if not dem_path.exists():
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        self.processed_dems[name] = dem_path
        logger.info(f"Added DEM '{name}': {dem_path}")

    def _create_transect_line(self) -> Optional[Path]:
        """
        Creates a transect line shapefile dynamically based on the common_extent,
        using the common CRS provided during initialization.

        Returns:
            Path to the created shapefile, or None if extent/CRS are missing
            or an error occurs.
        """
        logger.info("Attempting to create transect line dynamically from extent...")
        crs = self.common_crs
        extent = self.common_extent
        output_key = "transect_created_shp"

        if not crs or not extent:
            logger.warning(
                "Skipping dynamic transect creation: Missing common CRS or extent."
            )
            return None

        if not (isinstance(extent, tuple) and len(extent) == 4):
            logger.warning(
                f"Skipping dynamic transect creation: Invalid extent format ({extent}). Expected (minx, miny, maxx, maxy)."
            )
            return None

        output_path = self._get_output_path(output_key)
        minx, miny, maxx, maxy = extent
        width = maxx - minx
        height = maxy - miny

        # Calculate start/end points for a diagonal line, inset by 10%
        inset_factor = 0.10
        start_x = minx + width * inset_factor
        start_y = miny + height * inset_factor
        end_x = maxx - width * inset_factor
        end_y = maxy - height * inset_factor
        start_coords = (start_x, start_y)
        end_coords = (end_x, end_y)

        logger.info(
            f"  - Calculated Transect: Start={start_coords}, End={end_coords}, CRS={crs}"
        )

        try:
            # Create the LineString geometry
            line = LineString([start_coords, end_coords])

            # Create a GeoDataFrame using the common CRS
            gdf = gpd.GeoDataFrame({"id": [0]}, geometry=[line], crs=crs)
            logger.info(f"  - GeoDataFrame created with CRS: {gdf.crs}")  # Log GDF CRS

            # Save to shapefile
            output_path.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure directory exists
            gdf.to_file(str(output_path), driver="ESRI Shapefile")
            logger.info(f"  - Transect line created successfully: {output_path}")
            return output_path
        except ImportError:
            logger.error(
                "Failed to create transect: Geopandas or Shapely not installed."
            )
            # Re-raise or handle appropriately depending on requirements
            raise
        except Exception as e:
            logger.error(f"Failed to create or save transect line: {e}")
            return None

    def get_contours(self, dem_name: str, output_key: str) -> Path:
        """
        Generates contour lines from a specified DEM.

        Args:
            dem_name: The name of the DEM in processed_dems to use.
            output_key: The key in settings.output_files for the output contour shapefile.

        Returns:
            Path to the generated contour shapefile.

        Raises:
            KeyError: If dem_name is not found in processed_dems.
            Exception: If the WhiteboxTools command fails.
        """
        if dem_name not in self.processed_dems:
            raise KeyError(f"DEM name '{dem_name}' not found in processed DEMs.")

        dem_path = self.processed_dems[dem_name]
        output_path = self._get_output_path(output_key)
        interval = self.settings.processing.contour_interval

        logger.info(
            f"  - Generating contours for '{dem_name}' (interval: {interval}m)..."
        )
        try:
            self.wbt.contours_from_raster(
                i=str(dem_path),
                output=str(output_path),
                interval=interval,
            )
            logger.info(f"    - Contours from '{dem_name}' DEM saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate contours for '{dem_name}': {e}")
            raise

    def get_hillshade(self, dem_name: str, output_key: str) -> Path:
        """
        Generates a hillshade raster from a specified DEM.

        Args:
            dem_name: The name of the DEM in processed_dems to use.
            output_key: The key in settings.output_files for the output hillshade raster.

        Returns:
            Path to the generated hillshade raster.

        Raises:
            KeyError: If dem_name is not found in processed_dems.
            Exception: If the WhiteboxTools command fails.
        """
        if dem_name not in self.processed_dems:
            raise KeyError(f"DEM name '{dem_name}' not found in processed DEMs.")

        dem_path = self.processed_dems[dem_name]
        output_path = self._get_output_path(output_key)

        logger.info(f"  - Generating hillshade for '{dem_name}'...")
        try:
            self.wbt.hillshade(
                dem=str(dem_path),
                output=str(output_path),
                # Default azimuth=315, altitude=30 can be added to settings if needed
            )
            logger.info(
                f"    - Hillshade from '{dem_name}' DEM saved to: {output_path}"
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate hillshade for '{dem_name}': {e}")
            raise

    def get_slope(self, dem_name: str, output_key: str) -> Path:
        """
        Generates a slope raster from a specified DEM.

        Args:
            dem_name: The name of the DEM in processed_dems to use.
            output_key: The key in settings.output_files for the output slope raster.

        Returns:
            Path to the generated slope raster.

        Raises:
            KeyError: If dem_name is not found in processed_dems.
            Exception: If the WhiteboxTools command fails.
        """
        if dem_name not in self.processed_dems:
            raise KeyError(f"DEM name '{dem_name}' not found in processed DEMs.")

        dem_path = self.processed_dems[dem_name]
        output_path = self._get_output_path(output_key)

        logger.info(f"  - Calculating slope for '{dem_name}'...")
        try:
            self.wbt.slope(
                dem=str(dem_path),
                output=str(output_path),
                # Output units: degrees (default)
            )
            logger.info(f"    - Slope from '{dem_name}' DEM saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to calculate slope for '{dem_name}': {e}")
            raise

    def get_profile_analysis(
        self, dem_name: str, transect_path: Path, output_key: str
    ) -> Path:
        """
        Generates a profile plot along a transect line for a specified DEM.

        Args:
            dem_name: The name of the DEM in processed_dems to use.
            transect_path: Path to the vector polyline file representing the transect.
            output_key: The key in settings.output_files for the output profile HTML file.

        Returns:
            Path to the generated profile HTML file.

        Raises:
            KeyError: If dem_name is not found in processed_dems.
            FileNotFoundError: If the transect file does not exist.
            Exception: If the WhiteboxTools command fails.
        """
        if dem_name not in self.processed_dems:
            raise KeyError(f"DEM name '{dem_name}' not found in processed DEMs.")
        if not transect_path.exists():
            raise FileNotFoundError(f"Transect file not found: {transect_path}")

        dem_path = self.processed_dems[dem_name]
        output_path = self._get_output_path(output_key)

        logger.info(
            f"  - Generating profile analysis for '{dem_name}' along {transect_path}..."
        )
        dem_str = str(dem_path)
        lines_str = str(transect_path)
        output_str = str(output_path)

        # Log the parameters being passed to the tool
        logger.info(
            f"    - Calling wbt.profile with: surface='{dem_str}', lines='{lines_str}', output='{output_str}'"
        )
        try:
            # Note: wbt.profile generates an HTML file directly.
            # Corrected parameter name from 'dem'/'rasters' to 'surface'
            self.wbt.profile(surface=dem_str, lines=lines_str, output=output_str)
            logger.info(
                f"    - Profile analysis for '{dem_name}' saved to: {output_path}"
            )
            return output_path
        except Exception as e:
            # Log exception type for better debugging
            logger.error(
                f"Failed profile analysis for '{dem_name}' ({type(e).__name__}): {e}"
            )
            raise

    def get_dem_diff(
        self,
        dem1_name: Optional[str] = None,
        dem2_name: Optional[str] = None,
        output_key: str = "dem_diff_tif",
    ) -> Path:
        """
        Calculates the difference between two DEMs (DEM1 - DEM2).

        If dem1_name and dem2_name are not provided, it uses the last two
        DEMs added to the internal registry.

        Args:
            dem1_name: The name of the first DEM (subtrahend) in processed_dems.
            dem2_name: The name of the second DEM (minuend) in processed_dems.
            output_key: The key in settings.output_files for the output difference raster.

        Returns:
            Path to the generated difference raster.

        Raises:
            ValueError: If fewer than two DEMs are registered and names are not provided,
                        or if specified names are not found.
            Exception: If the WhiteboxTools command fails.
        """
        dem_names = list(self.processed_dems.keys())

        if dem1_name is None or dem2_name is None:
            if len(dem_names) < 2:
                raise ValueError(
                    "Need at least two registered DEMs to calculate difference by default."
                )
            # Default to last two added DEMs. Order might matter depending on desired diff (e.g., topo - interp)
            # Assuming the order added reflects the desired subtraction order (last added - second last added)
            dem1_name = dem_names[-1]  # Typically the 'newer' or reference DEM
            dem2_name = dem_names[-2]  # Typically the 'older' or comparison DEM
            logger.info(
                f"Defaulting DEM difference calculation to: '{dem1_name}' - '{dem2_name}'"
            )

        if dem1_name not in self.processed_dems:
            raise ValueError(
                f"DEM name '{dem1_name}' not found for difference calculation."
            )
        if dem2_name not in self.processed_dems:
            raise ValueError(
                f"DEM name '{dem2_name}' not found for difference calculation."
            )

        dem1_path = self.processed_dems[dem1_name]
        dem2_path = self.processed_dems[dem2_name]
        output_path = self._get_output_path(output_key)

        logger.info(f"  - Calculating DEM difference ({dem1_name} - {dem2_name})...")
        input1_str = str(dem1_path)
        input2_str = str(dem2_path)
        output_str = str(output_path)
        # Use debug level for potentially verbose parameter logging
        logger.debug(
            f"    - Calling wbt.subtract(input1='{input1_str}', input2='{input2_str}', output='{output_str}')"
        )
        try:
            # Ensure subtraction order is correct (input1 - input2)
            self.wbt.subtract(
                input1=input1_str,
                input2=input2_str,
                output=output_str,
            )
            logger.info(f"    - DEM difference map saved to: {output_path}")
            return output_path
        except Exception as e:
            # Log exception type for better debugging
            logger.error(f"Failed DEM difference calculation ({type(e).__name__}): {e}")
            raise


# --- Wrapper Function (for compatibility with workflow.py) ---


def generate_derived_products(
    settings: BaseModel,
    wbt: WhiteboxTools,
    # Updated DEM paths
    dem_interp_contour_path: Optional[Path],
    dem_topo_contour_path: Optional[Path],
    dem_interp_points_path: Optional[Path],
    dem_topo_points_path: Optional[Path],
    dem_stream_burn_path: Optional[Path],
    dem_toporaster_all_path: Optional[Path],  # Keep ANUDEM
    common_crs: Optional[str] = None,
    common_extent: Optional[tuple[float, float, float, float]] = None,
):
    """
    Wrapper function to generate derived raster and vector products using the
    DerivedProductGenerator class. Processes all available input DEMs.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        dem_interp_contour_path: Path to Natural Neighbor (Contour) DEM (or None).
        dem_topo_contour_path: Path to TIN Gridding (Contour) DEM (or None).
        dem_interp_points_path: Path to Natural Neighbor (Points) DEM (or None).
        dem_topo_points_path: Path to TIN Gridding (Points) DEM (or None).
        dem_stream_burn_path: Path to Stream Burn (Contour TIN based) DEM (or None).
        dem_toporaster_all_path: Path to ANUDEM (ArcGIS Pro) DEM (or None).
        common_crs: The common CRS string or object for output vector data.
        common_extent: Tuple representing the common bounding box (minx, miny, maxx, maxy).

    Raises:
        FileNotFoundError: If input DEMs are missing.
        Exception: If any processing step fails.
    """
    logger.info(
        "\n--- Starting Step 4: Further Analysis (Contours, Hillshade, Slope, Profile, Difference) ---"
    )

    # Pass common_crs and common_extent to the generator
    generator = DerivedProductGenerator(
        settings, wbt, common_crs=common_crs, common_extent=common_extent
    )

    try:
        # Define mapping for clarity and consistency
        dem_inputs = {
            "interp_contour": dem_interp_contour_path,
            "topo_contour": dem_topo_contour_path,
            "interp_points": dem_interp_points_path,
            "topo_points": dem_topo_points_path,
            "stream_burn": dem_stream_burn_path,
            "toporaster_all": dem_toporaster_all_path,
        }

        # Add available DEMs to the generator
        for name, path in dem_inputs.items():
            if path and Path(path).exists():
                try:
                    generator.add_dem(name, Path(path))
                except (
                    FileNotFoundError
                ):  # Should not happen due to check, but belt-and-suspenders
                    logger.warning(
                        f"File not found for DEM '{name}' at {path} despite initial check."
                    )
            else:
                logger.info(
                    f"Path for DEM '{name}' not provided or file not found. Skipping."
                )

        # Check if any DEMs were actually added
        if not generator.processed_dems:
            logger.error(
                "No valid DEMs were added to the generator. Cannot create derived products."
            )
            return  # Or raise an error

        # --- Generate products for each available DEM ---
        # Define mappings from internal name to output config keys
        contour_keys = {
            "interp_contour": "contours_interpolated_contour_shp",
            "topo_contour": "contours_topo_contour_shp",
            "interp_points": "contours_interpolated_points_shp",
            "topo_points": "contours_topo_points_shp",
            "stream_burn": "contours_stream_burned_shp",
            "toporaster_all": "contours_toporaster_all_shp",
        }
        hillshade_keys = {
            "interp_contour": "hillshade_interpolated_contour_tif",  # Use renamed key
            "topo_contour": "hillshade_topo_contour_tif",  # Use renamed key
            "interp_points": "hillshade_interpolated_points_tif",
            "topo_points": "hillshade_topo_points_tif",
            "stream_burn": "hillshade_stream_burned_tif",
            "toporaster_all": "hillshade_toporaster_all_tif",
        }
        slope_keys = {
            "interp_contour": "slope_interpolated_contour_tif",  # Use renamed key
            "topo_contour": "slope_topo_contour_tif",
            "interp_points": "slope_interpolated_points_tif",
            "topo_points": "slope_topo_points_tif",
            "stream_burn": "slope_stream_burned_tif",
            "toporaster_all": "slope_toporaster_all_tif",
        }

        # Generate Contours, Hillshade, Slope for each available DEM
        for dem_name in generator.processed_dems:
            logger.info(f"Processing DEM: {dem_name}...")
            try:
                if dem_name in contour_keys:
                    generator.get_contours(dem_name, contour_keys[dem_name])
                if dem_name in hillshade_keys:
                    # TODO: Add new hillshade keys to config.py if needed
                    if hasattr(settings.output_files, hillshade_keys[dem_name]):
                        generator.get_hillshade(dem_name, hillshade_keys[dem_name])
                    else:
                        logger.warning(
                            f"Skipping hillshade for {dem_name}: Output key '{hillshade_keys[dem_name]}' not found in config."
                        )
                if dem_name in slope_keys:
                    generator.get_slope(dem_name, slope_keys[dem_name])
            except KeyError as e:
                logger.warning(
                    f"Skipping product for {dem_name} due to missing output key: {e}"
                )
            except Exception as e:
                logger.error(f"Failed to generate products for {dem_name}: {e}")

        # --- Specific Analyses (can be expanded or made more dynamic) ---

        # Create Transect (dynamically) and Generate Profile Analysis for all available DEMs
        logger.info("Creating Transect and Generating Profile Analyses...")
        # Create the transect line dynamically using the extent stored in the generator
        created_transect_path = generator._create_transect_line()

        if created_transect_path and created_transect_path.exists():
            logger.info(
                f"Using dynamically generated transect: {created_transect_path}"
            )
            # Map internal DEM names to their profile output keys
            profile_output_keys = {
                "interp_contour": "profile_analysis_interp_contour_html",
                "topo_contour": "profile_analysis_topo_contour_html",
                "interp_points": "profile_analysis_interp_points_html",
                "topo_points": "profile_analysis_topo_points_html",
                "stream_burn": "profile_analysis_stream_burned_html",
                "toporaster_all": "profile_analysis_toporaster_all_html",
            }
            for dem_name in generator.processed_dems:
                if dem_name in profile_output_keys:
                    output_key = profile_output_keys[dem_name]
                    try:
                        # Use the created transect path
                        generator.get_profile_analysis(
                            dem_name, created_transect_path, output_key
                        )
                    except KeyError:
                        logger.warning(
                            f"Skipping profile analysis for '{dem_name}': Output key '{output_key}' not found in settings.output_files."
                        )
                    except Exception as profile_e:
                        # Log exception type for better debugging
                        logger.error(
                            f"Failed profile analysis for '{dem_name}' ({type(profile_e).__name__}): {profile_e}"
                        )
                else:
                    logger.info(
                        f"Skipping profile analysis for '{dem_name}': No output key mapping defined."
                    )
        else:
            logger.warning(
                "Skipping profile analysis for all DEMs: Transect line creation failed or was skipped."
            )

        # Calculate DEM Difference (TopoContour - InterpContour)
        logger.info("Calculating DEM Difference (TopoContour - InterpContour)...")
        if (
            "topo_contour" in generator.processed_dems
            and "interp_contour" in generator.processed_dems
        ):
            try:
                generator.get_dem_diff(
                    dem1_name="topo_contour",
                    dem2_name="interp_contour",
                    output_key="dem_diff_tif",  # Uses the renamed key from config
                )
            except Exception as diff_e:
                logger.error(f"Failed to calculate DEM difference: {diff_e}")
        else:
            logger.warning(
                "Skipping DEM difference: Required contour-based DEMs not available."
            )

    except FileNotFoundError as e:
        logger.error(f"Input file not found during Further Analysis: {e}")
        raise
    except KeyError as e:
        logger.error(f"Configuration key error during Further Analysis: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error during Further Analysis: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during Further Analysis: {e}")
        raise  # Re-raise the exception to halt the workflow if needed

    logger.info("--- Step 4: Further Analysis Complete ---")
