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
    """

    def __init__(
        self, settings: BaseModel, wbt: WhiteboxTools, common_crs: Optional[str] = None
    ):
        """
        Initializes the DerivedProductGenerator.

        Args:
            settings: The application configuration object.
            wbt: Initialized WhiteboxTools object.
            common_crs: The common CRS string (e.g., 'EPSG:XXXX') or object for outputs.
        """
        self.settings = settings
        self.wbt = wbt
        self.common_crs = common_crs  # Store the common CRS
        self.processed_dems: Dict[str, Path] = {}
        self.wbt.set_verbose_mode(self.settings.processing.wbt_verbose)
        logger.info("DerivedProductGenerator initialized.")
        if not common_crs:
            logger.warning(
                "Common CRS not provided to DerivedProductGenerator. Transect creation might fail or use default."
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
        Creates a transect line shapefile based on coordinates in settings,
        using the common CRS provided during initialization.

        Returns:
            Path to the created shapefile, or None if coordinates/CRS are missing
            or an error occurs.
        """
        logger.info("Attempting to create transect line...")
        try:
            start_coords = self.settings.processing.transect_start_coords
            end_coords = self.settings.processing.transect_end_coords
            # Use the common CRS stored in the instance
            crs = self.common_crs
            output_key = "transect_created_shp"  # Key for the output filename in OutputFilesConfig
            logger.info(
                f"  - Using Start: {start_coords}, End: {end_coords}, CRS: {crs}"
            )  # Log details
        except AttributeError as e:
            logger.warning(
                f"Skipping transect creation: Missing required setting ({e}). Ensure transect_start_coords, transect_end_coords are in ProcessingConfig and transect_created_shp is in OutputFilesConfig."
            )
            return None

        # Check if all necessary components are present
        # Modified check: Ensure coords are tuples and crs is truthy (exists and not None/empty)
        if not (
            isinstance(start_coords, tuple)
            and len(start_coords) == 2
            and isinstance(end_coords, tuple)
            and len(end_coords) == 2
            and crs
        ):  # Check if crs is truthy (accepts string or CRS object)
            logger.warning(
                "Skipping transect creation: Invalid or missing coordinates/CRS."
            )
            return None

        output_path = self._get_output_path(output_key)

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
    dem_interp_path: Path,
    dem_topo_path: Path,
    dem_toporaster_all_path: Optional[Path] = None,  # ANUDEM
    dem_stream_burn_path: Optional[Path] = None,  # TIN + Stream Burn
    common_crs: Optional[str] = None,  # Add common_crs parameter
):
    """
    Wrapper function to generate derived raster and vector products using the
    DerivedProductGenerator class. Processes interpolated, topo, ANUDEM, and Stream Burn DEMs if available.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        dem_interp_path: Path to the interpolated DEM raster.
        dem_topo_path: Path to the TIN gridded DEM raster.
        dem_toporaster_all_path: Optional path to the ANUDEM raster.
        dem_stream_burn_path: Optional path to the TIN + Stream Burn raster.
        common_crs: The common CRS string or object for output vector data.

    Raises:
        FileNotFoundError: If input DEMs or required transect file are missing.
        Exception: If any processing step fails.
    """
    logger.info(
        "\n--- Starting Step 4: Further Analysis (Contours, Hillshade, Slope, Profile, Difference) ---"
    )

    # Pass common_crs to the generator
    generator = DerivedProductGenerator(settings, wbt, common_crs=common_crs)

    try:
        # Add core DEMs (assuming these are always required by the workflow calling this)
        generator.add_dem("interpolated", dem_interp_path)
        generator.add_dem("topo", dem_topo_path)

        # Add optional DEMs if paths are provided and valid
        if (
            dem_toporaster_all_path and Path(dem_toporaster_all_path).exists()
        ):  # Check existence after ensuring Path
            # Use 'toporaster_all' to match the config input key
            generator.add_dem(
                "toporaster_all", Path(dem_toporaster_all_path)
            )  # Ensure Path object
        else:
            logger.info(
                "TopoToRaster (ANUDEM) path not provided or file not found. Skipping its derived products."
            )

        if (
            dem_stream_burn_path and Path(dem_stream_burn_path).exists()
        ):  # Check existence after ensuring Path
            # Use 'stream_burned' to match the config key dem_stream_burned_tif
            generator.add_dem(
                "stream_burned", Path(dem_stream_burn_path)
            )  # Ensure Path object
        else:
            logger.info(
                "TIN+Stream Burn path not provided or file not found. Skipping its derived products."
            )
        # --- Generate products for each available DEM ---

        # Interpolated DEM
        logger.info("Processing Interpolated DEM...")
        generator.get_contours("interpolated", "contours_interpolated_shp")
        generator.get_hillshade("interpolated", "hillshade_interpolated_tif")
        generator.get_slope("interpolated", "slope_interpolated_tif")

        # Topo DEM
        logger.info("Processing Topo DEM...")
        generator.get_contours("topo", "contours_topo_shp")
        generator.get_hillshade("topo", "hillshade_topo_tif")
        generator.get_slope("topo", "slope_topo_tif")

        # TopoToRaster (ANUDEM) (if available)
        if "toporaster_all" in generator.processed_dems:
            logger.info("Processing TopoToRaster (ANUDEM)...")
            # Use the correct config keys added previously
            try:
                generator.get_contours("toporaster_all", "contours_toporaster_all_shp")
                generator.get_hillshade(
                    "toporaster_all", "hillshade_toporaster_all_tif"
                )
                generator.get_slope("toporaster_all", "slope_toporaster_all_tif")
            except KeyError as e:
                logger.warning(
                    f"Skipping TopoToRaster product generation due to missing output key: {e}"
                )
            except Exception as e:
                logger.error(f"Failed to generate products for TopoToRaster: {e}")

        # TIN + Stream Burn (if available)
        if "stream_burned" in generator.processed_dems:
            logger.info("Processing TIN + Stream Burn DEM...")
            # Use the correct config keys added previously
            try:
                generator.get_contours("stream_burned", "contours_stream_burned_shp")
                generator.get_hillshade("stream_burned", "hillshade_stream_burned_tif")
                generator.get_slope("stream_burned", "slope_stream_burned_tif")
            except KeyError as e:
                logger.warning(
                    f"Skipping Stream Burn product generation due to missing output key: {e}"
                )
            except Exception as e:
                logger.error(f"Failed to generate products for Stream Burn DEM: {e}")

        # --- Specific Analyses (can be expanded or made more dynamic) ---

        # Create Transect and Generate Profile Analysis for all available DEMs
        logger.info("Creating Transect and Generating Profile Analyses...")
        # Create the transect line first using the new method
        created_transect_path = generator._create_transect_line()

        if created_transect_path and created_transect_path.exists():
            # Map internal DEM names to their profile output keys
            profile_output_keys = {
                "interpolated": "profile_analysis_interp_html",
                "topo": "profile_analysis_topo_html",
                "toporaster_all": "profile_analysis_toporaster_all_html",
                "stream_burned": "profile_analysis_stream_burned_html",
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

        # Calculate DEM Difference (Topo - Interpolated by default)
        logger.info("Calculating DEM Difference...")
        # Uses the default behavior: last added ('topo') - second last added ('interpolated')
        # Explicitly setting names for clarity, matching original script's intent (Topo - Interpolated)
        generator.get_dem_diff(
            dem1_name="topo", dem2_name="interpolated", output_key="dem_diff_tif"
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
