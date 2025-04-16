import os
import shutil
import subprocess
import tempfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel
from whitebox import WhiteboxTools

# Get a logger for this module
logger = logging.getLogger(__name__)

# --- Base Class for DEM Generation Methods ---


class BaseDemGenerator(ABC):
    """
    Abstract base class for different DEM generation methods.
    """

    def __init__(
        self,
        settings: BaseModel,
        wbt: WhiteboxTools,
        output_dir: Path,
        output_files: Any,
        processing_settings: Any,
    ):
        self.settings = settings  # Full settings object
        self.wbt = wbt
        self.output_dir = output_dir
        self.output_files = output_files  # Specific output file naming helper
        self.processing = processing_settings  # Specific processing settings
        self.wbt.set_verbose_mode(self.processing.wbt_verbose)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str, level: str = "info", indent: int = 1):
        """Helper for logging messages with indentation."""
        indent_str = "  " * indent
        # Map simple levels to logging levels
        level_map = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "debug": logging.DEBUG,
        }
        log_level = level_map.get(level.lower(), logging.INFO)  # Default to INFO
        # Use the module-level logger with the correct level and message
        logger.log(log_level, f"{indent_str}{message}")

    @abstractmethod
    def generate(self, inputs: Dict[str, Path]) -> bool:
        """
        Generates the specific DEM type.

        Args:
            inputs: A dictionary containing necessary input file paths for this method.

        Returns:
            bool: True if generation was successful, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Returns the name of the DEM generation method."""
        pass

    # Removed output_path_key property, will be passed during generation
    # Removed get_output_path, will be handled by the workflow


# --- Concrete DEM Generator Implementations ---
# Note: These generators are now more generic and rely on the workflow
# to provide the correct input path and output path.


class InterpolationDemGenerator(BaseDemGenerator):
    """Generates DEM using Natural Neighbour interpolation."""

    @property
    def method_name(self) -> str:
        return "Natural Neighbour Interpolation"

    # Removed output_path_key property

    def generate(self, input_points_path: Path, output_dem_path: Path) -> bool:
        """Generates DEM using Natural Neighbour interpolation."""
        self._log(f"Method: {self.method_name}...", indent=1)
        self._log(f"  Input points: {input_points_path.name}", indent=2)
        self._log(f"  Output DEM: {output_dem_path.name}", indent=2)

        if not input_points_path or not input_points_path.exists():
            self._log(
                f"Required input points not found at {input_points_path}",
                level="error",
                indent=2,
            )
            return False

        try:
            self._log("Running Natural Neighbour Interpolation...", indent=2)
            # Assuming the elevation field is 'VALUE' for both contour-derived points
            # and direct elevation points after the load_data step.
            self.wbt.natural_neighbour_interpolation(
                i=str(input_points_path),
                field="VALUE",
                output=str(output_dem_path),
                cell_size=self.processing.output_cell_size,
            )
            self._log(f"Interpolated DEM saved: {output_dem_path}", indent=2)
            if not output_dem_path.exists():
                raise FileNotFoundError(
                    f"Interpolated DEM file not created: {output_dem_path}"
                )
            return True
        except Exception as e:
            self._log(f"{self.method_name} failed: {e}", level="error", indent=1)
            return False


class TinGriddingDemGenerator(BaseDemGenerator):
    """Generates DEM using TIN Gridding."""

    @property
    def method_name(self) -> str:
        return "TIN Gridding"

    # Removed output_path_key property

    def generate(self, input_points_path: Path, output_dem_path: Path) -> bool:
        """Generates DEM using TIN Gridding."""
        self._log(f"Method: {self.method_name}...", indent=1)
        self._log(f"  Input points: {input_points_path.name}", indent=2)
        self._log(f"  Output DEM: {output_dem_path.name}", indent=2)

        if not input_points_path or not input_points_path.exists():
            self._log(
                f"Required input points not found at {input_points_path}",
                level="error",
                indent=2,
            )
            return False

        try:
            self._log("Running TIN Gridding...", indent=2)
            # Assuming the elevation field is 'VALUE' for both contour-derived points
            # and direct elevation points after the load_data step.
            self.wbt.tin_gridding(
                i=str(input_points_path),
                field="VALUE",
                output=str(output_dem_path),
                resolution=self.processing.output_cell_size,
            )
            self._log(f"TIN Gridding DEM saved: {output_dem_path}", indent=2)
            if not output_dem_path.exists():
                raise FileNotFoundError(
                    f"TIN Gridding DEM file not created: {output_dem_path}"
                )
            return True
        except Exception as e:
            self._log(f"{self.method_name} failed: {e}", level="error", indent=1)
            return False


class StreamBurnDemGenerator(BaseDemGenerator):
    """Generates a stream-burned DEM using GRASS GIS."""

    def __init__(
        self,
        settings: BaseModel,
        wbt: WhiteboxTools,
        output_dir: Path,
        output_files: Any,
        processing_settings: Any,
        stream_extract_threshold: Optional[int],
    ):
        super().__init__(settings, wbt, output_dir, output_files, processing_settings)
        self.stream_extract_threshold = stream_extract_threshold
        self._original_env: Optional[Dict[str, Optional[str]]] = (
            None  # For env var restoration
        )

    @property
    def method_name(self) -> str:
        return "Stream Extraction and Burning (GRASS)"

    # Removed output_path_key property

    def get_output_path(self) -> Path:
        """Gets the full output path for the stream burned DEM."""
        # Stream burn only happens once, based on contour TIN
        return self.output_files.get_full_path("dem_stream_burned_tif", self.output_dir)

    def _setup_grass_environment(self) -> bool:
        """Sets up GRASS environment variables."""
        self._log("Setting up GRASS environment...", indent=3)
        try:
            grass_exec_path_str = self.settings.paths.grass_executable_path
            if not grass_exec_path_str or not Path(grass_exec_path_str).exists():
                raise ValueError(
                    f"GRASS executable path invalid or not found: {grass_exec_path_str}"
                )
            grass_executable_path_obj = Path(grass_exec_path_str)
            gisbase = str(grass_executable_path_obj.parent.parent)
            self._log(f"Calculated GISBASE: {gisbase}", indent=4)
            if not Path(gisbase).is_dir():
                raise RuntimeError(
                    f"Calculated GISBASE directory does not exist: {gisbase}"
                )

            self._original_env = {
                "PATH": os.environ.get("PATH", ""),
                "DYLD_LIBRARY_PATH": os.environ.get("DYLD_LIBRARY_PATH", ""),
                "GISBASE": os.environ.get("GISBASE"),
                "GRASSBIN": os.environ.get("GRASSBIN"),
            }

            gisbase_bin = Path(gisbase) / "bin"
            gisbase_scripts = Path(gisbase) / "scripts"
            gisbase_lib = Path(gisbase) / "lib"

            os.environ["PATH"] = (
                f"{gisbase_bin}:{gisbase_scripts}:{self._original_env['PATH']}"
            )
            os.environ["DYLD_LIBRARY_PATH"] = (
                f"{gisbase_lib}:{self._original_env['DYLD_LIBRARY_PATH']}"
            )
            os.environ["GISBASE"] = gisbase
            os.environ["GRASSBIN"] = str(grass_executable_path_obj)

            self._log(f"Set GISBASE: {os.environ['GISBASE']}", indent=4)
            self._log(f"Set GRASSBIN: {os.environ['GRASSBIN']}", indent=4)
            return True
        except Exception as e:
            self._log(
                f"Failed to set up GRASS environment: {e}", level="error", indent=3
            )
            return False

    def _restore_grass_environment(self):
        """Restores original environment variables."""
        if self._original_env is not None:
            self._log("Restoring original environment variables...", indent=3)
            os.environ["PATH"] = self._original_env["PATH"]
            os.environ["DYLD_LIBRARY_PATH"] = self._original_env["DYLD_LIBRARY_PATH"]
            if self._original_env["GISBASE"] is None:
                if "GISBASE" in os.environ:
                    del os.environ["GISBASE"]
            else:
                os.environ["GISBASE"] = self._original_env["GISBASE"]
            if self._original_env["GRASSBIN"] is None:
                if "GRASSBIN" in os.environ:
                    del os.environ["GRASSBIN"]
            else:
                os.environ["GRASSBIN"] = self._original_env["GRASSBIN"]
            self._original_env = None  # Mark as restored

    def _create_grass_location(
        self, temp_dir: Path, location_name: str, base_raster: Path
    ) -> Optional[Path]:
        """Creates a temporary GRASS location."""
        grass_location_path = temp_dir / location_name
        grass_executable = os.environ.get("GRASSBIN")
        if not grass_executable:
            self._log("GRASSBIN environment variable not set.", level="error", indent=3)
            return None

        self._log(
            f"Creating temporary GRASS location '{location_name}' at: {grass_location_path}",
            indent=3,
        )
        create_loc_cmd = [
            grass_executable,
            "-c",
            str(base_raster),
            str(grass_location_path),
            "-e",
        ]
        self._log(f"Running: {' '.join(create_loc_cmd)}", indent=4)
        try:
            result = subprocess.run(
                create_loc_cmd, check=True, capture_output=True, text=True, timeout=90
            )
            if result.stdout:
                self._log(
                    f"GRASS Location Creation stdout:\n{result.stdout.strip()}",
                    indent=5,
                )
            if result.stderr:
                self._log(
                    f"GRASS Location Creation stderr:\n{result.stderr.strip()}",
                    indent=5,
                )  # Info often goes to stderr
            self._log("Location created successfully.", indent=4)
            return grass_location_path
        except subprocess.CalledProcessError as cpe:
            self._log(
                f"Failed to create GRASS location. Return Code: {cpe.returncode}",
                level="error",
                indent=4,
            )
            self._log(f"Stderr: {cpe.stderr}", level="error", indent=5)
            return None
        except Exception as e:
            self._log(
                f"Unexpected error during GRASS location creation: {e}",
                level="error",
                indent=4,
            )
            return None

    def _run_grass_commands(
        self, temp_grass_db: Path, location_name: str, input_dem_path: Path
    ) -> bool:
        """Runs the core GRASS commands within a session."""
        try:
            from grass_session import Session
            import grass.script as gs
        except ImportError as e:
            self._log(
                f"Failed to import GRASS Python libraries: {e}. "
                f"Ensure GRASS environment is active or GRASSBIN is set.",
                level="error",
                indent=3,
            )
            return False

        mapset = "PERMANENT"
        in_rast = "input_dem"
        streams_rast = "streams_extracted"
        out_rast = "dem_burned"
        burn_val = self.processing.stream_burn_value
        output_path = self.get_output_path()

        self._log(f"Starting GRASS session in location '{location_name}'...", indent=3)
        try:
            with Session(
                gisdb=str(temp_grass_db), location=location_name, mapset=mapset
            ):
                self._log("GRASS session started. Running commands...", indent=4)
                gs.run_command(
                    "r.in.gdal",
                    input=str(input_dem_path),
                    output=in_rast,
                    flags="o",
                    overwrite=True,
                    quiet=False,
                )
                gs.run_command("g.region", raster=in_rast, flags="p")
                gs.run_command(
                    "r.stream.extract",
                    elevation=in_rast,
                    threshold=self.stream_extract_threshold,
                    stream_raster=streams_rast,
                    overwrite=True,
                    verbose=True,
                )
                mapcalc_expr = f"{out_rast} = if(isnull({streams_rast}), {in_rast}, {in_rast} + ({burn_val}))"
                self._log(f"Mapcalc expression: {mapcalc_expr}", indent=5)
                gs.run_command(
                    "r.mapcalc", expression=mapcalc_expr, overwrite=True, verbose=True
                )
                # Verify output map exists
                gs.find_file(
                    name=out_rast, element="cell"
                )  # Raises exception if not found
                self._log(f"Exporting burned DEM to: {output_path}...", indent=5)
                gs.run_command(
                    "r.out.gdal",
                    input=out_rast,
                    output=str(output_path),
                    format="GTiff",
                    type="Float32",
                    createopt="PROFILE=GeoTIFF,TFW=YES",
                    flags="f",
                    overwrite=True,
                    verbose=True,
                )
            self._log("GRASS session closed.", indent=4)
            return True
        except Exception as e:
            self._log(
                f"GRASS session or command execution failed: {e}",
                level="error",
                indent=4,
            )
            if hasattr(e, "stderr"):
                self._log(f"GRASS stderr:\n{e.stderr}", level="error", indent=5)
            return False

    def generate(self, input_contour_tin_dem_path: Path) -> bool:
        """Generates the stream-burned DEM based on the contour-derived TIN DEM."""
        self._log(f"Method: {self.method_name}...", indent=1)
        output_path = self.get_output_path()  # Use the specific method here
        self._log(
            f"  Input DEM (Contour TIN): {input_contour_tin_dem_path.name}", indent=2
        )
        self._log(f"  Output DEM: {output_path.name}", indent=2)

        # --- Pre-checks ---
        if not self.processing.enable_stream_burning:
            self._log("Stream burning is disabled in settings.", indent=2)
            return True  # Skipped successfully
        if not self.stream_extract_threshold:
            self._log(
                "Stream burning enabled, but 'stream_extract_threshold' not provided.",
                level="error",
                indent=2,
            )
            return False
        if not input_contour_tin_dem_path or not input_contour_tin_dem_path.exists():
            self._log(
                f"Required input Contour TIN DEM for stream burning not found at {input_contour_tin_dem_path}",
                level="error",
                indent=2,
            )
            return False

        temp_grass_db_path = None
        success = False
        env_setup = False
        try:
            if not self._setup_grass_environment():
                return False
            env_setup = True

            temp_grass_db_path = Path(tempfile.mkdtemp(prefix="grass_stream_burn_"))
            location_name = "temp_stream_burn_loc"
            self._log(f"Using temporary GRASS DB: {temp_grass_db_path}", indent=2)

            grass_loc_path = self._create_grass_location(
                temp_grass_db_path, location_name, input_contour_tin_dem_path
            )
            if not grass_loc_path:
                return False

            if not self._run_grass_commands(
                temp_grass_db_path, location_name, input_contour_tin_dem_path
            ):
                return False

            if not output_path.exists():
                raise FileNotFoundError(
                    f"Stream-burned DEM file was not created: {output_path}"
                )
            self._log(
                f"Stream-burned DEM successfully created: {output_path}", indent=2
            )
            success = True

        except Exception as e:
            self._log(
                f"An unexpected error occurred during stream burning: {e}",
                level="error",
                indent=2,
            )
            success = False
        finally:
            if env_setup:
                self._restore_grass_environment()
            if temp_grass_db_path and temp_grass_db_path.exists():
                self._log(
                    f"Cleaning up temporary GRASS directory: {temp_grass_db_path}",
                    indent=2,
                )
                try:
                    shutil.rmtree(temp_grass_db_path)
                except OSError as e:
                    self._log(
                        f"Could not remove temp GRASS dir {temp_grass_db_path}: {e}",
                        level="warning",
                        indent=3,
                    )

        return success


# --- Workflow Orchestrator ---


class DemGenerationWorkflow:
    """Orchestrates the generation of multiple DEM types."""

    def __init__(
        self,
        settings: BaseModel,
        wbt: WhiteboxTools,
        contour_shp_path: Path,  # Contour lines input
        contour_elev_field: str,  # Field name for contour elevation
        elevation_points_shp_path: Path,  # Direct elevation points input
        river_shp_path: Optional[Path] = None,
        stream_extract_threshold: Optional[int] = None,
    ):
        self.settings = settings
        self.wbt = wbt
        self.contour_shp_path = contour_shp_path
        self.contour_elev_field = contour_elev_field
        self.elevation_points_shp_path = elevation_points_shp_path
        self.river_shp_path = river_shp_path
        self.stream_extract_threshold = stream_extract_threshold

        self.output_dir = self.settings.paths.output_dir
        self.output_files = self.settings.output_files
        self.processing = self.settings.processing

        # Re-introduce intermediate paths for contour processing
        self.contour_raster_path = self.output_files.get_full_path(
            "contour_raster_temp", self.output_dir
        )
        self.contour_points_with_value_path = self.output_files.get_full_path(
            "contour_points_with_value_shp", self.output_dir
        )

        # Instantiate generator classes (now generic)
        self.interpolation_generator = InterpolationDemGenerator(
            settings, wbt, self.output_dir, self.output_files, self.processing
        )
        self.tin_gridding_generator = TinGriddingDemGenerator(
            settings, wbt, self.output_dir, self.output_files, self.processing
        )
        self.stream_burn_generator = StreamBurnDemGenerator(
            settings=settings,
            wbt=wbt,
            output_dir=self.output_dir,
            output_files=self.output_files,
            processing_settings=self.processing,
            stream_extract_threshold=stream_extract_threshold,
        )  # Corrected closing parenthesis placement

    def _log(self, message: str, level: str = "info", indent: int = 1):
        """Orchestrator logging."""
        indent_str = "  " * indent
        # Map simple levels to logging levels
        level_map = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "debug": logging.DEBUG,
        }
        log_level = level_map.get(level.lower(), logging.INFO)  # Default to INFO
        # Use the module-level logger
        logger.log(log_level, f"{indent_str}{message}")

    # Re-introduce _prepare_contour_points method
    def _prepare_contour_points(self) -> bool:
        """Generates the shared intermediate contour points shapefile from contour lines."""
        self._log("Preparing intermediate contour points with elevation...", indent=1)
        try:
            # Convert contour lines to raster
            self._log(
                f"Converting contour lines to raster ({self.contour_raster_path.name})...",
                indent=2,
            )
            self.wbt.vector_lines_to_raster(
                i=str(self.contour_shp_path),
                output=str(self.contour_raster_path),
                field=self.contour_elev_field,
                nodata=-9999.0,
                cell_size=self.processing.output_cell_size,
                base=None,  # Use default extent based on input vector
            )
            if not self.contour_raster_path.exists():
                raise FileNotFoundError("Intermediate contour raster not created.")
            self._log(f"Contour raster saved: {self.contour_raster_path}", indent=2)

            # Convert contour raster back to points
            self._log(
                f"Converting contour raster to points ({self.contour_points_with_value_path.name})...",
                indent=2,
            )
            self.wbt.raster_to_vector_points(
                i=str(self.contour_raster_path),
                output=str(self.contour_points_with_value_path),
            )
            if not self.contour_points_with_value_path.exists():
                raise FileNotFoundError(
                    "Intermediate contour points file with values not created."
                )
            self._log(
                f"Intermediate contour points saved: {self.contour_points_with_value_path}",
                indent=2,
            )

            self._log("Intermediate contour points preparation successful.", indent=1)
            return True
        except Exception as e:
            self._log(
                f"Failed to prepare intermediate contour points: {e}",
                level="error",
                indent=1,
            )
            return False

    def run_all(self) -> bool:
        """Runs the full DEM generation workflow for both contour and point inputs."""
        logger.info("Starting DEM Generation (Contours and Elevation Points)...")
        overall_success = True
        results: Dict[str, bool] = {}  # Track success of each step

        # --- Step 1: Prepare Intermediate Contour Points (if needed) ---
        if not self._prepare_contour_points():
            logger.error("--- DEM Generation Failed (Contour Points Preparation) ---")
            return False  # Cannot proceed with contour-based methods

        # --- Step 2: Generate DEMs from Contours ---
        logger.info("Generating DEMs from Contours...")
        # 2a: Interpolation (Contour)
        interp_contour_out = self.output_files.get_full_path(
            "dem_interpolated_contour_tif", self.output_dir
        )
        results["interpolation_contour"] = self.interpolation_generator.generate(
            input_points_path=self.contour_points_with_value_path,
            output_dem_path=interp_contour_out,
        )
        if not results["interpolation_contour"]:
            overall_success = False

        # 2b: TIN Gridding (Contour)
        tin_contour_out = self.output_files.get_full_path(
            "dem_topo_contour_tif", self.output_dir
        )
        results["tin_gridding_contour"] = self.tin_gridding_generator.generate(
            input_points_path=self.contour_points_with_value_path,
            output_dem_path=tin_contour_out,
        )
        if not results["tin_gridding_contour"]:
            overall_success = False

        # --- Step 3: Generate DEMs from Elevation Points ---
        logger.info("Generating DEMs from Elevation Points...")
        # 3a: Interpolation (Points)
        interp_points_out = self.output_files.get_full_path(
            "dem_interpolated_points_tif", self.output_dir
        )
        results["interpolation_points"] = self.interpolation_generator.generate(
            input_points_path=self.elevation_points_shp_path,
            output_dem_path=interp_points_out,
        )
        if not results["interpolation_points"]:
            overall_success = False

        # 3b: TIN Gridding (Points)
        tin_points_out = self.output_files.get_full_path(
            "dem_topo_points_tif", self.output_dir
        )
        results["tin_gridding_points"] = self.tin_gridding_generator.generate(
            input_points_path=self.elevation_points_shp_path,
            output_dem_path=tin_points_out,
        )
        if not results["tin_gridding_points"]:
            overall_success = False

        # --- Step 4: Run Stream Burning (based on Contour TIN) ---
        if self.processing.enable_stream_burning:
            if results.get(
                "tin_gridding_contour", False
            ):  # Check if contour TIN succeeded
                results["stream_burn"] = self.stream_burn_generator.generate(
                    input_contour_tin_dem_path=tin_contour_out  # Use the contour TIN output
                )
                if not results["stream_burn"]:
                    overall_success = False
            else:
                self._log(
                    f"Skipping {self.stream_burn_generator.method_name} because prerequisite Contour TIN Gridding failed.",
                    level="warning",
                    indent=1,
                )
                results["stream_burn"] = False  # Mark as failed due to dependency
                overall_success = False
        else:
            # Use the correct generator instance variable name
            self._log(
                f"Skipping {self.stream_burn_generator.method_name} (disabled in settings).",
                indent=1,
            )
            results["stream_burn"] = True  # Skipped successfully

        # --- Summary ---
        logger.info("--- DEM Generation Summary ---")
        # Define friendly names for logging
        method_names = {
            "interpolation_contour": "Natural Neighbour (Contour)",
            "tin_gridding_contour": "TIN Gridding (Contour)",
            "interpolation_points": "Natural Neighbour (Points)",
            "tin_gridding_points": "TIN Gridding (Points)",
            "stream_burn": "Stream Burning (GRASS, based on Contour TIN)",
        }
        for key, success in results.items():
            method_name = method_names.get(key, key)  # Get friendly name or use key
            status = (
                "Success"
                if success
                else (
                    "Failed"
                    if key != "stream_burn" or self.processing.enable_stream_burning
                    else "Skipped"
                )
            )
            logger.info(f"  - {method_name}: {status}")

        if overall_success:
            logger.info(
                "--- All enabled DEM generation steps completed successfully ---"
            )
        else:
            logger.warning(
                "--- One or more DEM generation steps failed or were skipped due to errors ---"
            )

        return overall_success


# --- Original Function Wrapper (Maintains Compatibility) ---


def generate_dems(
    settings: BaseModel,
    wbt: WhiteboxTools,
    contour_shp_path: Path,  # Added back
    contour_elev_field: str,  # Added back
    elevation_points_shp_path: Path,
    river_shp_path: Optional[Path] = None,
    stream_extract_threshold: Optional[int] = None,
) -> bool:
    """
    Generates DEM rasters using various methods orchestrated by DemGenerationWorkflow,
    supporting both contour lines and elevation points as inputs.

    Args:
        settings: The application configuration object.
        wbt: Initialized WhiteboxTools object.
        contour_shp_path: Path to the contour shapefile.
        contour_elev_field: Name of the elevation field in the contour shapefile.
        elevation_points_shp_path: Path to the elevation points shapefile.
                                   (Assumes 'VALUE' field for elevation).
        river_shp_path: Optional path to the river shapefile.
        stream_extract_threshold: Optional threshold value for r.stream.extract.

    Returns:
        bool: True if all enabled generation steps were successful, False otherwise.
    """
    workflow = DemGenerationWorkflow(
        settings=settings,
        wbt=wbt,
        contour_shp_path=contour_shp_path,  # Pass contour path
        contour_elev_field=contour_elev_field,  # Pass contour field
        elevation_points_shp_path=elevation_points_shp_path,  # Pass points path
        river_shp_path=river_shp_path,
        stream_extract_threshold=stream_extract_threshold,
    )
    return workflow.run_all()


# --- Example Usage ---
if __name__ == "__main__":
    try:
        # This assumes execution from a context where 'src' is importable
        # e.g., running 'python -m lab.GIS5.src.tasks.generate_dems' from project root
        # Or having the project root in PYTHONPATH
        from src.config import settings
    except ImportError:
        logger.error("Could not import settings from src.config.")
        logger.error(
            "Ensure you are running this script from the project root or have set PYTHONPATH."
        )
        exit(1)

    logger.info("--- Testing generate_dems.py ---")

    # Initialize WhiteboxTools
    try:
        wbt_test = WhiteboxTools()
        # Set WBT working dir - Use project root if available, else output dir
        wbt_working_dir = settings.paths.output_dir  # Default
        if (
            hasattr(settings.paths, "project_root")
            and settings.paths.project_root.is_dir()
        ):
            wbt_working_dir = settings.paths.project_root
        else:
            logger.warning(
                f"settings.paths.project_root not found or invalid. "
                f"Using output_dir as WBT working dir: {wbt_working_dir}"
            )
        wbt_test.set_working_dir(str(wbt_working_dir))
        logger.info(f"WhiteboxTools initialized. Working directory: {wbt_working_dir}")
    except Exception as e:
        logger.error(f"Error initializing WhiteboxTools: {e}")
        exit(1)

    # Define inputs using paths from settings for testing
    contour_shp_test_path = settings.output_files.get_full_path(
        "contour_shp", settings.paths.output_dir
    )
    points_shp_test_path = settings.output_files.get_full_path(
        "points_shp", settings.paths.output_dir
    )
    river_shp_test_path = settings.output_files.get_full_path(
        "river_shp", settings.paths.output_dir
    )
    contour_elev_field_test = settings.input_layers.contour_elevation_field

    # Check required inputs for testing
    if not contour_shp_test_path.exists():
        logger.error(
            f"Contour shapefile for testing not found at {contour_shp_test_path}"
        )
        logger.error("Please run the 'load_data' step first.")
        exit(1)
    if not points_shp_test_path.exists():
        logger.error(
            f"Elevation points shapefile for testing not found at {points_shp_test_path}"
        )
        logger.error("Please run the 'load_data' step first.")
        exit(1)
    # River check remains the same for stream burning
    if settings.processing.enable_stream_burning and (
        not river_shp_test_path or not river_shp_test_path.exists()
    ):
        logger.warning(
            f"Stream burning enabled, but river shapefile not found at {river_shp_test_path}."
        )

    logger.info(f"Using Contour Shapefile: {contour_shp_test_path}")
    logger.info(f"Using Contour Elevation Field: {contour_elev_field_test}")
    logger.info(f"Using Elevation Points Shapefile: {points_shp_test_path}")
    if settings.processing.enable_stream_burning:
        logger.info(
            f"Stream Burning Enabled: True (Threshold: {settings.processing.stream_extract_threshold}, "
            f"Burn Value: {settings.processing.stream_burn_value})"
        )
        logger.info(f"GRASS Executable: {settings.paths.grass_executable_path}")
    else:
        logger.info("Stream Burning Enabled: False")

    try:
        # Call the main wrapper function
        test_success = generate_dems(
            settings=settings,
            wbt=wbt_test,
            contour_shp_path=contour_shp_test_path,  # Pass contour path
            contour_elev_field=contour_elev_field_test,  # Pass contour field
            elevation_points_shp_path=points_shp_test_path,  # Pass points path
            river_shp_path=river_shp_test_path,
            stream_extract_threshold=(
                settings.processing.stream_extract_threshold
                if settings.processing.enable_stream_burning
                else None
            ),
        )
        if test_success:
            logger.info("--- generate_dems.py test completed successfully ---")
        else:
            logger.warning(
                "--- generate_dems.py test finished with errors (check logs above) ---"
            )

    except Exception as e:
        logger.error(
            f"--- generate_dems.py test failed with unhandled exception: {e} ---",
            exc_info=True,
        )
        # import traceback # No longer needed, logger handles it
        # traceback.print_exc()
