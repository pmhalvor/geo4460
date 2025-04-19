import logging
import dask
from pathlib import Path

logger = logging.getLogger(__name__)

# Local imports (adjust relative paths as needed)
try:
    from .feature_base import FeatureBase
    from src.config import AppConfig
    from src.utils import load_raster_data  # Needed for placeholder logic
except ImportError:
    logger.warning(
        "Could not import from src.* or .feature_base directly, attempting relative import..."
    )
    from feature_base import FeatureBase
    from ...config import AppConfig
    from ...utils import load_raster_data


class CostDistance(FeatureBase):
    """Calculates a cost surface based on slope, speed, and road restrictions."""

    # This class needs access to the outputs of other features (Slope, Roads, maybe Heatmap)
    # It might be better structured as a function called *after* other features are built,
    # or it needs references to the other feature objects passed during initialization.

    def __init__(
        self,
        settings: AppConfig,
        wbt,  # Pass WhiteboxTools instance
        slope_raster_path: Path | str,  # Make path mandatory
        roads_raster_path: Path | str = None,  # Optional rasterized roads
        speed_raster_path: Path | str = None,  # Optional speed raster
    ):
        super().__init__(settings, wbt)
        # Ensure paths are Path objects
        self.slope_raster_path = Path(slope_raster_path) if slope_raster_path else None
        self.roads_raster_path = Path(roads_raster_path) if roads_raster_path else None
        self.speed_raster_path = Path(speed_raster_path) if speed_raster_path else None
        self.load_data()  # Validate inputs in init

    def load_data(self):
        # Validate required inputs passed during initialization
        logger.info("CostDistance initialized with input raster paths.")
        if not self.slope_raster_path or not self.slope_raster_path.exists():
            raise FileNotFoundError(
                f"Slope raster path is required for CostDistance and not found: {self.slope_raster_path}"
            )
        if self.roads_raster_path and not self.roads_raster_path.exists():
            logger.warning(
                f"Provided roads raster path not found: {self.roads_raster_path}"
            )
            self.roads_raster_path = None  # Reset if not found
        if self.speed_raster_path and not self.speed_raster_path.exists():
            logger.warning(
                f"Provided speed raster path not found: {self.speed_raster_path}"
            )
            self.speed_raster_path = None  # Reset if not found

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
                # Save using the base class helper method
                self._save_raster(slope_data, profile, "cost_function_raster")
                logger.info(f"Placeholder: Copied slope raster to {output_path}")
                # Return the path stored by _save_raster
                return self.output_paths.get("cost_function_raster")
            except Exception as e:
                logger.error(f"Failed to create placeholder cost raster: {e}")
                return None
        return None

    def build(self):
        """Builds the cost raster."""
        # load_data is called in __init__ for validation
        task = self._build_cost_raster()
        result = dask.compute(task)[0]
        return result  # Returns the path or None
