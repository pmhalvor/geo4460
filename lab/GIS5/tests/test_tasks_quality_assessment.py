from pathlib import Path
from shapely.geometry import Point
from src.config import AppConfig
from src.tasks.quality_assessment import assess_dem_quality
from unittest.mock import MagicMock
import pytest
import tempfile

import geopandas as gpd

# Example usage
settings = AppConfig()
# Set up a test for QualityAssessor


@pytest.fixture
def mock_whitebox():
    """Create a mock WhiteboxTools object."""
    mock_wbt = MagicMock()
    return mock_wbt


@pytest.fixture
def mock_settings():
    """Create mock settings with necessary attributes."""
    mock = MagicMock()
    mock.paths.output_dir = Path(tempfile.mkdtemp())
    mock.output_files.get_full_path = lambda name, dir: dir / f"{name}.shp"
    mock.processing.wbt_verbose = False
    return mock


@pytest.fixture
def test_points_file():
    """Create a temporary points shapefile for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple point dataset
        points = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
                "Elevation": [100, 110, 120],
            }
        )
        points_path = Path(temp_dir) / "test_points.shp"
        points.to_file(points_path)
        yield points_path


@pytest.fixture
def test_dem_file():
    """Mock a DEM file that exists."""
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        dem_path = Path(tmp.name)
        yield dem_path


def test_assess_dem_quality_successful_run(
    mock_settings, mock_whitebox, test_points_file, test_dem_file
):
    """Test successful execution of assess_dem_quality."""

    # Set up extract_raster_values_at_points to add VALUE1 column to points file
    def mock_extract(inputs, points):
        # Read the points file
        points_gdf = gpd.read_file(points)
        # Add a VALUE1 column
        points_gdf["VALUE1"] = points_gdf["Elevation"] + 5  # Simulating some difference
        # Save back to file
        points_gdf.to_file(points)

    mock_whitebox.extract_raster_values_at_points.side_effect = mock_extract

    # Run the function
    result = assess_dem_quality(
        settings=mock_settings,
        wbt=mock_whitebox,
        points_shp_path=test_points_file,
        dem_interp_path=test_dem_file,
        dem_topo_path=None,  # Test with some paths as None
        dem_toporaster_all_path=None,
        dem_stream_burn_path=None,
        point_elev_field="Elevation",
    )

    # Verify results
    assert result is not None
    assert "RMSE" in result.columns
    assert "DEM_Type" in result.columns
    assert len(result) == 1  # Only one DEM was provided

    # Verify the mock was called correctly
    mock_whitebox.extract_raster_values_at_points.assert_called_once()


def test_assess_dem_quality_missing_points(mock_settings, mock_whitebox, test_dem_file):
    """Test assess_dem_quality with missing points file."""

    # Use a non-existent points file
    non_existent_file = Path("non_existent_file.shp")

    result = assess_dem_quality(
        settings=mock_settings,
        wbt=mock_whitebox,
        points_shp_path=non_existent_file,
        dem_interp_path=test_dem_file,
        dem_topo_path=None,
        dem_toporaster_all_path=None,
        dem_stream_burn_path=None,
        point_elev_field="Elevation",
    )

    # Should fail and return None
    assert result is None
    # Verify the extract method was not called
    mock_whitebox.extract_raster_values_at_points.assert_not_called()


def test_assess_dem_quality_extraction_error(
    mock_settings, mock_whitebox, test_points_file, test_dem_file
):
    """Test assess_dem_quality when extraction fails."""

    # Make the extract method raise an exception
    mock_whitebox.extract_raster_values_at_points.side_effect = Exception(
        "Extraction failed"
    )

    result = assess_dem_quality(
        settings=mock_settings,
        wbt=mock_whitebox,
        points_shp_path=test_points_file,
        dem_interp_path=test_dem_file,
        dem_topo_path=None,
        dem_toporaster_all_path=None,
        dem_stream_burn_path=None,
        point_elev_field="Elevation",
    )

    # Should handle the error and return None
    assert result is None
