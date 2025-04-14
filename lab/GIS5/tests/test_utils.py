import pytest
import geopandas as gpd
from shapely.geometry import Point

from src.utils import find_elevation_field

# Sample data for testing find_elevation_field
@pytest.fixture
def sample_gdf() -> gpd.GeoDataFrame:
    """Creates a sample GeoDataFrame for testing."""
    data = {'colA': [1, 2], 'HOEYDE': [10.5, 20.1], 'geometry': [Point(1, 1), Point(2, 2)]}
    gdf = gpd.GeoDataFrame(data, crs="EPSG:32632") # Example CRS
    return gdf

@pytest.fixture
def sample_gdf_alt_field() -> gpd.GeoDataFrame:
    """Creates a sample GeoDataFrame with a different elevation field name."""
    data = {'colA': [1, 2], 'Elevation': [10.5, 20.1], 'geometry': [Point(1, 1), Point(2, 2)]}
    gdf = gpd.GeoDataFrame(data, crs="EPSG:32632")
    return gdf

@pytest.fixture
def sample_gdf_no_field() -> gpd.GeoDataFrame:
    """Creates a sample GeoDataFrame with no matching elevation field."""
    data = {'colA': [1, 2], 'OtherData': [10.5, 20.1], 'geometry': [Point(1, 1), Point(2, 2)]}
    gdf = gpd.GeoDataFrame(data, crs="EPSG:32632")
    return gdf


def test_find_elevation_field_exact_match(sample_gdf):
    """Test finding the field when it's the first candidate."""
    candidates = ["HOEYDE", "Elevation", "Z_Value"]
    field = find_elevation_field(sample_gdf, candidates)
    assert field == "HOEYDE"

def test_find_elevation_field_second_match(sample_gdf_alt_field):
    """Test finding the field when it's not the first candidate."""
    candidates = ["HOEYDE", "Elevation", "Z_Value"]
    field = find_elevation_field(sample_gdf_alt_field, candidates)
    assert field == "Elevation"

def test_find_elevation_field_no_match(sample_gdf_no_field):
    """Test when no candidate field exists in the GeoDataFrame."""
    candidates = ["HOEYDE", "Elevation", "Z_Value"]
    field = find_elevation_field(sample_gdf_no_field, candidates)
    assert field is None

def test_find_elevation_field_empty_candidates(sample_gdf):
    """Test with an empty list of candidates."""
    candidates = []
    field = find_elevation_field(sample_gdf, candidates)
    assert field is None

def test_find_elevation_field_case_sensitive(sample_gdf):
    """Test if the field matching is case-sensitive (it should be)."""
    candidates = ["hoeyde", "elevation"]
    field = find_elevation_field(sample_gdf, candidates)
    assert field is None # 'HOEYDE' exists, but not 'hoeyde'
