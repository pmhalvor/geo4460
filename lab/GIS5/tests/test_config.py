import pytest
from pathlib import Path

from src.config import settings, AppConfig, PathsConfig


def test_config_loading():
    """Test that the main AppConfig loads without errors."""
    assert settings is not None
    assert isinstance(settings, AppConfig)
    print(
        f"Settings loaded: {settings.model_dump()}"
    )  # Print for debugging in test output


def test_config_paths_exist():
    """Test that essential path configurations exist."""
    assert hasattr(settings, "paths")
    assert isinstance(settings.paths, PathsConfig)
    assert hasattr(settings.paths, "base_dir")
    assert hasattr(settings.paths, "data_dir")
    assert hasattr(settings.paths, "gdb_path")
    assert hasattr(settings.paths, "output_dir")


def test_config_path_types():
    """Test that path attributes are Path objects."""
    assert isinstance(settings.paths.base_dir, Path)
    assert isinstance(settings.paths.data_dir, Path)
    assert isinstance(settings.paths.gdb_path, Path)
    assert isinstance(settings.paths.output_dir, Path)


def test_output_dir_default_name():
    """Check the default name of the output directory."""
    # This assumes the default factory is used
    assert settings.paths.output_dir.name == "output_py"


# Note: Testing the existence of data_dir and gdb_path requires the actual data
# to be present where the test runner expects it. These tests might fail in CI
# or different environments if data isn't available.
# Consider using mocking or fixtures with temporary data for more robust tests.

# Example of how you might test data dir existence if data is guaranteed
# def test_data_dir_exists(monkeypatch):
#     # This test assumes the test runner is in a context where the relative path works
#     # It might be fragile.
#     expected_data_path = Path(__file__).parent.parent / "GIS5_datafiles"
#     # Monkeypatch the factory if needed, or ensure the default works
#     if expected_data_path.exists():
#          assert settings.paths.data_dir.exists()
#     else:
#          pytest.skip(f"Test data directory not found at {expected_data_path}, skipping check.")
