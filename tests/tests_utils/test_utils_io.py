# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import os
from osgeo import gdal, ogr

from buteo.utils import utils_io
from pathlib import Path

# Fixtures
@pytest.fixture
def sample_raster(tmp_path):
    """Create a sample GeoTIFF file"""
    filename = str(tmp_path / "test.tif")
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filename, 10, 10, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    ds = None
    return filename

@pytest.fixture
def sample_vector(tmp_path):
    """Create a sample vector file"""
    filename = str(tmp_path / "test.gpkg")
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(filename)
    layer = ds.CreateLayer('test')
    ds = None
    return filename

def same_path(path1: str, path2: str) -> bool:
    """Check if two paths point to the same location regardless of path separator style."""
    try:
        return Path(path1).resolve() == Path(path2).resolve()
    except Exception:
        return os.path.normpath(path1) == os.path.normpath(path2)

@pytest.fixture
def path_comparison_cases():
    """Fixture providing test cases for path comparison."""
    return [
        ('C:/path/to/file.txt', 'C:\\path\\to\\file.txt'),
        ('/path/to/file.txt', '/path/to/file.txt'),
        ('path/to/file.txt', 'path\\to\\file.txt'),
    ]

@pytest.fixture
def memory_dataset():
    """Create an in-memory dataset"""
    return gdal.GetDriverByName('MEM').Create('', 10, 10, 1, gdal.GDT_Byte)

class TestInputPaths:
    """Test _get_input_paths function"""

    def test_none_input(self):
        """Test handling of None input"""
        with pytest.raises(ValueError):
            utils_io._get_input_paths(None)

    def test_single_string_path(self, sample_raster):
        """Test handling of single string path"""
        paths = utils_io._get_input_paths(sample_raster)

        assert isinstance(paths, list)
        assert len(paths) == 1
        assert same_path(paths[0], sample_raster)

    def test_gdal_dataset(self, sample_raster):
        """Test handling of GDAL dataset"""
        ds = gdal.Open(sample_raster)
        paths = utils_io._get_input_paths(ds)
        assert isinstance(paths, list)
        assert len(paths) == 1
        ds = None

    def test_ogr_datasource(self, sample_vector):
        """Test handling of OGR datasource"""
        ds = ogr.Open(sample_vector)
        paths = utils_io._get_input_paths(ds)
        assert isinstance(paths, list)
        assert len(paths) == 1
        ds = None

    def test_list_of_paths(self, sample_raster, sample_vector):
        """Test handling of list of paths"""
        paths = utils_io._get_input_paths([sample_raster, sample_vector])
        assert isinstance(paths, list)
        assert len(paths) == 2

    def test_invalid_input_type(self):
        """Test handling of invalid input type"""
        with pytest.raises(TypeError):
            utils_io._get_input_paths(123)

    def test_raster_type_validation(self, sample_raster, sample_vector):
        """Test raster type validation"""
        with pytest.raises(TypeError):
            utils_io._get_input_paths(sample_vector, input_type="raster")

    def test_vector_type_validation(self, sample_raster, sample_vector):
        """Test vector type validation"""
        with pytest.raises(TypeError):
            utils_io._get_input_paths(sample_raster, input_type="vector")

class TestOutputPaths:
    """Test _get_output_paths function"""

    def test_in_place_operation(self, sample_raster):
        """Test in-place operation"""
        paths = utils_io._get_output_paths(sample_raster, in_place=True)
        assert same_path(paths[0], sample_raster)

    def test_memory_output(self, sample_raster):
        """Test memory output"""
        paths = utils_io._get_output_paths(sample_raster, output_path=None)
        assert all(path.startswith("/vsimem/") for path in paths)

    def test_directory_output(self, sample_raster, tmp_path):
        """Test directory output"""
        paths = utils_io._get_output_paths(sample_raster, output_path=str(tmp_path))
        expected_path = str(tmp_path / "test.tif")
        assert len(paths) == 1
        assert same_path(paths[0], expected_path)

    def test_single_file_output(self, sample_raster, tmp_path):
        """Test single file output"""
        output_path = str(tmp_path / "output.tif")
        paths = utils_io._get_output_paths(sample_raster, output_path=output_path)
        assert len(paths) == 1
        assert same_path(paths[0], output_path)

    def test_multiple_outputs_mismatch(self, sample_raster):
        """Test error on output path count mismatch"""
        with pytest.raises(ValueError):
            utils_io._get_output_paths([sample_raster, sample_raster], 
                                     output_path=["output.tif"])

    def test_prefix_suffix(self, sample_raster):
        """Test prefix and suffix addition"""
        paths = utils_io._get_output_paths(sample_raster, prefix="pre_", suffix="_post")
        assert all("pre_" in path for path in paths)
        assert all("_post" in path for path in paths)

    def test_uuid_addition(self, sample_raster):
        """Test UUID addition"""
        paths = utils_io._get_output_paths(sample_raster, add_uuid=True)
        assert len(paths) == 1
        assert len(os.path.basename(paths[0])) > len(os.path.basename(sample_raster))

    def test_timestamp_addition(self, sample_raster):
        """Test timestamp addition"""
        paths = utils_io._get_output_paths(sample_raster, add_timestamp=True)
        assert len(paths) == 1
        assert len(os.path.basename(paths[0])) > len(os.path.basename(sample_raster))

    def test_extension_change(self, sample_raster):
        """Test extension change"""
        paths = utils_io._get_output_paths(sample_raster, change_ext="jpg")
        assert all(path.endswith(".jpg") for path in paths)

    def test_invalid_output_path(self):
        """Test invalid output path"""
        with pytest.raises(ValueError):
            utils_io._get_output_paths("input.tif", output_path=123)

    def test_invalid_input_none(self):
        """Test None input"""
        with pytest.raises(ValueError):
            utils_io._get_output_paths(None)

class TestInputValidation:
    """Test input validation"""

    def test_invalid_input_type(self):
        """Test invalid input_type parameter"""
        with pytest.raises(ValueError):
            utils_io._get_input_paths("test.tif", input_type="invalid")

    def test_empty_input_list(self):
        """Test empty input list"""
        with pytest.raises(ValueError):
            utils_io._get_input_paths([])

    def test_mixed_input_types(self, sample_raster, sample_vector):
        """Test mixed input types"""
        paths = utils_io._get_input_paths([sample_raster, sample_vector], input_type="mixed")
        assert len(paths) == 2

    def test_glob_pattern(self, tmp_path):
        """Test glob pattern input"""
        # Create multiple files
        for i in range(3):
            path = tmp_path / f"test_{i}.tif"
            gdal.GetDriverByName('GTiff').Create(str(path), 10, 10, 1, gdal.GDT_Byte)

        paths = utils_io._get_input_paths(str(tmp_path) + "\\test_*.tif:glob") # add :glob to enable globbing
        assert len(paths) == 3

class TestCheckOverwritePolicy:
    """Tests for the _check_overwrite_policy function."""

    def test_overwrite_true_files_exist(self, tmp_path):
        """Test with overwrite=True when files exist."""
        output_file = tmp_path / "output_exists.tif"
        output_file.touch()  # Create the file to simulate existing file

        # Should pass without exception since overwrite is True
        result = utils_io._check_overwrite_policy([str(output_file)], overwrite=True)
        assert result is True

    def test_overwrite_true_files_do_not_exist(self, tmp_path):
        """Test with overwrite=True when files do not exist."""
        output_file = tmp_path / "output_not_exist.tif"

        # Should pass without exception
        result = utils_io._check_overwrite_policy([str(output_file)], overwrite=True)
        assert result is True

    def test_overwrite_false_files_exist(self, tmp_path):
        """Test with overwrite=False when files exist."""
        output_file = tmp_path / "output_exists.tif"
        output_file.touch()  # Create the file to simulate existing file

        # Should raise FileExistsError
        with pytest.raises(FileExistsError):
            utils_io._check_overwrite_policy([str(output_file)], overwrite=False)

    def test_overwrite_false_files_do_not_exist(self, tmp_path):
        """Test with overwrite=False when files do not exist."""
        output_file = tmp_path / "output_not_exist.tif"

        # Should pass without exception
        result = utils_io._check_overwrite_policy([str(output_file)], overwrite=False)
        assert result is True

    def test_overwrite_list_matching_lengths(self, tmp_path):
        """Test with overwrite as a list matching output_paths length."""
        output_file1 = tmp_path / "output1.tif"
        output_file2 = tmp_path / "output2.tif"
        output_file1.touch()  # Create first file

        output_paths = [str(output_file1), str(output_file2)]
        overwrite_flags = [True, False]

        # Should pass since overwrite flags correspond to output paths
        result = utils_io._check_overwrite_policy(output_paths, overwrite_flags)
        assert result is True

    def test_overwrite_list_length_mismatch(self, tmp_path):
        """Test when overwrite list length doesn't match output_paths length."""
        output_file1 = tmp_path / "output1.tif"
        output_file2 = tmp_path / "output2.tif"
        output_paths = [str(output_file1), str(output_file2)]
        overwrite_flags = [True]  # Mismatch in lengths

        with pytest.raises(ValueError):
            utils_io._check_overwrite_policy(output_paths, overwrite_flags)

    def test_overwrite_list_invalid_type(self, tmp_path):
        """Test when overwrite is neither bool nor list."""
        output_file = tmp_path / "output.tif"
        output_paths = [str(output_file)]

        with pytest.raises(TypeError):
            utils_io._check_overwrite_policy(output_paths, overwrite="invalid")

    def test_overwrite_list_invalid_elements(self, tmp_path):
        """Test when overwrite list contains non-boolean values."""
        output_file1 = tmp_path / "output1.tif"
        output_file2 = tmp_path / "output2.tif"
        output_paths = [str(output_file1), str(output_file2)]
        overwrite_flags = [True, "no"]

        with pytest.raises(TypeError):
            utils_io._check_overwrite_policy(output_paths, overwrite_flags)

    def test_output_paths_invalid_type(self):
        """Test when output_paths is not a list."""
        with pytest.raises(TypeError):
            utils_io._check_overwrite_policy("not_a_list", overwrite=True)

    def test_output_paths_empty_list(self):
        """Test with empty output_paths list."""
        result = utils_io._check_overwrite_policy([], overwrite=True)
        assert result is True

    def test_overwrite_with_no_overwrite_needed(self, tmp_path):
        """Test with overwrite=False when no files exist."""
        output_file = tmp_path / "output_not_exist.tif"

        # Should pass since file doesn't exist
        result = utils_io._check_overwrite_policy([str(output_file)], overwrite=False)
        assert result is True