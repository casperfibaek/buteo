# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../")

import pytest
import os
from pathlib import Path
from uuid import UUID
import fnmatch
from datetime import datetime
from osgeo import gdal, ogr

from buteo.utils import utils_path


def same_path(path1: str, path2: str) -> bool:
    """Check if two paths point to the same location regardless of path separator style."""
    try:
        return Path(path1).resolve() == Path(path2).resolve()
    except Exception:
        return os.path.normpath(path1) == os.path.normpath(path2)

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

# Fixtures
@pytest.fixture
def temp_files(tmp_path):
    """Create temporary files and folders for testing"""
    # Create files
    file1 = tmp_path / "test.tif"
    file2 = tmp_path / "test.txt"
    folder1 = tmp_path / "subfolder"
    folder1.mkdir()
    file3 = folder1 / "test.shp"
    
    file1.write_text("test")
    file2.write_text("test")
    file3.write_text("test")
    
    return tmp_path

@pytest.fixture
def vsimem_file():
    """Create a vsimem file for testing"""
    path = "/vsimem/test.tif"
    ds = gdal.GetDriverByName('GTiff').Create(path, 1, 1, 1)
    ds = None
    return path

class TestPathConversion:
    def test_get_unix_path(self):
        """Test unix path conversion"""
        assert utils_path._get_unix_path(r"C:\test\path.tif") == "C:/test/path.tif"
        assert utils_path._get_unix_path("/test/path.tif") == "/test/path.tif"
        
        with pytest.raises(TypeError):
            utils_path._get_unix_path(None)
        with pytest.raises(TypeError):
            utils_path._get_unix_path(123)
        with pytest.raises(ValueError):
            utils_path._get_unix_path("")
            
    def test_get_unix_path_list(self):
        paths = [r"C:\test1.tif", r"C:\test2.tif"]
        result = utils_path._get_unix_path_list(paths)
        assert all("/" in p for p in result)
        
        with pytest.raises(TypeError):
            utils_path._get_unix_path_list(None)
        with pytest.raises(TypeError):
            utils_path._get_unix_path_list("not_a_list")

class TestFileOperations:
    def test_check_file_exists(self, temp_files, vsimem_file):
        """Test file existence checking"""
        assert utils_path._check_file_exists(str(temp_files / "test.tif"))
        assert utils_path._check_file_exists(vsimem_file)
        assert not utils_path._check_file_exists("nonexistent.tif")
        
        with pytest.raises(TypeError):
            utils_path._check_file_exists(None)
        with pytest.raises(ValueError):
            utils_path._check_file_exists("")

    def test_check_file_exists_vsimem(self, vsimem_file):
        """Test vsimem file existence checking"""
        assert utils_path._check_file_exists_vsimem(vsimem_file)
        assert not utils_path._check_file_exists_vsimem("/vsimem/nonexistent.tif")

class TestDirectoryOperations:
    def test_check_dir_exists(self, temp_files):
        """Test directory existence checking"""
        assert utils_path._check_dir_exists(str(temp_files))
        assert utils_path._check_dir_exists(str(temp_files / "subfolder"))
        assert not utils_path._check_dir_exists(str(temp_files / "nonexistent"))
        
        with pytest.raises(TypeError):
            utils_path._check_dir_exists(None)
            
    def test_create_dir_if_not_exists(self, tmp_path):
        """Test directory creation"""
        new_dir = str(tmp_path / "newdir")
        result = utils_path._create_dir_if_not_exists(new_dir)
        assert os.path.exists(new_dir)
        assert isinstance(result, str)
        
        with pytest.raises(TypeError):
            utils_path._create_dir_if_not_exists(None)

class TestPathManipulation:
    def test_get_dir_from_path(self, temp_files):
        """Test directory extraction"""
        path = str(temp_files / "test.tif")
        result = utils_path._get_dir_from_path(path)
        assert same_path(result, str(temp_files) + "/")
        
        with pytest.raises(TypeError):
            utils_path._get_dir_from_path(None)
            
    def test_get_filename_from_path(self):
        """Test filename extraction"""
        tests = [
            ("path/to/file.tif", True, "file.tif"),
            ("path/to/file.tif", False, "file"),
            ("/vsimem/test.tif", True, "test.tif"),
        ]
        
        for path, with_ext, expected in tests:
            assert utils_path._get_filename_from_path(path, with_ext) == expected
            
    def test_get_ext_from_path(self):
        """Test extension extraction"""
        assert utils_path._get_ext_from_path("test.TIF") == "tif"
        assert utils_path._get_ext_from_path("path/to/test.tif") == "tif"
        
        with pytest.raises(RuntimeError):
            utils_path._get_ext_from_path("test")

class TestPathValidation:
    @pytest.mark.parametrize("path,expected", [
        ("/vsimem/test.tif", True),
        ("test.tif", True),
        ("test", False),
        ("", False),
        (None, False),
    ])
    def test_check_is_valid_filepath(self, path, expected):
        """Test filepath validation"""
        if path is None:
            with pytest.raises(TypeError):
                utils_path._check_is_valid_filepath(path)
        else:
            assert utils_path._check_is_valid_filepath(path) == expected
            
    def test_check_is_valid_output_filepath(self, temp_files):
        """Test output filepath validation"""
        new_file = str(temp_files / "new.tif")
        existing_file = str(temp_files / "test.tif")
        
        assert utils_path._check_is_valid_output_filepath(new_file)
        assert utils_path._check_is_valid_output_filepath(existing_file, overwrite=True)
        assert not utils_path._check_is_valid_output_filepath(existing_file, overwrite=False)

class TestPathAugmentation:
    def test_get_augmented_path(self):
        """Test path augmentation"""
        path = "test.tif"
        result = utils_path._get_augmented_path(
            path,
            prefix="pre_",
            suffix="_post",
            add_uuid=True,
            add_timestamp=True
        )

        basename = os.path.basename(result)
        
        assert basename.startswith("pre_")
        assert "_post" in basename
        assert basename.endswith(".tif")
        assert len(basename) > len(os.path.basename(path))
        
        # Test UUID validity
        uuid_part = result.split("_")[2]
        try:
            UUID(uuid_part)
        except ValueError:
            pytest.fail("Invalid UUID in augmented path")
            
        # Test timestamp validity
        timestamp_date = basename.split("_")[3] # 20241125 yyyyMMdd
        timestamp_time = basename.split("_")[4] # 235959 HHmmSS
        try:
            datetime.strptime(timestamp_date, "%Y%m%d")
            datetime.strptime(timestamp_time, "%H%M%S")
        except ValueError:
            pytest.fail("Invalid timestamp in augmented path")

class TestGlobOperations:
    def test_check_is_path_glob(self):
        """Test glob pattern detection"""
        assert utils_path._check_is_path_glob("test*.tif:glob")
        assert not utils_path._check_is_path_glob("test.tif")
        
    def test_get_paths_from_glob(self, temp_files):
        """Test glob pattern resolution"""
        pattern = str(temp_files / "*.tif:glob")
        results = utils_path._get_paths_from_glob(pattern)
        assert len(results) == 1
        assert results[0].endswith(".tif")

class TestCheckIsDir:
    def test_check_is_dir_valid_physical(self, tmp_path):
        """Test _check_is_dir with a valid physical directory"""
        dir_path = str(tmp_path)
        assert utils_path._check_is_dir(dir_path) is True

    def test_check_is_dir_invalid_physical(self, tmp_path):
        """Test _check_is_dir with an invalid physical directory"""
        dir_path = str(tmp_path / "nonexistent")
        assert utils_path._check_is_dir(dir_path) is False

    def test_check_is_dir_valid_vsimem(self):
        """Test _check_is_dir with a valid vsimem directory"""
        dir_path = "/vsimem/test_dir/"
        ds = gdal.GetDriverByName('GTiff').Create(dir_path + "test.tif", 1, 1, 1)
        ds = None
        assert utils_path._check_is_dir(dir_path) is True
        gdal.Unlink(dir_path + "test.tif")

    def test_check_is_dir_invalid_vsimem(self):
        """Test _check_is_dir with an invalid vsimem directory"""
        dir_path = "/vsimem/nonexistent_dir"
        assert utils_path._check_is_dir(dir_path) is False

    def test_check_is_dir_root_vsimem(self):
        """Test _check_is_dir with root vsimem path"""
        assert utils_path._check_is_dir("/vsimem") is True

    def test_check_is_dir_file_in_vsimem(self):
        """Test _check_is_dir with a file path in vsimem"""
        file_path = "/vsimem/test_file.tif"
        ds = gdal.GetDriverByName('GTiff').Create(file_path, 10, 10, 1, gdal.GDT_Byte)
        ds = None
        assert utils_path._check_is_dir(file_path) is False
        gdal.Unlink(file_path)

    def test_check_is_dir_invalid_inputs(self):
        """Test _check_is_dir with invalid inputs"""
        with pytest.raises(TypeError):
            utils_path._check_is_dir(None)
        with pytest.raises(ValueError):
            utils_path._check_is_dir("   ")

    def test_check_is_dir_with_trailing_slash(self, tmp_path):
        """Test _check_is_dir with a trailing slash"""
        dir_path = str(tmp_path) + "/"
        assert utils_path._check_is_dir(dir_path) is True

    def test_check_is_dir_with_trailing_backslash(self, tmp_path):
        """Test _check_is_dir with a trailing backslash"""
        dir_path = str(tmp_path) + "\\"
        assert utils_path._check_is_dir(dir_path) is True

    def test_check_is_dir_file_path(self, temp_files):
        """Test _check_is_dir with a file path"""
        file_path = str(temp_files / "test.tif")
        assert utils_path._check_is_dir(file_path) is False


class TestGetAugmentedPath:
    def test_valid_augmentation(self, tmp_path):
        """Test _get_augmented_path with valid inputs"""
        original_path = str(tmp_path / "test.tif")
        Path(original_path).touch()
        augmented_path = utils_path._get_augmented_path(
            original_path,
            prefix="pre_",
            suffix="_suf",
            change_ext="jpg",
            folder=str(tmp_path),
            add_uuid=True,
            add_timestamp=True
        )

        # get directory of the augmented path
        aug_dir = os.path.dirname(augmented_path)
        assert same_path(str(tmp_path), aug_dir)

        assert "pre_test" in augmented_path
        assert "_suf.jpg" in augmented_path
        assert os.path.exists(os.path.dirname(augmented_path))
        assert augmented_path.endswith(".jpg")

        # Check for UUID and timestamp in the filename
        parts = os.path.basename(augmented_path).split("_")
        assert len(parts) >= 5  # prefix, filename, uuid, timestamp_date, timestamp_time, suffix.ext
        try:
            UUID(parts[2])
        except ValueError:
            pytest.fail("Invalid UUID in augmented path")
        try:
            datetime.strptime(parts[3], "%Y%m%d")
            datetime.strptime(parts[4], "%H%M%S")
        except ValueError:
            pytest.fail("Invalid timestamp in augmented path")

    def test_invalid_path(self):
        """Test _get_augmented_path with an invalid path"""
        with pytest.raises(ValueError):
            utils_path._get_augmented_path(
                "",
                prefix="pre_",
                suffix="_suf"
            )

    def test_invalid_types(self):
        """Test _get_augmented_path with invalid input types"""
        with pytest.raises(TypeError):
            utils_path._get_augmented_path(
                123,  # Invalid path type
                prefix="pre_"
            )
        with pytest.raises(TypeError):
            utils_path._get_augmented_path(
                "test.tif",
                prefix=123  # Invalid prefix type
            )

    def test_change_extension(self):
        """Test _get_augmented_path with extension change"""
        path = "file.txt"
        new_path = utils_path._get_augmented_path(
            path,
            change_ext="md"
        )
        assert new_path.endswith(".md")

    def test_no_modifications(self):
        """Test _get_augmented_path with no modifications"""
        path = "file.txt"
        augmented_path = utils_path._get_augmented_path(path)
        assert augmented_path == utils_path._get_unix_path(os.path.abspath(path))

class TestGetAugmentedPathList:
    def test_valid_augmentation_list(self, tmp_path):
        """Test _get_augmented_path_list with valid inputs"""
        original_paths = [str(tmp_path / f"test_{i}.tif") for i in range(3)]
        for path in original_paths:
            Path(path).touch()
        augmented_paths = utils_path._get_augmented_path_list(
            original_paths,
            prefix="pre_",
            suffix="_suf",
            change_ext="jpg",
            folder=str(tmp_path),
            add_uuid=False,
            add_timestamp=False
        )
        assert len(augmented_paths) == 3
        for aug_path in augmented_paths:
            assert "pre_test_" in aug_path
            assert "_suf.jpg" in aug_path
            assert aug_path.endswith(".jpg")
            assert os.path.exists(os.path.dirname(aug_path))

    def test_empty_path_list(self):
        """Test _get_augmented_path_list with an empty list"""
        with pytest.raises(ValueError):
            utils_path._get_augmented_path_list([])

    def test_invalid_path_in_list(self):
        """Test _get_augmented_path_list with invalid path in list"""
        path_list = ["valid.tif", None]
        with pytest.raises(TypeError):
            utils_path._get_augmented_path_list(path_list)

    def test_invalid_types(self):
        """Test _get_augmented_path_list with invalid input types"""
        with pytest.raises(TypeError):
            utils_path._get_augmented_path_list(
                "not a list",  # Invalid path_list type
                prefix="pre_"
            )
        with pytest.raises(TypeError):
            utils_path._get_augmented_path_list(
                ["test.tif"],
                prefix=123  # Invalid prefix type
            )

    def test_folder_does_not_exist(self):
        """Test _get_augmented_path_list with a nonexistent folder"""
        with pytest.raises(RuntimeError):
            utils_path._get_augmented_path_list(
                ["test.tif"],
                folder="/nonexistent/folder"
            )

class TestGetTempFilePath:
    def test_valid_temp_filepath(self):
        """Test _get_temp_filepath with valid inputs"""
        temp_path = utils_path._get_temp_filepath(
            name="tempfile",
            ext="tif",
            prefix="pre_",
            suffix="_suf",
            add_uuid=True,
            add_timestamp=True
        )
        assert temp_path.startswith("/vsimem/")
        assert "pre_tempfile" in temp_path
        assert "_suf.tif" in temp_path
        parts = os.path.basename(temp_path).split("_")
        assert len(parts) >= 4  # prefix, name, uuid, timestamp, suffix
        try:
            UUID(parts[2])
        except ValueError:
            pytest.fail("Invalid UUID in temp filepath")
        try:
            datetime.strptime(parts[3], "%Y%m%d") # 20241125 yyyyMMdd
            datetime.strptime(parts[4], "%H%M%S") # 235959 HHmmSS
        except ValueError:
            pytest.fail("Invalid timestamp in temp filepath")

    def test_default_parameters(self):
        """Test _get_temp_filepath with default parameters"""
        temp_path = utils_path._get_temp_filepath()
        assert temp_path.startswith("/vsimem/")
        assert temp_path.endswith(".tif")

    def test_invalid_name(self):
        """Test _get_temp_filepath with an invalid name"""
        with pytest.raises(ValueError):
            utils_path._get_temp_filepath(name="")

    def test_invalid_types(self):
        """Test _get_temp_filepath with invalid input types"""
        with pytest.raises(TypeError):
            utils_path._get_temp_filepath(
                name=123,  # Invalid name type
                prefix="pre_"
            )
        with pytest.raises(TypeError):
            utils_path._get_temp_filepath(
                name="temp",
                ext=123  # Invalid ext type
            )

    def test_invalid_extension(self):
        """Test _get_temp_filepath with an invalid extension"""
        with pytest.raises(ValueError):
            utils_path._get_temp_filepath(ext="invalid_ext")

    def test_dataset_name_input(self, sample_raster):
        """Test _get_temp_filepath with a GDAL dataset as name"""
        ds = gdal.Open(sample_raster)
        temp_path = utils_path._get_temp_filepath(
            name=ds,
            add_uuid=False,
            add_timestamp=False
        )
        assert "test" in temp_path
        assert temp_path.endswith(".tif")

    def test_ogr_datasource_name_input(self, sample_vector):
        """Test _get_temp_filepath with an OGR datasource as name"""
        ds = ogr.Open(sample_vector)
        temp_path = utils_path._get_temp_filepath(
            name=ds,
            ext="gpkg",
            add_uuid=False,
            add_timestamp=False
        )
        assert "test" in temp_path
        assert temp_path.endswith(".gpkg")

    def test_handle_filename_collision(self):
        """Test _get_temp_filepath handles filename collisions"""
        # Create a file in /vsimem/ with the same name
        existing_path = "/vsimem/pre_temp.tif"
        ds = gdal.GetDriverByName('GTiff').Create(existing_path, 1, 1, 1)
        ds = None

        temp_path = utils_path._get_temp_filepath(
            name="temp",
            prefix="pre_",
            add_uuid=False,
            add_timestamp=False
        )
        assert temp_path != existing_path
        assert temp_path.startswith("/vsimem/")
        assert temp_path.endswith(".tif")

        # Clean up
        gdal.Unlink(existing_path)
