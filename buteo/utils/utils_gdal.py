"""
### Utility functions to work with GDAL ###

These functions are used to interact with basic GDAL objects.
"""

# Standard Library
import sys; sys.path.append("../../")
from typing import Optional, Union, List, Any, Tuple
from warnings import warn
import os

# External
from osgeo import gdal, ogr
import psutil
import numpy as np

# Internal
from buteo.utils import utils_path, utils_translate



def _get_default_creation_options(
    options: Optional[list] = None,
) -> list:
    """
    Takes a list of GDAL creation options and adds the following defaults to it if their not specified: </br>

    Default options are:
    ```python
    >>> default_options = [
    ...     "TILED=YES",
    ...     "NUM_THREADS=ALL_CPUS",
    ...     "BIGTIFF=IF_SAFER",
    ...     "COMPRESS=LZW",
    ...     "BLOCKXSIZE=256",
    ...     "BLOCKYSIZE=256",
    ... ]
    ```

    Parameters
    ----------
    options : Optional[list], optional
        A list of GDAL creation options, default: None

    Returns
    -------
    list
        A list of GDAL creation options with the defaults added.
    """
    assert isinstance(options, (list, type(None))), "Options must be a list or None."

    if options is None:
        options = []

    internal_options = list(options)

    opt_str = " ".join(internal_options)
    if "TILED" not in opt_str:
        internal_options.append("TILED=YES")

    if "NUM_THREADS" not in opt_str:
        internal_options.append("NUM_THREADS=ALL_CPUS")

    if "BIGTIFF" not in opt_str:
        internal_options.append("BIGTIFF=IF_SAFER")

    if "COMPRESS" not in opt_str:
        internal_options.append("COMPRESS=LZW")

    if "BLOCKXSIZE" not in opt_str:
        internal_options.append("BLOCKXSIZE=256")

    if "BLOCKYSIZE" not in opt_str:
        internal_options.append("BLOCKYSIZE=256")

    return internal_options


def get_gdal_memory() -> list:
    """
    Get at list of all active memory layers in GDAL.
    
    Returns
    -------
    list
        A list of all active memory layers in GDAL.
    """
    if hasattr(gdal, "listdir"):
        datasets = []
        for ds in gdal.listdir("/vsimem"):
            name = ds.name
            if name.startswith("/vsimem"):
                datasets.append(name)
            else:
                datasets.append("/vsimem/" + name)
    elif hasattr(gdal, "ReadDir"):
        datasets = ["/vsimem/" + ds for ds in gdal.ReadDir("/vsimem")]
    else:
        warn("Unable to get list of memory datasets.", RuntimeWarning)
        return []

    return datasets


def clear_gdal_memory() -> None:
    """
    Clears all gdal memory.

    Notes
    -----
    This function is not guaranteed to work.
    It is a best effort attempt to clear all gdal memory.
    """
    memory = get_gdal_memory()

    for dataset in memory:
        gdal.Unlink(dataset)

    if len(get_gdal_memory()) != 0:
        for dataset in get_gdal_memory():
            opened = None

            if _check_is_raster(dataset):
                opened = gdal.Open(dataset)
            elif _check_is_raster(dataset):
                opened = ogr.Open(dataset)
            else:
                print(f"Unable to open dataset: {dataset}")
                continue
            driver = opened.GetDriver()
            driver.Delete(dataset)

            opened = None
            driver = None

    if len(get_gdal_memory()) != 0:
        warn("Failed to clear all GDAL memory.", RuntimeWarning)


def _check_is_valid_ext(ext: str) -> bool:
    """
    Check if a file extension has a valid GDAL or OGR driver.

    Parameters
    ----------
    ext : str
        The file extension to check.

    Returns
    -------
    bool
        True if the file extension is a valid GDAL or OGR driver, False otherwise.
    """
    assert isinstance(ext, str), "ext must be a string."
    if len(ext) == 0:
        return False

    if ext in utils_translate._get_valid_driver_extensions():
        return True

    return False


def _check_is_valid_raster_ext(ext: str) -> bool:
    """
    Check if a file extension has a valid GDAL driver.

    Parameters
    ----------
    ext : str
        The file extension to check.

    Returns
    -------
    bool
        True if the file extension is a valid GDAL Raster driver, False otherwise.
    """
    assert isinstance(ext, str), "ext must be a string."
    if len(ext) == 0:
        return False

    if ext in utils_translate._get_valid_raster_driver_extensions():
        return True

    return False


def _check_is_valid_vector_ext(ext: str) -> bool:
    """
    Check if a file extension has a valid OGR driver.

    Parameters
    ----------
    ext : str
        The file extension to check.

    Returns
    -------
    bool
        True if the file extension is a valid OGR Vector driver, False otherwise.
    """
    assert isinstance(ext, str), "ext must be a string."
    if len(ext) == 0:
        return False

    if ext in utils_translate._get_valid_vector_driver_extensions():
        return True

    return False


def _check_is_file_valid_ext(file_path: str) -> bool:
    """
    Check if a file path has a valid GDAL or OGR driver.

    Parameters
    ----------
    file_path : str
        The file path to check.

    Returns
    -------
    bool
        True if the file path is a valid GDAL or OGR driver, False otherwise.
    """
    assert isinstance(file_path, str), "file_path must be a string."
    assert len(file_path) > 0, "file_path cannot be an empty string."

    ext = utils_path._get_ext_from_path(file_path)

    if ext in utils_translate._get_valid_driver_extensions():
        return True

    return False


def _check_is_file_valid_raster_ext(file_path: str) -> bool:
    """
    Check if a file path has a valid GDAL driver.

    Parameters
    ----------
    file_path : str
        The file path to check.

    Returns
    -------
    bool
        True if the file path is a valid GDAL Raster driver, False otherwise.
    """
    assert isinstance(file_path, str), "file_path must be a string."
    assert len(file_path) > 0, "file_path cannot be an empty string."

    ext = utils_path._get_ext_from_path(file_path)

    if ext in utils_translate._get_valid_raster_driver_extensions():
        return True

    return False


def _check_is_file_valid_vector_ext(file_path: str) -> bool:
    """
    Check if a file path has a valid OGR driver.

    Parameters
    ----------
    file_path : str
        The file path to check.

    Returns
    -------
    bool
        True if the file path is a valid OGR Vector driver, False otherwise.
    """
    assert isinstance(file_path, str), "file_path must be a string."
    assert len(file_path) > 0, "file_path cannot be an empty string."

    ext = utils_path._get_ext_from_path(file_path)

    if ext in utils_translate._get_valid_vector_driver_extensions():
        return True

    return False


def _get_default_driver_raster() -> gdal.Driver:
    """
    Get the default GDAL raster driver.

    Returns
    -------
    gdal.Driver
        The default GDAL raster driver.
    """
    return gdal.GetDriverByName("GTiff")


def _get_default_driver_vector() -> ogr.Driver:
    """
    Get the default OGR vector driver.

    Returns
    -------
    ogr.Driver
        The default OGR vector driver.
    """
    return ogr.GetDriverByName("GPKG")


def _get_driver_name_from_path(file_path: str) -> str:
    """
    Convert a file path to a GDAL or OGR driver ShortName (e.g. "GTiff" for "new_york.tif")

    Parameters
    ----------
    file_path : str
        The file path to convert.

    Returns
    -------
    str
        The GDAL or OGR driver ShortName.
    """
    assert isinstance(file_path, str), "file_path must be a string."
    assert len(file_path) > 0, "file_path cannot be an empty string."

    ext = utils_path._get_ext_from_path(file_path)

    if _check_is_file_valid_ext(file_path):
        return utils_translate._get_driver_shortname_from_ext(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def _get_vector_driver_name_from_path(file_path: str) -> str:
    """
    Convert a file path to an OGR driver ShortName (e.g. "FlatGeoBuf" for "new_york.fgb")

    Parameters
    ----------
    file_path : str
        The file path to convert.

    Returns
    -------
    str
        The OGR driver ShortName.
    """
    assert isinstance(file_path, str), "file_path must be a string."
    assert len(file_path) > 0, "file_path cannot be an empty string."

    ext = utils_path._get_ext_from_path(file_path)

    if _check_is_file_valid_vector_ext(file_path):
        return utils_translate._get_vector_shortname_from_ext(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def _get_raster_driver_name_from_path(file_path: str) -> str:
    """
    Convert a file path to a GDAL driver ShortName (e.g. "GTiff" for "new_york.tif")

    Parameters
    ----------
    file_path : str
        The file path to convert.

    Returns
    -------
    str
        The GDAL driver ShortName.
    """
    assert isinstance(file_path, str), "file_path must be a string."
    assert len(file_path) > 0, "file_path cannot be an empty string."

    ext = utils_path._get_ext_from_path(file_path)

    if _check_is_file_valid_raster_ext(file_path):
        return utils_translate._get_raster_shortname_from_ext(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def _check_is_dataset_in_memory(
    raster_or_vector: Union[str, gdal.Dataset, ogr.DataSource],
) -> bool:
    """
    Check if vector is in memory

    Parameters
    ----------
    raster_or_vector : Union[str, gdal.Dataset, ogr.DataSource]
        The raster or vector to check.
    
    Returns
    -------
    bool
        True if the raster or vector is in memory, False otherwise.
    """
    assert isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)), "raster_or_vector must be a string, gdal.Dataset, or ogr.DataSource."

    if isinstance(raster_or_vector, str):
        if utils_path._check_file_exists_vsimem(raster_or_vector):
            return True

        return False

    elif isinstance(raster_or_vector, (gdal.Dataset, ogr.DataSource)):
        driver = raster_or_vector.GetDriver()
        driver_short_name = None
        try:
            driver_short_name = driver.GetName()
        except AttributeError:
            driver_short_name = driver.ShortName

        if driver_short_name in ["MEM", "Memory", "VirtualMem", "VirtualOGR", "Virtual"]:
            return True

        path = raster_or_vector.GetDescription()
        if utils_path._check_file_exists_vsimem(path):
            return True

        return False

    else:
        raise TypeError("vector_or_raster must be a string, ogr.DataSource, or gdal.Dataset")


def delete_dataset_if_in_memory(
    raster_or_vector: Union[str, gdal.Dataset, ogr.DataSource],
) -> bool:
    """
    Delete raster or vector if it is in memory

    Parameters
    ----------
    raster_or_vector : Union[str, gdal.Dataset, ogr.DataSource]
        The raster or vector to delete.

    Returns
    -------
    bool
        True if the raster or vector was deleted, False otherwise.
    """
    if not isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)):
        return False

    if not isinstance(raster_or_vector, str):
        path = _get_path_from_dataset(raster_or_vector)
    else:
        path = raster_or_vector

    if _check_is_dataset_in_memory(raster_or_vector):
        if isinstance(raster_or_vector, str):
            gdal.Unlink(raster_or_vector)
        else:
            raster_or_vector.Destroy()
            raster_or_vector = None

            gdal.Unlink(path)

        datasets = get_gdal_memory()
        if path not in datasets:
            return True

    return False


def delete_dataset_if_in_memory_list(
    list_of_raster_or_vectors: List[Union[str, gdal.Dataset, ogr.DataSource]],
) -> bool:
    """
    Deletes a list of raster or vector if they are in memory

    Parameters
    ----------
    list_of_raster_or_vectors : List[Union[str, gdal.Dataset, ogr.DataSource]]
        A list of rasters or vectors to delete.

    Returns
    -------
    bool
        True if any of the rasters or vectors were deleted, False otherwise.
    """
    assert isinstance(list_of_raster_or_vectors, list), "list_of_raster_or_vectors must be a list."

    if len(list_of_raster_or_vectors) == 0:
        return True

    deleted = []
    for raster_or_vector in list_of_raster_or_vectors:
        deleted.append(delete_dataset_if_in_memory(raster_or_vector))

    if all(deleted):
        return True

    return False


def _delete_raster_or_vector(
    raster_or_vector: Union[str, gdal.Dataset, ogr.DataSource],
) -> bool:
    """
    Delete raster or vector. Can be used on both in memory and on disk.

    Parameters
    ----------
    raster_or_vector : Union[str, gdal.Dataset, ogr.DataSource]
        The raster or vector to delete.

    Returns
    -------
    bool
        True if the raster or vector was deleted, False otherwise.
    """
    assert isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)), "raster_or_vector must be a string, gdal.Dataset, or ogr.DataSource."

    if delete_dataset_if_in_memory(raster_or_vector):
        return True

    driver_shortname = _get_driver_name_from_path(raster_or_vector)
    driver = gdal.GetDriverByName(driver_shortname)
    driver.Delete(raster_or_vector)

    if not utils_path._check_file_exists(raster_or_vector):
        return True

    return False


def _check_is_raster_empty(
    raster: gdal.Dataset,
) -> bool:
    """
    Check if a raster has bands or zero width and zero height.

    Parameters
    ----------
    raster : gdal.Dataset
        The raster to check.

    Returns
    -------
    bool
        True if the raster has bands and width and height greater than zero, False otherwise.
    """
    assert isinstance(raster, gdal.Dataset), "raster must be a gdal.Dataset."

    if raster.RasterCount == 0:
        return True

    if raster.RasterXSize == 0 or raster.RasterYSize == 0:
        return True

    return False


def _check_is_vector_empty(
    vector: ogr.DataSource,
) -> bool:
    """
    Check if a vector has features with geometries

    Parameters
    ----------
    vector : ogr.DataSource
        The vector to check.

    Returns
    -------
    bool
        True if the vector has features with geometries, False otherwise.
    """
    assert isinstance(vector, ogr.DataSource), "vector must be an ogr.DataSource."

    layer_count = vector.GetLayerCount()

    if layer_count == 0:
        return True

    for layer in range(0, layer_count):
        layer = vector.GetLayerByIndex(layer)
        layer.ResetReading()

        if layer.GetFeatureCount() > 0:
            feature_count = layer.GetFeatureCount()

            for feature in range(feature_count):
                feature = layer.GetNextFeature()

                # TODO: Expand this to check geometry.
                if feature is not None:
                    return False

    return True


def _check_is_raster(
    potential_raster: Any,
    empty_is_invalid: bool = True,
) -> bool:
    """
    Checks if a variable is a valid raster.

    Parameters
    ----------
    potential_raster : Union[str, gdal.Dataset]
        The variable to check.

    empty_is_invalid : bool, optional
        If True, an empty raster is considered invalid. The default is True.

    Returns
    -------
    bool
        True if the variable is a valid raster, False otherwise.
    """
    if isinstance(potential_raster, str):
        if not utils_path._check_file_exists(potential_raster):
            return False

        try:
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            opened = gdal.Open(potential_raster, 0)
            gdal.PopErrorHandler()
        except RuntimeError:
            return False

        if opened is None:
            return False

        if empty_is_invalid and _check_is_raster_empty(opened):
            return False

        opened = None

        return True

    if isinstance(potential_raster, gdal.Dataset):

        if empty_is_invalid and _check_is_raster_empty(potential_raster):
            return False

        return True

    return False


def _check_is_raster_list(
    potential_raster_list,
    empty_is_invalid=True,
) -> bool:
    """
    Checks if a list of variables are full of valid rasters.

    Parameters
    ----------
    potential_raster_list : List[Union[str, gdal.Dataset]]
        The list of variables to check.

    empty_is_invalid : bool, optional
        If True, an empty raster is considered invalid. Default: True.
    """
    if not isinstance(potential_raster_list, list):
        return False

    if len(potential_raster_list) == 0:
        return False

    for element in potential_raster_list:
        if not _check_is_raster(element, empty_is_invalid=empty_is_invalid):
            return False

    return True


def _check_is_vector(
    potential_vector: List[Any],
    empty_is_invalid: bool = True,
) -> bool:
    """
    Checks if a variable is a valid vector.

    Parameters
    ----------
    potential_vector : Union[str, ogr.DataSource]
        The variable to check.

    empty_is_invalid : bool, optional
        If True, an empty vector is considered invalid. The default is True.

    Returns
    -------
    bool
        True if the variable is a valid vector, False otherwise.
    """
    if isinstance(potential_vector, ogr.DataSource):

        if empty_is_invalid and _check_is_vector_empty(potential_vector):
            print(f"Vector: {potential_vector} was empty.")

            return False

        return True

    if isinstance(potential_vector, str):
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        opened = ogr.Open(potential_vector, 0)
        if opened is None:
            extension = os.path.splitext(potential_vector)[1][1:]

            if extension == "memory" or "mem":
                driver = ogr.GetDriverByName("Memory")
                opened = driver.Open(potential_vector)

        gdal.PopErrorHandler()

        if isinstance(opened, ogr.DataSource):

            if empty_is_invalid and _check_is_vector_empty(opened):
                return False

            return True

    return False


def _check_is_vector_list(
    potential_vector_list: List[Any],
    empty_is_invalid=True,
) -> bool:
    """
    Checks if a variable is a valid list of vectors.

    Parameters
    ----------
    potential_vector_list : List[any]
        The variable to check.

    empty_is_invalid : bool, optional
        If True, an empty vector is considered invalid. Default: True.

    Returns
    -------
    bool
        True if the variable is a valid list of vectors, False otherwise.
    """
    if not isinstance(potential_vector_list, list):
        return False

    if len(potential_vector_list) == 0:
        return False

    for element in potential_vector_list:
        if not _check_is_vector(element, empty_is_invalid=empty_is_invalid):
            return False

    return True


def _check_is_raster_or_vector(
    potential_raster_or_vector: Any,
    empty_is_invalid: bool = True,
) -> bool:
    """
    Checks if a variable is a valid raster or vector.

    Parameters
    ----------
    potential_raster_or_vector : Union[str, gdal.Dataset, ogr.DataSource]
        The variable to check.

    empty_is_invalid : bool, optional
        If True, an empty raster or vector is considered invalid. The default is True.

    Returns
    -------
    bool
        True if the variable is a valid raster or vector, False otherwise.
    """
    if _check_is_raster(potential_raster_or_vector, empty_is_invalid=empty_is_invalid):
        return True

    if _check_is_vector(potential_raster_or_vector, empty_is_invalid=empty_is_invalid):
        return True

    return False


def _check_is_raster_or_vector_list(
    potential_raster_or_vector_list: List[Any],
    empty_is_invalid: bool = True,
) -> bool:
    """
    Checks if a variable is a valid list of rasters or vectors.

    Parameters
    ----------
    potential_raster_or_vector_list : List[Union[str, gdal.Dataset, ogr.DataSource]]
        The variable to check.

    empty_is_invalid : bool, optional
        If True, an empty raster or vector is considered invalid. Default: True.

    Returns
    -------
    bool
        True if the variable is a valid list of rasters or vectors, False otherwise.
    """
    if not isinstance(potential_raster_or_vector_list, list):
        return False

    if len(potential_raster_or_vector_list) == 0:
        return False

    for element in potential_raster_or_vector_list:
        if not _check_is_raster_or_vector(element, empty_is_invalid=empty_is_invalid):
            return False

    return True


def _get_path_from_dataset(
    dataset: Union[str, gdal.Dataset, ogr.DataSource],
    dataset_type: Optional[bool] = None,
) -> str:
    """
    Gets the path from a datasets. Can be vector or raster, string or opened.

    Parameters
    ----------
    dataset : Union[str, gdal.Dataset, ogr.DataSource]
        The dataset to get the path from.

    dataset_type : Optional[bool], optional
        The type of the dataset. Can be "raster", "vector" or None. If None, the type is guessed. Default: None.
    """
    if (dataset_type == "raster" or dataset_type is None) and _check_is_raster(dataset, empty_is_invalid=False):
        if isinstance(dataset, str):
            raster = gdal.Open(dataset, 0)
        elif isinstance(dataset, gdal.Dataset):
            raster = dataset
        else:
            raise RuntimeError(f"Could not read input raster: {raster}")

        path = raster.GetDescription()
        raster = None

        return utils_path._get_unix_path(path)

    if (dataset_type == "vector" or dataset_type is None) and _check_is_vector(dataset, empty_is_invalid=False):
        if isinstance(dataset, str):
            vector = ogr.Open(dataset, 0)
        elif isinstance(dataset, ogr.DataSource):
            vector = dataset
        else:
            raise RuntimeError(f"Could not read input vector: {vector}")

        path = vector.GetDescription()
        vector = None

        return utils_path._get_unix_path(path)

    raise ValueError("The dataset is not a raster or vector.")


def _get_path_from_dataset_list(
    datasets: List[Union[str, gdal.Dataset, ogr.DataSource]],
    allow_mixed: bool = False,
    dataset_type: Optional[bool] = None,
):
    """
    Gets the paths from a list of datasets.

    Parameters
    ----------
    datasets : List[Union[str, gdal.Dataset, ogr.DataSource]]
        The datasets to get the paths from.

    allow_mixed : bool, optional
        If True, vectors and rasters can be mixed. Default: False.

    dataset_type : Optional[bool], optional
        The type of the datasets. Can be "raster", "vector" or None. If None, the type is guessed. Default: None.

    Returns
    -------
    List[str]
        The paths of the datasets.
    """
    assert isinstance(datasets, list), "The datasets must be a list."
    assert isinstance(dataset_type, (str, type(None))), "The dataset_type must be 'raster', 'vector', or None."

    rasters = False
    vectors = False

    outputs = []
    for dataset in datasets:
        if (dataset_type == "raster" or dataset_type is None) and _check_is_raster(dataset, empty_is_invalid=False):
            dataset_type = "raster"
            rasters = True
        elif (dataset_type == "vector" or dataset_type is None) and _check_is_vector(dataset, empty_is_invalid=False):
            dataset_type = "vector"
            vectors = True
        else:
            raise ValueError(f"The dataset is not a raster or vector. {dataset}")

        if rasters and vectors and not allow_mixed:
            raise ValueError("vectors and rasters are mixed.")

        outputs.append(_get_path_from_dataset(dataset, dataset_type=dataset_type))

    return outputs


def _get_raster_size(
    raster: Union[gdal.Dataset, str],
) -> Tuple[float, float]:
    """
    Gets the size of a raster (xres, yres).

    Parameters
    ----------
    target : Union[gdal.Dataset, str]
        The target to get the size from.

    Returns
    -------
    Tuple[float, float] - xres, yres
    """
    assert _check_is_raster(raster), "The target must be a raster."

    try:
        reference = raster if isinstance(raster, gdal.Dataset) else gdal.Open(raster, 0)
    except RuntimeError:
        raise RuntimeError(f"Could not read input raster: {raster}") from None

    transform = reference.GetGeoTransform()
    reference = None

    x_res = transform[1]
    y_res = abs(transform[5])

    return x_res, y_res


def _get_dynamic_memory_limit(
    proportion: float = 0.8,
    *,
    min_mb: int = 100,
    max_mb: Optional[int] = None,
    available: bool = False,
) -> int:
    """
    Returns a dynamic memory limit taking into account total memory and CPU cores.
    The return is in mbytes. For GDAL.

    The value is interpreted as being in megabytes if the value is less than 10000. For values >=10000, this is interpreted as bytes.

    Parameters
    ----------
    percentage : float, optional.
        The percentage of the total memory to use. Default: 0.8.
    
    min_mb : int, optional.
        The minimum number of megabytes to be returned. Default: 100.
    
    available : bool, optional.
        If True, consider available memory instead of total memory. Default: False.

    Returns
    -------
    int
        The dynamic memory limit in bytes.
    """
    assert isinstance(proportion, (int, float)), "percentage must be an integer."
    assert isinstance(min_mb, int), "min_mb must be an integer."
    assert isinstance(available, bool), "available must be a boolean."
    assert min_mb > 0, "min_mb must be > 0."
    assert proportion > 0.0 and proportion <= 1.0, "percentage must be > 0 and <= 1."

    if available:
        dyn_limit = np.rint(
            (psutil.virtual_memory().available * proportion)  / (1024 ** 2),
        )
    else:
        dyn_limit = np.rint(
            (psutil.virtual_memory().total * proportion)  / (1024 ** 2),
        )

    if dyn_limit < min_mb:
        dyn_limit = min_mb

    if max_mb is not None:
        if dyn_limit > max_mb:
            dyn_limit = max_mb

    # GDALWarpMemoryLimit() expects the value in bytes if it is >= 10000
    if dyn_limit > 10000:
        dyn_limit = dyn_limit * (1024 ** 2)

    return int(dyn_limit)


def _convert_to_band_list(
    band_number: Union[int, List[int]],
    band_count: int,
) -> List[int]:
    """
    Converts a band number or list of band numbers to a list of band numbers.

    Parameters
    ----------
    band_number : Union[int, List[int]]
        The band number or list of band numbers to convert.

    band_count : int
        The number of bands in the raster.

    Returns
    -------
    List[int]
        The list of band numbers.
    """
    if not isinstance(band_number, (int, list)):
        raise TypeError(f"Invalid type for band: {type(band_number)}")
    if not isinstance(band_count, int):
        raise TypeError(f"Invalid type for band_count: {type(band_count)}")

    if band_number == -1:
        band_number = list(range(1, band_count + 1))

    input_list = band_number if isinstance(band_number, list) else [band_number]
    output_list = []

    if band_count <= 0:
        raise ValueError("Band count cannot be 0.")

    if len(input_list) == 0:
        raise ValueError("Band number cannot be 0.")

    if any([val <= 0 or val > band_count for val in input_list]):
        raise ValueError("Band number cannot be 0 or above the band count.")

    for val in input_list:
        output_list.append(int(val))

    return output_list
