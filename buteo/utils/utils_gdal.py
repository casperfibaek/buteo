"""
### Utility functions to work with GDAL ###

These functions are used to interact with basic GDAL objects.
"""

# Standard Library
import sys; sys.path.append("../../")
from typing import Optional, Union, List, Any
from warnings import warn
import os

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import utils_path, utils_gdal_translate, utils_aux, utils_base



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


def _get_gdal_memory() -> list:
    """
    Get at list of all active memory layers in GDAL.
    
    Returns
    -------
    list
        A list of all active memory layers in GDAL.
    """
    if hasattr(gdal, "listdir"):
        datasets = [ds.name for ds in gdal.listdir("/vsimem")]
    elif hasattr(gdal, "ReadDir"):
        datasets = ["/vsimem/" + ds for ds in gdal.ReadDir("/vsimem")]
    else:
        warn("Unable to get list of memory datasets.", RuntimeWarning)
        return []

    return datasets


def _clear_gdal_memory() -> None:
    """
    Clears all gdal memory.

    Notes
    -----
    This function is not guaranteed to work.
    It is a best effort attempt to clear all gdal memory.
    """
    memory = _get_gdal_memory()

    for dataset in memory:
        gdal.Unlink(dataset)

    if len(_get_gdal_memory()) != 0:
        for dataset in _get_gdal_memory():
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

    if len(_get_gdal_memory()) != 0:
        print("Failed to clear all GDAL memory.")


def _print_gdal_memory() -> None:
    """ Prints all gdal memory. """
    memory = _get_gdal_memory()

    for dataset in memory:
        print(dataset)



def _check_is_file_valid_dtype(file_path: str) -> bool:
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

    if ext in utils_gdal_translate._get_valid_driver_extensions():
        return True

    return False


def _check_is_valid_raster_dtype(file_path: str) -> bool:
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

    if ext in utils_gdal_translate._get_valid_raster_driver_extensions():
        return True

    return False


def _check_is_valid_vector_dtype(file_path: str) -> bool:
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

    if ext in utils_gdal_translate._get_valid_vector_driver_extensions():
        return True

    return False


def _get_driver_from_path(file_path: str) -> str:
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

    if _check_is_file_valid_dtype(ext):
        return utils_gdal_translate._get_driver_shortname_from_ext(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def _get_vector_driver_from_path(file_path: str) -> str:
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

    if _check_is_valid_vector_dtype(file_path):
        return utils_gdal_translate._get_raster_shortname_from_ext(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def _get_raster_driver_from_path(file_path: str) -> str:
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

    if _check_is_valid_raster_dtype(file_path):
        return utils_gdal_translate._get_raster_shortname_from_ext(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def _check_dataset_in_memory(
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


def _delete_dataset_if_in_memory(
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

    if _check_dataset_in_memory(raster_or_vector):
        if isinstance(raster_or_vector, str):
            gdal.Unlink(raster_or_vector)
        else:
            raster_or_vector.Destroy()
            raster_or_vector = None

            gdal.Unlink(path)

        datasets = _get_gdal_memory()
        if path not in datasets:
            return True

    return False


def delete_if_in_memory_list(
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
        deleted.append(_delete_dataset_if_in_memory(raster_or_vector))

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

    if _delete_dataset_if_in_memory(raster_or_vector):
        return True

    driver_shortname = _get_driver_from_path(raster_or_vector)
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

        if layer.GetFeatureCount() > 0:
            feature_count = layer.GetFeatureCount()

            for feature in range(0, feature_count):
                feature = layer.GetNextFeature()

                if feature.GetGeometryRef() is not None:
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

        return path

    if (dataset_type == "vector" or dataset_type is None) and _check_is_vector(dataset, empty_is_invalid=False):
        if isinstance(dataset, str):
            vector = ogr.Open(dataset, 0)
        elif isinstance(dataset, ogr.DataSource):
            vector = dataset
        else:
            raise RuntimeError(f"Could not read input vector: {vector}")

        path = vector.GetDescription()
        vector = None

        return path

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


def _get_vector_from_geom(geom: ogr.Geometry) -> ogr.DataSource:
    """
    Converts a geometry to a vector.

    Parameters
    ----------
    geom : ogr.Geometry
        The geometry to convert.

    Returns
    -------
    ogr.DataSource
    """
    assert isinstance(geom, ogr.Geometry), "geom must be an ogr.Geometry."

    path = utils_path._get_augmented_path(
        "converted_geom.gpkg",
        add_uuid=True,
        folder="/vsimem/",
    )

    driver = ogr.GetDriverByName(_get_vector_driver_from_path(path))
    vector = driver.CreateDataSource(path)

    layer = vector.CreateLayer("converted_geom", geom.GetSpatialReference(), geom.GetGeometryType())

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geom)

    layer.CreateFeature(feature)
    feature.Destroy()

    return vector


def _get_raster_size(
    raster: Union[gdal.Dataset, str],
):
    """
    Gets the size of a raster (xres, yres).

    Parameters
    ----------
    target : Union[gdal.Dataset, str]
        The target to get the size from.

    target_in_pixels : bool, optional
        If True, the target is in pixels. Default: False.

    Returns
    -------
    Tuple[float, float]
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


def _get_gdalwarp_ram_limit(
    limit_in_mb: Union[str, int] = "auto",
) -> int:
    """
    Converts a RAM limit to a GDALWarp RAM limit.

    Parameters
    ----------
    limit_in_mb : Union[str, int], optional
        The limit in MB. Can be "auto" or a number. Default: "auto".

    Returns
    -------
    int
    """
    assert isinstance(limit_in_mb, (str, int)), "limit must be a string or integer."

    min_ram = 1000000
    limit = min_ram

    if isinstance(limit_in_mb, str):
        if limit_in_mb.lower() == "auto":
            return utils_aux._get_dynamic_memory_limit_bytes(percentage=80.0)
        else:
            if "%" not in limit_in_mb:
                raise ValueError(f"Invalid limit: {limit_in_mb}")

            limit_in_percentage = limit_in_mb.replace("%", "")
            limit_in_percentage = int(limit_in_percentage)

            if limit_in_percentage <= 0 or limit_in_percentage > 100:
                raise ValueError(f"Invalid limit: {limit_in_mb}")

            limit = utils_aux._get_dynamic_memory_limit_bytes(percentage=limit_in_percentage)

            if limit > min_ram:
                return limit

    if limit > min_ram:
        return int(limit_in_mb * (1024 ** 2))

    return min_ram


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

    input_list = band_number if isinstance(band_number, list) else [band_number]
    output_list = []

    if len(input_list) == 0:
        raise ValueError("Band number cannot be 0.")

    if band_count <= 0:
        raise ValueError("Band count cannot be 0.")

    if any([val <= 0 or val > band_count for val in input_list]):
        raise ValueError("Band number cannot be 0 or above the band count.")

    for val in input_list:
        output_list.append(int(val))

    return output_list


# TODO: Verify this function, it looks funky.
def save_dataset_to_disk(
    dataset: Union[gdal.Dataset, ogr.DataSource, str],
    out_path: Union[str, List[str]],
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
) -> Union[str, List[str]]:
    """
    Writes a dataset to disk. Can be a raster or a vector.

    Parameters
    ----------
    dataset : Union[gdal.Dataset, ogr.DataSource, str]
        The dataset to write to disk.

    out_path : Union[str, List[str]]
        The output path or list of output paths.
    
    overwrite : bool, optional
        If True, the output will be overwritten if it already exists. Default: True.

    creation_options : Optional[List[str]], optional
        A list of creation options. Default: None.

    prefix : str, optional
        A prefix to add to the output path. Default: "".

    suffix : str, optional
        A suffix to add to the output path. Default: "".

    add_uuid : bool, optional
        If True, a UUID will be added to the output path. Default: False.

    Returns
    -------
    Union[str, List[str]]
        The output path or list of output paths.
    """
    datasets = utils_base._get_variable_as_list(dataset)
    datasets_paths = _get_path_from_dataset_list(datasets, allow_mixed=True)
    out_paths = utils_path._get_output_path_list(
        datasets_paths,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    options = None

    for index, dataset_ in enumerate(datasets):
        opened_dataset = None
        dataset_type = None

        if _check_is_raster(dataset_):
            options = _get_default_creation_options(creation_options)
            dataset_type = "raster"
            if isinstance(dataset_, str):
                opened_dataset = gdal.Open(dataset_, 0)
            elif isinstance(dataset_, gdal.Dataset):
                opened_dataset = dataset_
            else:
                raise RuntimeError(f"Could not read input raster: {dataset_}")

        elif _check_is_vector(dataset_):
            dataset_type = "vector"
            if isinstance(dataset_, str):
                opened_dataset = ogr.Open(dataset_, 0)
            elif isinstance(dataset_, ogr.DataSource):
                opened_dataset = dataset_
            else:
                raise RuntimeError(f"Could not read input vector: {dataset_}")

        else:
            raise RuntimeError(f"Invalid dataset type: {dataset_}")

        driver_destination = None

        if dataset_type == "raster":
            driver_destination = gdal.GetDriverByName(_get_raster_driver_from_path(out_paths[index]))
        else:
            driver_destination = ogr.GetDriverByName(_get_vector_driver_from_path(out_paths[index]))

        assert driver_destination is not None, "Could not get driver for output dataset."

        utils_path._delete_if_required(out_paths[index], overwrite)

        driver_destination.CreateCopy(
            out_path[index],
            opened_dataset,
            options=options,
        )

    if isinstance(dataset, list):
        return out_paths[0]

    return out_paths


def save_dataset_to_memory(
    dataset: Union[gdal.Dataset, ogr.DataSource, str],
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = True,
) -> Union[str, List[str]]:
    """
    Writes a dataset to memory. Can be a raster or a vector.

    Parameters
    ----------
    dataset : Union[gdal.Dataset, ogr.DataSource, str]
        The dataset to write to memory.

    overwrite : bool, optional
        If True, the output will be overwritten if it already exists. Default: True.

    creation_options : Optional[List[str]], optional
        A list of creation options. Default: None.

    prefix : str, optional
        A prefix to add to the output path. Default: "".

    suffix : str, optional
        A suffix to add to the output path. Default: "".

    add_uuid : bool, optional
        If True, a UUID will be added to the output path. Default: False.

    Returns
    -------
    Union[str, List[str]]
        The output path or list of output paths.
    """
    return save_dataset_to_disk(
        dataset,
        out_path=None,
        overwrite=overwrite,
        creation_options=creation_options,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )


def _parse_input_data(
    input_data: Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]],
    input_type: str,
) -> List[str]:
    """
    Parses the input data to a list of paths.

    Parameters
    ----------
    input_data : Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]]
        The input data to parse.

    input_type : str, optional
        The input type. Can be "raster", "vector" or "mixed". Default: "mixed".

    Returns
    -------
    List[str]
        The list of paths.
    """
    assert input_type in ["raster", "vector"], "Invalid input type."
    assert isinstance(input_data, (gdal.Dataset, ogr.DataSource, str, list)), "Invalid type for input data."

    if isinstance(input_data, str):
        if utils_path._check_is_path_glob(input_data):
            input_data = utils_path._get_paths_from_glob(input_data)
        else:
            input_data = utils_base._get_variable_as_list(input_data)

    elif isinstance(input_data, (gdal.Dataset, ogr.DataSource)):
        input_data = [input_data.GetDescription()]

    # Input is list
    elif isinstance(input_data, list):
        if len(input_data) == 0:
            raise ValueError("Input data cannot be empty.")

        if not all([isinstance(val, (gdal.Dataset, ogr.DataSource, str)) for val in input_data]):
            raise TypeError("Invalid type for input data.")

        input_data = _get_path_from_dataset_list(input_data, allow_mixed=True)

        if input_type == "mixed":
            if not all([_check_is_file_valid_dtype(val) for val in input_data]):
                raise TypeError("Invalid type for input data.")
        elif input_type == "raster":
            if not all([_check_is_raster(val) for val in input_data]):
                raise TypeError("Invalid type for input data.")
        elif input_type == "vector":
            if not all([_check_is_vector(val) for val in input_data]):
                raise TypeError("Invalid type for input data.")

        input_data = [val.GetDescription() if isinstance(val, (gdal.Dataset, ogr.DataSource)) else val for val in input_data]

    else:
        raise TypeError("Invalid type for input data.")

    if not utils_path._check_is_valid_filepath_list(input_data):
        raise ValueError("Invalid input data.")

    if input_type == "raster":
        for val in input_data:
            if not _check_is_raster(val):
                raise TypeError("Invalid type for input data.")
    elif input_type == "vector":
        for val in input_data:
            if not _check_is_vector(val):
                raise TypeError("Invalid type for input data.")
    else:
        for val in input_data:
            if not _check_is_raster_or_vector(val):
                raise TypeError("Invalid type for input data.")

    return input_data


def _parse_output_data(
    input_data: Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]],
    output_data: Optional[Union[str, List[str]]],
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
    change_ext: bool = False,
) -> List[str]:
    """
    Parses the output data to a list of paths.

    Parameters
    ----------
    input_data : List[str]
        The input data.

    output_data : Optional[Union[str, List[str]]]
        The output data.

    prefix : str, optional
        A prefix to add to the output path. Default: "".

    suffix : str, optional
        A suffix to add to the output path. Default: "".

    add_uuid : bool, optional
        If True, a UUID will be added to the output path. Default: False.

    overwrite : bool, optional
        If True, the output will be overwritten if it already exists. Default: True.

    change_ext : bool, optional
        If True, the extension of the output path will be changed to the extension of the input path. Default: False.

    Returns
    -------
    List[str]
        The list of paths.
    """
    assert isinstance(input_data, list), "Invalid type for input data."
    assert isinstance(output_data, (str, list, type(None))), "Invalid type for output data."
    assert isinstance(prefix, str), "Invalid type for prefix."
    assert isinstance(suffix, str), "Invalid type for suffix."
    assert isinstance(add_uuid, bool), "Invalid type for add_uuid."
    assert isinstance(overwrite, bool), "Invalid type for overwrite."
    assert isinstance(change_ext, bool), "Invalid type for change_ext."

    output_data = utils_path._get_output_path_list(
        input_data,
        output_data,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
        change_ext=change_ext,
    )

    return output_data
