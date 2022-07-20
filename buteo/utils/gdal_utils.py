"""
### Utility functions to work with GDAL ###

These functions are used to interact with basic GDAL objects.

TODO:
    * Should file_path_lists be able to handle mixed inputs?
    * Make delete_raster_or_vector and the like accept a list of file paths?
"""

# Standard Library
import sys; sys.path.append("../../")
import os

# External
import numpy as np
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import core_utils, gdal_enums



def default_creation_options(options=None):
    """
    Takes a list of GDAL creation options and adds the following defaults to it if their not specified: </br>

    * `"TILED=YES"`
    * `"NUM_THREADS=ALL_CPUS"`
    * `"BIGG_TIF=YES"`
    * `"COMPRESS=LZW"`

    If any of the options are already specified, they are not added.

    ## Kwargs:
    `options` (_list_/None): The GDAL creation options to add to. (Default: **None**)

    ## Returns:
    (_list_): A list containing the default values.
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
        internal_options.append("BIGTIFF=YES")

    if "COMPRESS" not in opt_str:
        internal_options.append("COMPRESS=LZW")

    return internal_options


def is_valid_datatype(file_path):
    """
    Check if a file path has a valid GDAL or OGR driver.

    ## Args:
    `file_path` (_str_): The file path to check.

    ## Returns:
    (_bool_): **True** if the file path is a valid GDAL or OGR driver, **False** otherwise.
    """
    assert isinstance(file_path, str), "file_path must be a string."

    ext = core_utils.path_to_ext(file_path)

    if ext in gdal_enums.get_valid_driver_extensions():
        return True

    return False


def is_valid_raster_datatype(file_path):
    """
    Check if a file path has a valid GDAL driver.

    ## Args:
    `file_path` (_str_): The file path to check.

    ## Returns:
    (_bool_): **True** if the file path is a valid GDAL Raster driver, **False** otherwise.
    """
    assert isinstance(file_path, str), "file_path must be a string."

    ext = core_utils.path_to_ext(file_path)

    if ext in gdal_enums.get_valid_raster_driver_extensions():
        return True

    return False


def is_valid_vector_datatype(file_path):
    """
    Check if a file path has a valid OGR driver.

    ## Args:
    `file_path` (_str_): The file path to check.

    ## Returns:
    (_bool_): **True** if the file path is a valid OGR Vector driver, **False** otherwise.
    """
    assert isinstance(file_path, str), "file_path must be a string."

    ext = core_utils.path_to_ext(file_path)

    if ext in gdal_enums.get_valid_vector_driver_extensions():
        return True

    return False


def path_to_driver(file_path):
    """
    Convert a file path to a GDAL or OGR driver ShortName (e.g. "GTiff" for "new_york.tif")

    ## Args:
    `file_path` (_str_): The file path to convert.

    ## Returns:
    (_str_): The GDAL or OGR driver ShortName.

    ## Raises:
    `ValueError`: If the file path is not a valid GDAL or OGR driver.
    """
    assert isinstance(file_path, str), "file_path must be a string."

    ext = core_utils.path_to_ext(file_path)

    if is_valid_datatype(file_path):
        return gdal_enums.convert_extension_to_driver(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def path_to_driver_vector(file_path):
    """
    Convert a file path to an OGR driver ShortName (e.g. "FlatGeoBuf" for "new_york.fgb")

    ## Args:
    `file_path` (_str_): The file path to convert.

    ## Returns:
    (_str_): The OGR driver ShortName.

    ## Raises:
    `ValueError`: If the file path is not a valid OGR driver.
    """
    assert isinstance(file_path, str), "file_path must be a string."

    ext = core_utils.path_to_ext(file_path)

    if is_valid_vector_datatype(file_path):
        return gdal_enums.convert_extension_to_driver(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def path_to_driver_raster(file_path):
    """
    Convert a file path to a GDAL driver ShortName (e.g. "GTiff" for "new_york.tif")

    ## Args:
    `file_path` (_str_): The file path to convert.

    ## Returns:
    (_str_): The GDAL driver ShortName.

    ## Raises:
    `ValueError`: If the file path is not a valid GDAL driver.
    """
    assert isinstance(file_path, str), "file_path must be a string."

    ext = core_utils.path_to_ext(file_path)

    if is_valid_raster_datatype(file_path):
        return gdal_enums.convert_extension_to_driver(ext)

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def is_in_memory(raster_or_vector):
    """
    Check if vector is in memory

    ## Args:
    `raster_or_vector` (_str_/_gdal.Dataset_/ogr.DataSource): The vector or raster to check.

    ## Returns:
    (_bool_): **True** if the vector is in memory, **False** otherwise.
    """
    assert isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)), "raster_or_vector must be a string, gdal.Dataset, or ogr.DataSource."

    if isinstance(raster_or_vector, str):
        if raster_or_vector.startswith("/vsimem/"):
            return True

        return False

    elif isinstance(raster_or_vector, (gdal.Dataset, ogr.DataSource)):
        driver_short_name = raster_or_vector.GetDriver().ShortName
        if driver_short_name== "MEM":
            return True

        if driver_short_name == "Memory":
            return True

        if driver_short_name == "VirtualMem":
            return True

        if driver_short_name == "VirtualOGR":
            return True

        if raster_or_vector.GetDescription().startswith("/vsimem/"):
            return True

        return False

    else:
        raise TypeError("vector_or_raster must be a string, ogr.DataSource, or gdal.Dataset")


def delete_if_in_memory(raster_or_vector):
    """
    Delete raster or vector if it is in memory

    ## Args:
    `raster_or_vector` (_str_/_gdal.Dataset_/_ogr.DataSource_): The vector or raster to check.

    ## Returns:
    (_bool_): **True** if the vector is deleted, **False** otherwise.
    """
    assert isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)), "raster_or_vector must be a string, gdal.Dataset, or ogr.DataSource."

    if is_in_memory(raster_or_vector):
        if isinstance(raster_or_vector, str):
            gdal.Unlink(raster_or_vector)
        else:
            raster_or_vector.Destroy()

        return True

    return False


def delete_raster_or_vector(raster_or_vector):
    """
    Delete raster or vector. Can be used on both in memory and on disk.

    ## Args:
    `raster_or_vector` (_str_/_gdal.Dataset_/_ogr.DataSource_): The vector or raster to check.

    ## Returns:
    (_bool_): **True** if the file is deleted, **False** otherwise.
    """
    assert isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)), "raster_or_vector must be a string, gdal.Dataset, or ogr.DataSource."

    if delete_if_in_memory(raster_or_vector):
        return True

    driver_shortname = path_to_driver(raster_or_vector)
    driver = gdal.GetDriverByName(driver_shortname)
    driver.Delete(raster_or_vector)

    if not core_utils.file_exists(raster_or_vector):
        return True

    return core_utils.delete_file(raster_or_vector)


def is_raster_empty(raster):
    """
    Check if a raster has bands or zero width and zero height.

    ## Args:
    `raster` (_gdal.Dataset_): The raster to check.

    ## Returns:
    (_bool_): **True** if the raster has bands, **False** otherwise.
    """
    assert isinstance(raster, gdal.Dataset), "raster must be a gdal.Dataset."

    if raster.RasterCount == 0:
        return True

    if raster.RasterXSize == 0 or raster.RasterYSize == 0:
        return True

    return False


def is_vector_empty(vector):
    """
    Check if a vector has features with geometries

    ## Args:
    `vector` (_ogr.DataSource_): The vector to check.

    ## Returns:
    (_bool_): **True** if the vector has features, **False** otherwise.
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
                feature = layer.GetFeature(feature)

                if feature.GetGeometryRef() is not None:
                    return False

    return True


def get_gdal_memory():
    """ Get at list of all active memory layers in GDAL. """
    datasets = [ds.name for ds in gdal.listdir('/vsimem')]
    return datasets


def clear_gdal_memory():
    """ Clears all gdal memory. """
    memory = get_gdal_memory()

    for dataset in memory:
        gdal.Unlink(dataset)


def gdal_print_memory():
    """ Prints all gdal memory. """
    memory = get_gdal_memory()

    for dataset in memory:
        print(dataset)


def is_path_in_memory(path):
    """
    Check if a path is in memory.

    ## Args:
    `path` (_str_): The path to the file. </br>

    ## Returns:
    (_bool_): **True** if the path is in memory, **False** otherwise.
    """
    core_utils.is_valid_file_path(path)

    if path.startswith("/vsimem"):
        return True

    return False


def is_raster(potential_raster, *, empty_is_invalid=True):
    """Checks if a variable is a valid raster.

    ## Args:
    `potential_raster` (_any_): The variable to check.

    ## Kwargs:
    `empty_is_invalid` (_bool_): If **True**, an empty raster is considered invalid. (Default: **True**)

    ## Returns:
    (_bool_): **True** if the variable is a valid raster, **False** otherwise.
    """

    if isinstance(potential_raster, str):
        if not core_utils.file_exists(potential_raster) and not is_path_in_memory(potential_raster):
            return False

        try:
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            opened = gdal.Open(potential_raster, 0)
            gdal.PopErrorHandler()
        except Exception:
            return False

        if opened is None:
            return False

        if empty_is_invalid and is_raster_empty(opened):
            return False

        opened = None

        return True

    if isinstance(potential_raster, gdal.Dataset):

        if empty_is_invalid and is_raster_empty(potential_raster):
            return False

        return True

    return False


def is_raster_list(potential_raster_list, *, empty_is_invalid=True):
    """
    Checks if a variable is a valid list of rasters.

    ## Args:
    `potential_raster_list` (_any_): The variable to check.

    ## Kwargs:
    `empty_is_invalid` (_bool_): If **True**, an empty raster is considered invalid. (Default: **True**)
    """
    if not isinstance(potential_raster_list, list):
        return False

    if len(potential_raster_list) == 0:
        return False

    for element in potential_raster_list:
        if not is_raster(element, empty_is_invalid=empty_is_invalid):
            return False

    return True


def is_vector(potential_vector, empty_is_invalid=True):
    """
    Checks if a variable is a valid vector.

    ## Args:
    `potential_vector` (_any_): The variable to check.

    ## Kwargs:
    `empty_is_invalid` (_bool_): If **True**, an empty vector is considered invalid. (Default: **True**)

    ## Returns:
    (_bool_): **True** if the variable is a valid vector, **False** otherwise.
    """
    if isinstance(potential_vector, ogr.DataSource):

        if empty_is_invalid and is_vector_empty(potential_vector):
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

            if empty_is_invalid and is_vector_empty(opened):
                return False

            return True

    return False


def is_vector_list(potential_vector_list, *, empty_is_invalid=True):
    """
    Checks if a variable is a valid list of vectors.

    ## Args:
    `potential_vector_list` (_any_): The variable to check.

    ## Kwargs:
    `empty_is_invalid` (_bool_): If **True**, an empty vector is considered invalid. (Default: **True**)
    """
    if not isinstance(potential_vector_list, list):
        return False

    if len(potential_vector_list) == 0:
        return False

    for element in potential_vector_list:
        if not is_vector(element, empty_is_invalid=empty_is_invalid):
            return False

    return True


def is_raster_or_vector(potential_raster_or_vector, *, empty_is_invalid=True):
    """
    Checks if a variable is a valid raster or vector.

    ## Args:
    `potential_raster_or_vector` (_any_): The variable to check.

    ## Kwargs:
    `empty_is_invalid` (_bool_): If **True**, an empty raster or vector is considered invalid. (Default: **True**)
    """
    if is_raster(potential_raster_or_vector, empty_is_invalid=empty_is_invalid):
        return True

    if is_vector(potential_raster_or_vector, empty_is_invalid=empty_is_invalid):
        return True

    return False


def is_raster_or_vector_list(potential_raster_or_vector_list, *, empty_is_invalid=True):
    """
    Checks if a variable is a valid list of rasters or vectors.

    ## Args:
    `potential_raster_or_vector_list` (_any_): The variable to check.

    ## Kwargs:
    `empty_is_invalid` (_bool_): If **True**, an empty raster or vector is considered invalid. (Default: **True**)
    """
    if not isinstance(potential_raster_or_vector_list, list):
        return False

    if len(potential_raster_or_vector_list) == 0:
        return False

    for element in potential_raster_or_vector_list:
        if not is_raster_or_vector(element, empty_is_invalid=empty_is_invalid):
            return False

    return True


def create_memory_path(path, *, prefix="", suffix="", add_uuid=True):
    """
    Gets a memory path from a string in the format: </br>
    `/vsimem/prefix_basename_time_uuid_suffix.ext`

    ## Args:
    `path` (_str_): The path to the original file. </br>

    ## Kwargs:
    `prefix` (_str_): The prefix to add to the memory path. (**Default**: `""`) </br>
    `suffix` (_str_): The suffix to add to the memory path. (**Default**: `""`) </br>
    `add_uuid` (_bool_): If True, add a uuid to the memory path. (**Default**: `True`) </br>

    ## Returns:
    (_str_): A string to the memory path. `/vsimem/prefix_basename_time_uuid_suffix.ext`
    """
    assert isinstance(path, str), f"path must be a string. Received: {path}"
    assert len(path) > 0, f"path must not be empty. Received: {path}"
    assert not path.startswith("/vsimem"), f"path must not already be in memory. Received: {path}"

    basename = os.path.basename(path)

    return core_utils.get_augmented_path(
        f"/vsimem/{basename}",
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        folder=None,
    )


def get_path_from_dataset(dataset, *, dataset_type=None):
    """
    Gets the path from a datasets. Can be vector or raster, string or opened.

    ## Args:
    `dataset` (_str_/_gdal.Dataset_/_ogr.DataSource_): The dataset.

    ## Kwargs:
    `dataset_type` (_str_): The type of dataset. If not specified, it is guessed. (**Default**: `None`)

    ## Returns:
    (_list_): The path to the dataset.
    """
    if (dataset_type == "raster" or dataset_type is None) and is_raster(dataset, empty_is_invalid=False):
        if isinstance(dataset, str):
            raster = gdal.Open(dataset, 0)
        elif isinstance(dataset, gdal.Dataset):
            raster = dataset
        else:
            raise Exception(f"Could not read input raster: {raster}")

        path = raster.GetDescription()
        raster = None

        return path

    if (dataset_type == "vector" or dataset_type is None) and is_vector(dataset, empty_is_invalid=False):
        if isinstance(dataset, str):
            vector = ogr.Open(dataset, 0)
        elif isinstance(dataset, ogr.DataSource):
            vector = dataset
        else:
            raise Exception(f"Could not read input vector: {vector}")

        path = vector.GetDescription()
        vector = None

        return path

    raise ValueError("The dataset is not a raster or vector.")


def get_path_from_dataset_list(datasets, *, allow_mixed=False, dataset_type=None):
    """
    Gets the paths from a list of datasets.

    ## Args:
    `datasets` (_list_): The list of datasets.

    ## Kwargs:
    `allow_mixed` (_bool_): If True, allow mixed raster/vector datasets. (**Default**: `False`) </br>
    `dataset_type` (_str_/_None_): The type of dataset. If not specified, it is guessed. (**Default**: `None`)

    ## Returns:
    (_list_): A list of paths.
    """
    assert isinstance(datasets, list), "The datasets must be a list."
    assert isinstance(dataset_type, (str, type(None))), "The dataset_type must be 'raster', 'vector', or None."

    rasters = False
    vectors = False

    outputs = []
    for dataset in datasets:
        if (dataset_type == "raster" or dataset_type is None) and is_raster(dataset, empty_is_invalid=False):
            dataset_type = "raster"
            rasters = True
        elif (dataset_type == "vector" or dataset_type is None) and is_vector(dataset, empty_is_invalid=False):
            dataset_type = "vector"
            vectors = True
        else:
            raise ValueError(f"The dataset is not a raster or vector. {dataset}")

        if rasters and vectors and not allow_mixed:
            raise ValueError("The datasets cannot be vectors and rasters mixed.")

        outputs.append(get_path_from_dataset(dataset, dataset_type=dataset_type))

    return outputs


def convert_geom_to_vector(geom):
    """
    Converts a geometry to a vector.

    ## Args:
    `geom` (_ogr.Geometry_): The geometry to convert.

    ## Kwargs:
    `layer_name` (_str_): The name of the layer. (Default: **"geom"**)
    `add_uuid` (_bool_): If **True**, a UUID will be added to the layer. (Default: **True**)

    ## Returns:
    (_ogr.DataSource_): The vector.
    """
    assert isinstance(geom, ogr.Geometry), "geom must be an ogr.Geometry."

    path = create_memory_path("converted_geom.fgb", add_uuid=True)
    driver = path_to_driver_vector(path)

    vector = driver.CreateDataSource(path)

    layer = vector.CreateLayer("converted_geom", geom.GetSpatialReference(), geom.GetGeometryType())

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geom)

    layer.CreateFeature(feature)
    feature.Destroy()

    return vector


def parse_projection(projection, *, return_wkt=False):
    """
    Parses a gdal, ogr og osr data source and extraction the projection. If
    a string or int is passed, it attempts to open it and return the projection as
    an osr.SpatialReference.

    ## Args:
    `projection` (_str_/_int_/_gdal.Dataset_/_ogr.DataSource_/_osr.SpatialReference_): The projection to parse.

    ## Kwargs:
    `return_wkt` (_bool_): If **True** the projection will be returned as a WKT string, otherwise an osr.SpatialReference is returned. (Default: **False**)
    """
    assert isinstance(projection, (str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference)), "projection must be a string, int, gdal.Dataset, ogr.DataSource, or osr.SpatialReference."

    err_msg = f"Unable to parse target projection: {projection}"
    target_proj = osr.SpatialReference()

    # Suppress gdal errors and handle them ourselves.
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    if isinstance(projection, ogr.DataSource):
        layer = projection.GetLayer()
        target_proj = layer.GetSpatialRef()

    elif isinstance(projection, gdal.Dataset):
        target_proj.ImportFromWkt(projection.GetProjection())

    elif isinstance(projection, osr.SpatialReference):
        target_proj = projection

    elif isinstance(projection, str):
        ref = gdal.Open(projection, 0)

        if ref is not None:
            target_proj.ImportFromWkt(ref.GetProjection())
        else:
            ref = ogr.Open(projection, 0)

            if ref is not None:
                layer = ref.GetLayer()
                target_proj = layer.GetSpatialRef()
            else:
                code = target_proj.ImportFromWkt(projection)
                if code != 0:
                    code = target_proj.ImportFromProj4(projection)
                    if code != 0:
                        raise ValueError(err_msg)

    elif isinstance(projection, int):
        code = target_proj.ImportFromEPSG(projection)
        if code != 0:
            raise ValueError(err_msg)

    else:
        raise ValueError(err_msg)

    gdal.PopErrorHandler()

    if isinstance(target_proj, osr.SpatialReference):
        if target_proj.GetName() is None:
            raise ValueError(err_msg)

        if return_wkt:
            return target_proj.ExportToWkt()

        return target_proj
    else:
        raise ValueError(err_msg)


def parse_raster_size(target, *, target_in_pixels=False):
    """
    Parses the raster size from either a list of numbers or a GDAL raster.

    ## Args:
    `target_size` (_any_): The target to parse raster_size from.

    ## Kwargs:
    `target_in_pixels` (_bool_): If **True**, the target size is in pixels, otherwise it is in the rasters_units. (Default: **False**)

    ## Returns:
    (_tuple_): The raster size in the form of: `(x_res, y_res, x_size, y_size)`.
    """
    assert target is not None, "target_size cannot be None."

    x_res = None
    y_res = None

    x_pixels = None
    y_pixels = None

    if isinstance(target, (gdal.Dataset, str)):
        reference = (
            target
            if isinstance(target, gdal.Dataset)
            else gdal.Open(target, 0)
        )

        transform = reference.GetGeoTransform()
        reference = None

        x_res = transform[1]
        y_res = abs(transform[5])

    elif target_in_pixels:
        if isinstance(target, tuple) or isinstance(target, list):
            if len(target) == 1:
                if core_utils.is_number(target[0]):
                    x_pixels = int(target[0])
                    y_pixels = int(target[0])
                else:
                    raise ValueError(
                        "target_size_pixels is not a number or a list/tuple of numbers."
                    )
            elif len(target) == 2:
                if core_utils.is_number(target[0]) and core_utils.is_number(target[1]):
                    x_pixels = int(target[0])
                    y_pixels = int(target[1])
            else:
                raise ValueError("target_size_pixels is either empty or larger than 2.")
        elif core_utils.is_number(target):
            x_pixels = int(target)
            y_pixels = int(target)
        else:
            raise ValueError("target_size_pixels is invalid.")

        x_res = None
        y_res = None
    else:
        if isinstance(target, tuple) or isinstance(target, list):
            if len(target) == 1:
                if core_utils.is_number(target[0]):
                    x_res = float(target[0])
                    y_res = float(target[0])
                else:
                    raise ValueError(
                        "target_size is not a number or a list/tuple of numbers."
                    )
            elif len(target) == 2:
                if core_utils.is_number(target[0]) and core_utils.is_number(target[1]):
                    x_res = float(target[0])
                    y_res = float(target[1])
            else:
                raise ValueError("target_size is either empty or larger than 2.")
        elif core_utils.is_number(target):
            x_res = float(target)
            y_res = float(target)
        else:
            raise ValueError("target_size is invalid.")

        x_pixels = None
        y_pixels = None

    return x_res, y_res, x_pixels, y_pixels


def get_gdalwarp_ram_limit(limit_in_mb):
    """
    Converts a RAM limit to a GDALWarp RAM limit.

    ## Args:
    `limit` (_str_/_int_): The RAM limit to convert. Can be auto, a percentage "80%" or a number in MB.

    ## Returns:
    (_int_): The GDALWarp RAM limit in bytes.
    """
    assert isinstance(limit_in_mb, (str, int)), "limit must be a string or integer."

    min_ram = 1000000
    limit = min_ram

    if isinstance(limit_in_mb, str):
        if limit_in_mb.lower() == "auto":
            return core_utils.get_dynamic_memory_limit_bytes()
        else:
            if "%" not in limit_in_mb:
                raise ValueError(f"Invalid limit: {limit_in_mb}")

            limit_in_percentage = limit_in_mb.replace("%", "")
            limit_in_percentage = int(limit_in_percentage)

            if limit_in_percentage <= 0 or limit_in_percentage > 100:
                raise ValueError(f"Invalid limit: {limit_in_mb}")

            limit = core_utils.get_percentage_of_total_ram_mb(limit_in_percentage) * (1024 ** 2)

            if limit > min_ram:
                return limit

    if limit > min_ram:
        return int(limit_in_mb * (1024 ** 2))

    return min_ram


def to_array_list(array_or_list_of_array):
    """
    Converts a numpy array or list of numpy arrays to a list of arrays.

    ## Args:
    `array_or_list_of_array` (_numpy.ndarray_/_list_/_str_): The numpy array or list of numpy arrays to convert to a list of arrays.

    ## Returns:
    (_list_): The list of arrays.
    """
    assert isinstance(array_or_list_of_array, (np.ndarray, list, str)), "array_or_list_of_array must be a numpy array, list of numpy arrays, or string."

    return_list = [array_or_list_of_array] if isinstance(array_or_list_of_array, np.ndarray) else array_or_list_of_array

    if len(return_list) == 0:
        raise ValueError("Empty array list.")

    for array in return_list:
        if not isinstance(array, np.ndarray):
            if isinstance(array, str) and core_utils.file_exists(array):
                try:
                    _ = np.load(array)
                except:
                    raise ValueError(f"Invalid array in list: {array}") from None
        else:
            raise ValueError(f"Invalid array in list: {array}")

    return return_list


def to_band_list(
    band_number,
    band_count,
):
    """
    Converts a band number or list of band numbers to a list of band numbers.

    ## Args:
    `band_number` (_int_/_float_/_list_): The band number or list of band numbers to convert to a list of band numbers. </br>
    `band_count` (_int_): The number of bands in the raster. </br>

    ## Returns:
    (_list_): The list of band numbers.
    """

    return_list = []
    if not isinstance(band_number, (int, float, list)):
        raise TypeError(f"Invalid type for band: {type(band_number)}")

    if isinstance(band_number, list):
        if len(band_number) == 0:
            raise ValueError("Provided list of bands is empty.")
        for val in band_number:
            try:
                band_int = int(val)
            except Exception:
                raise ValueError(
                    f"List of bands contained non-valid band number: {val}"
                ) from None

            if band_int > band_count - 1:
                raise ValueError("Requested a higher band that is available in raster.")
            else:
                return_list.append(band_int)
    elif band_number == -1:
        for val in range(band_count):
            return_list.append(val)
    else:
        if band_number > band_count + 1:
            raise ValueError("Requested a higher band that is available in raster.")
        else:
            return_list.append(int(band_number))

    return return_list


def save_dataset_to_disk(
    dataset,
    out_path,
    *,
    overwrite=True,
    creation_options=None,
    prefix="",
    suffix="",
    add_uuid=False,
):
    """
    Writes a dataset to disk. Can be a raster or a vector.

    ## Args:
    `dataset` (_str_/_gdal.Dataset_/_ogr.DataSource_/_list_): The dataset(s) to save. </br>
    `out_path` (_str_/_list_): The path(s) to save the dataset(s) to. </br>

    ## Kwargs:
    `overwrite` (_bool_): Whether to overwrite the file if it already exists. (Default: **True**) </br>
    `creation_options` (_list_/_None_): A list of creation options to pass to GDAL if saving as raster. (Default: **True**) </br>
    `prefix` (_str_): A prefix to add to the file name. (Default: **""**) </br>
    `suffix` (_str_): A suffix to add to the file name. (Default: **""**) </br>
    `add_uuid` (_bool_): Whether to add a UUID to the file name. (Default: **False**) </br>
    """
    datasets = core_utils.ensure_list(dataset)
    datasets_paths = get_path_from_dataset_list(datasets, allow_mixed=True)
    out_paths = core_utils.create_output_paths(datasets_paths, out_path, prefix=prefix, suffix=suffix, add_uuid=add_uuid)

    options = None

    for index, dataset_ in enumerate(datasets):
        opened_dataset = None
        dataset_type = None

        if is_raster(dataset_):
            options = default_creation_options(creation_options)
            dataset_type = "raster"
            if isinstance(dataset_, str):
                opened_dataset = gdal.Open(dataset_, 0)
            elif isinstance(dataset_, gdal.Dataset):
                opened_dataset = dataset_
            else:
                raise Exception(f"Could not read input raster: {dataset_}")

        elif is_vector(dataset_):
            dataset_type = "vector"
            if isinstance(dataset_, str):
                opened_dataset = ogr.Open(dataset_, 0)
            elif isinstance(dataset_, ogr.DataSource):
                opened_dataset = dataset_
            else:
                raise Exception(f"Could not read input vector: {dataset_}")

        else:
            raise Exception(f"Invalid dataset type: {dataset_}")

        driver_destination = None

        if dataset_type == "raster":
            driver_destination = gdal.GetDriverByName(path_to_driver_raster(out_paths[index]))
        else:
            driver_destination = ogr.GetDriverByName(path_to_driver_vector(out_paths[index]))

        assert driver_destination is not None, "Could not get driver for output dataset."

        core_utils.remove_if_required(out_paths[index], overwrite)

        driver_destination.CreateCopy(
            out_path[index],
            opened_dataset,
            options=options,
        )

    if isinstance(dataset, list):
        return out_paths[0]

    return out_paths


def save_dataset_to_memory(
    dataset,
    *,
    overwrite=True,
    creation_options=None,
    prefix="",
    suffix="",
    add_uuid=True,
):
    """
    Writes a dataset to memory. Can be a raster or a vector.

    ## Args:
    `dataset` (_str_/_gdal.Dataset_/_ogr.DataSource_/_list_): The dataset(s) to save. </br>

    ## Kwargs:
    `overwrite` (_bool_): Whether to overwrite the file if it already exists. (Default: **True**) </br>
    `creation_options` (_list_/_None_): A list of creation options to pass to GDAL if saving as raster. (Default: **True**) </br>
    `prefix` (_str_): A prefix to add to the file name. (Default: **""**) </br>
    `suffix` (_str_): A suffix to add to the file name. (Default: **""**) </br>
    `add_uuid` (_bool_): Whether to add a UUID to the file name. (Default: **True**) </br>
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
