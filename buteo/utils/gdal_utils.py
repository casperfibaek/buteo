"""
Utilities for working with GDAL.

TODO:
    - Documentation
"""

import os

import numpy as np
from osgeo import gdal, ogr, osr

from buteo.utils.core import (
    path_to_ext,
    is_number,
    file_exists,
    path_is_in_memory,
)


# is vector_empty
# is raster_empty


def default_options(options):
    """Takes a list of GDAL options and adds the following
        defaults to it:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW
        If any of the options are already specified, they are not
        added.

    Args:
        options (List): A list of options (str). Can be empty.

    Returns:
        A list of strings with the default options for a GDAL
        raster.
    """
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


def path_to_driver_raster(file_path, return_bool=False):
    """Takes a file path of a raster and returns the driver name."""

    raster_drivers = [
        "tif",
        "tiff",
        "img",
        "vrt",
        "jp2",
        "ecw,",
    ]

    ext = path_to_ext(file_path)

    if return_bool:
        if ext in raster_drivers:
            return True

        return False

    if ext == "tif" or ext == "tiff":
        return "GTiff"
    elif ext == "img":
        return "HFA"
    elif ext == "vrt":
        return "VRT"
    elif ext == "jp2":
        return "JP2ECW"
    elif ext == "ecw":
        return "ECW"

    raise ValueError(f"Unable to parse GDAL raster driver from path: {file_path}")


def path_to_driver_vector(file_path, return_bool=False):
    """Takes a file path of a raster and returns the driver name."""

    vector_drivers = [
        "shp",
        "gpkg",
        "fgb",
        "json",
        "geojson",
    ]

    ext = path_to_ext(file_path)

    if return_bool:
        if ext in vector_drivers:
            return True

        return False

    if ext == "shp":
        return "ESRI Shapefile"
    elif ext == "gpkg":
        return "GPKG"
    elif ext == "fgb":
        return "FlatGeobuf"
    elif ext == "json" or ext == "geojson":
        return "GeoJSON"

    raise ValueError(f"Unable to parse GDAL vector driver from path: {file_path}")


def destroy_raster(raster):
    driver = gdal.GetDriverByName(path_to_driver_raster(raster))
    driver.Delete(raster)

    return None


def is_in_memory(raster_or_vector):
    """
    Check if vector is in memory
    """
    if isinstance(raster_or_vector, str):
        if raster_or_vector.startswith("/vsimem/"):
            return True

        return False

    elif isinstance(raster_or_vector, ogr.DataSource) or isinstance(raster_or_vector, gdal.Dataset):
        if raster_or_vector.GetDriver().ShortName == "MEM":
            return True

        if raster_or_vector.GetDriver().ShortName == "Memory":
            return True

        if raster_or_vector.GetDriver().ShortName == "VirtualMem":
            return True

        if raster_or_vector.GetDriver().ShortName == "VirtualOGR":
            return True

        if raster_or_vector.GetDescription().startswith("/vsimem/"):
            return True

        return False

    else:
        raise TypeError("vector_or_raster must be a string, ogr.DataSource, or gdal.Dataset")


def delete_if_in_memory(raster_or_vector):
    """
    Delete raster or vector if it is in memory
    """
    if is_in_memory(raster_or_vector):
        if isinstance(raster_or_vector, str):
            gdal.Unlink(raster_or_vector)
        else:
            raster_or_vector.Destroy()


def is_raster(raster):
    """Checks if a raster is valid.

    Args:
        raster (str | gdal.Dataset): A path to a raster or a GDAL dataframe.

    Returns:
        A boolean.
    """
    if isinstance(raster, str):
        if not file_exists(raster) and not path_is_in_memory(raster):
            return False

        try:
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            opened = gdal.Open(raster, 0)
            gdal.PopErrorHandler()
        except Exception:
            return False

        if opened is None:
            return False

        return True

    if isinstance(raster, gdal.Dataset):
        return True

    return False


def is_vector(vector):
    """Takes a string or an ogr.DataSource and returns a boolean
    indicating if it is a valid vector.

    Args:
        file_path (path | DataSource): A path to a vector or an ogr DataSource.

    Returns:
        A boolean. True if input is a valid vector, false otherwise.
    """
    if isinstance(vector, ogr.DataSource):
        return True

    if isinstance(vector, ogr.Layer):
        return True

    if isinstance(vector, str):
        gdal.PushErrorHandler("CPLQuietErrorHandler")

        ref = ogr.Open(vector, 0)
        if ref is None:
            extension = os.path.splitext(vector)[1][1:]

            if extension == "memory" or "mem":
                driver = ogr.GetDriverByName("Memory")
                ref = driver.Open(vector)

        gdal.PopErrorHandler()

        if isinstance(ref, ogr.DataSource):
            ref = None
            return True

    return False


def clear_gdal_memory():
    """ Clears all gdal memory. """
    datasets = [ds.name for ds in gdal.listdir('/vsimem')]

    for dataset in datasets:
        gdal.Unlink(dataset)


def parse_projection(target, return_wkt=False):
    """Parses a gdal, ogr og osr data source and extraction the projection. If
        a string is passed, it attempts to open it and return the projection as
        an osr.SpatialReference.
    Args:
        target (str | gdal.datasource): A gdal data source or the path to one.

    **kwargs:
        return_wkt (bool): Indicates if the function should return a wkt string
        instead of an osr.SpatialReference.

    Returns:
        An osr.SpatialReference matching the input. If return_wkt is true, WKT
        string representing the projection is returned.
    """
    err_msg = f"Unable to parse target projection: {target}"
    target_proj = osr.SpatialReference()

    # Suppress gdal errors and handle them ourselves.
    # This ensures that the console is not flooded.
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    if isinstance(target, ogr.DataSource):
        layer = target.GetLayer()
        target_proj = layer.GetSpatialRef()
    elif isinstance(target, gdal.Dataset):
        target_proj.ImportFromWkt(target.GetProjection())
    elif isinstance(target, osr.SpatialReference):
        target_proj = target
    elif isinstance(target, str):
        ref = gdal.Open(target, 0)

        if ref is not None:
            target_proj.ImportFromWkt(ref.GetProjection())
        else:
            ref = ogr.Open(target, 0)

            if ref is not None:
                layer = ref.GetLayer()
                target_proj = layer.GetSpatialRef()
            else:
                code = target_proj.ImportFromWkt(target)
                if code != 0:
                    code = target_proj.ImportFromProj4(target)
                    if code != 0:
                        raise ValueError(err_msg)
    elif isinstance(target, int):
        code = target_proj.ImportFromEPSG(target)
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



def raster_size_from_list(target_size, target_in_pixels=False):
    x_res = None
    y_res = None

    x_pixels = None
    y_pixels = None

    if target_size is None:
        return x_res, y_res, x_pixels, y_pixels

    if isinstance(target_size, gdal.Dataset) or isinstance(target_size, str):
        reference = (
            target_size
            if isinstance(target_size, gdal.Dataset)
            else gdal.Open(target_size, 0)
        )

        transform = reference.GetGeoTransform()

        x_res = transform[1]
        y_res = abs(transform[5])

    elif target_in_pixels:
        if isinstance(target_size, tuple) or isinstance(target_size, list):
            if len(target_size) == 1:
                if is_number(target_size[0]):
                    x_pixels = int(target_size[0])
                    y_pixels = int(target_size[0])
                else:
                    raise ValueError(
                        "target_size_pixels is not a number or a list/tuple of numbers."
                    )
            elif len(target_size) == 2:
                if is_number(target_size[0]) and is_number(target_size[1]):
                    x_pixels = int(target_size[0])
                    y_pixels = int(target_size[1])
            else:
                raise ValueError("target_size_pixels is either empty or larger than 2.")
        elif is_number(target_size):
            x_pixels = int(target_size)
            y_pixels = int(target_size)
        else:
            raise ValueError("target_size_pixels is invalid.")

        x_res = None
        y_res = None
    else:
        if isinstance(target_size, tuple) or isinstance(target_size, list):
            if len(target_size) == 1:
                if is_number(target_size[0]):
                    x_res = float(target_size[0])
                    y_res = float(target_size[0])
                else:
                    raise ValueError(
                        "target_size is not a number or a list/tuple of numbers."
                    )
            elif len(target_size) == 2:
                if is_number(target_size[0]) and is_number(target_size[1]):
                    x_res = float(target_size[0])
                    y_res = float(target_size[1])
            else:
                raise ValueError("target_size is either empty or larger than 2.")
        elif is_number(target_size):
            x_res = float(target_size)
            y_res = float(target_size)
        else:
            raise ValueError("target_size is invalid.")

        x_pixels = None
        y_pixels = None

    return x_res, y_res, x_pixels, y_pixels



# TODO: Verify folder exists.
def to_path_list(variable):
    return_list = []
    if isinstance(variable, list):
        return_list = variable
    else:
        return_list.append(variable)

    if len(return_list) == 0:
        raise ValueError("Empty array list.")

    for path in return_list:
        if not isinstance(path, str):
            raise ValueError(f"Invalid string in  path list: {variable}")

    return return_list


def to_array_list(variable):
    return_list = []
    if isinstance(variable, list):
        return_list = variable
    else:
        return_list.append(variable)

    if len(return_list) == 0:
        raise ValueError("Empty array list.")

    for array in return_list:
        if not isinstance(array, np.ndarray):
            if isinstance(array, str) and os.path.exists(array):
                try:
                    _ = np.load(array)
                except:
                    raise ValueError(f"Invalid array in list: {array}")
        else:
            raise ValueError(f"Invalid array in list: {array}")

    return return_list


def to_band_list(
    variable,
    band_count,
):
    return_list = []
    if not isinstance(variable, (int, float, list)):
        raise TypeError(f"Invalid type for band: {type(variable)}")

    if isinstance(variable, list):
        if len(variable) == 0:
            raise ValueError("Provided list of bands is empty.")
        for val in variable:
            try:
                band_int = int(val)
            except Exception:
                raise ValueError(
                    f"List of bands contained non-valid band number: {val}"
                )

            if band_int > band_count - 1:
                raise ValueError("Requested a higher band that is available in raster.")
            else:
                return_list.append(band_int)
    elif variable == -1:
        for val in range(band_count):
            return_list.append(val)
    else:
        if variable > band_count + 1:
            raise ValueError("Requested a higher band that is available in raster.")
        else:
            return_list.append(int(variable))

    return return_list
