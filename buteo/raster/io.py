"""
This module does standard raster operations related to read and write.

TODO:
    - Copy, seperate, expand, create, delete
    - Remove typings
    - Documentation
"""

import sys; sys.path.append("../../") # Path: buteo/raster/io.py
import os
from math import ceil
from uuid import uuid4

import numpy as np
from osgeo import gdal, osr, ogr

from buteo.utils.core import (
    overwrite_required,
    path_to_ext,
    remove_if_overwrite,
    type_check,
    folder_exists,
    list_is_all_the_same,
    path_is_in_memory,
    file_exists,
)
from buteo.utils.gdal_utils import (
    expand_extent,
    path_to_driver_raster,
    is_raster,
    numpy_to_gdal_datatype,
    gdal_to_numpy_datatype,
    default_options,
    to_band_list,
    translate_resample_method,
    translate_datatypes,
    numpy_fill_values,
)


def _open_raster(raster, writeable):
    """INTERNAL. DO NOT USE."""
    try:
        opened = None
        if isinstance(raster, str):
            gdal.PushErrorHandler("CPLQuietErrorHandler")

            opened = gdal.Open(raster, 1) if writeable else gdal.Open(raster, 0)

            gdal.PopErrorHandler()
        elif isinstance(raster, gdal.Dataset):
            opened = raster
        else:
            raise Exception(f"Could not read input raster: {raster}")
    except:
        raise Exception(f"Could not read input raster: {raster}") from None

    if opened is None:
        raise Exception(f"Could not read input raster: {raster}")

    return opened


def open_raster(raster, *, writeable=True):
    """Opens a raster to a gdal.Dataset class.

    Args:
        raster (list | path | Dataset): A path to a raster or a GDAL dataframe.

        convert_mem_driver (bool): Converts MEM driver rasters to /vsimem/ Gtiffs.

        writable (bool): Should the opened raster be writeable.

    Returns:
        A gdal.Dataset
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(writeable, [bool], "writeable")

    return_list = []
    if isinstance(raster, list):
        input_list = raster
    else:
        input_list = [raster]

    for readied_raster in input_list:
        if isinstance(readied_raster, str):
            if path_is_in_memory(readied_raster) or file_exists(readied_raster):
                return_list.append(_open_raster(readied_raster, writeable))
            else:
                raise ValueError(f"Path does not exists: {readied_raster}")
        elif isinstance(readied_raster, gdal.Dataset):
            return_list.append(readied_raster)

    if isinstance(raster, list):
        return return_list

    return return_list[0]


def _get_raster_path(raster):
    """INTERNAL. DO NOT USE."""
    opened = open_raster(raster)

    try:
        path = opened.GetDescription()
    except:
        raise Exception(f"Could not read input raster: {raster}") from None

    return path


def get_raster_path(raster, return_list=False):
    """Takes a string or a gdal.Dataset and returns its path.

    Args:
        raster (list | path | Dataset): A path to a raster or a GDAL dataframe.

        return_list (bool): If True, returns a list of paths.

    Returns:
        A string representing the path to the raster.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")

    the_return_list = []
    if isinstance(raster, list):
        input_list = raster
    else:
        input_list = [raster]

    for readied_raster in input_list:
        the_return_list.append(_get_raster_path(readied_raster))

    if isinstance(raster, list) or return_list:
        return the_return_list

    return the_return_list[0]


def _raster_to_metadata(raster):
    """INTERNAL. DO NOT USE."""
    dataset = open_raster(raster)

    raster_driver = dataset.GetDriver()

    path = dataset.GetDescription()
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]
    ext = os.path.splitext(basename)[1]

    driver = raster_driver.ShortName

    in_memory = False
    if driver == "MEM":
        in_memory = True
    elif len(path) >= 8 and path[0:8] == "/vsimem/":
        in_memory = True

    transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    original_projection = osr.SpatialReference()
    original_projection.ImportFromWkt(projection)
    projection_osr = original_projection

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    band_count = dataset.RasterCount

    size = [dataset.RasterXSize, dataset.RasterYSize]
    shape = (width, height, band_count)

    pixel_width = abs(transform[1])
    pixel_height = abs(transform[5])

    x_min = transform[0]
    y_max = transform[3]

    x_max = x_min + width * transform[1] + height * transform[2]  # Handle skew
    y_min = y_max + width * transform[4] + height * transform[5]  # Handle skew

    band0 = dataset.GetRasterBand(1)

    datatype_gdal_raw = band0.DataType
    datatype_gdal = gdal.GetDataTypeName(datatype_gdal_raw)

    datatype = gdal_to_numpy_datatype(band0.DataType)

    nodata_value = band0.GetNoDataValue()
    has_nodata = True if nodata_value is not None else False

    extent = [x_min, y_max, x_max, y_min]
    extent_ogr = [x_min, x_max, y_min, y_max]
    extent_gdal_warp = [x_min, y_min, x_max, y_max]

    extent_dict = {
        "left": x_min,
        "top": y_max,
        "right": x_max,
        "bottom": y_min,
    }

    x_min, x_max, y_min, y_max = extent_ogr

    expanded_extents = expand_extent(extent_ogr, projection_osr)

    metadata = {
        "path": path,
        "basename": basename,
        "name": name,
        "ext": ext,
        "transform": transform,
        "in_memory": in_memory,
        "projection": projection,
        "projection_osr": projection_osr,
        "width": width,
        "height": height,
        "band_count": band_count,
        "driver": driver,
        "size": size,
        "shape": shape,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "x_min": x_min,
        "y_max": y_max,
        "x_max": x_max,
        "y_min": y_min,
        "datatype": datatype,
        "datatype_gdal": datatype_gdal,
        "datatype_gdal_raw": datatype_gdal_raw,
        "nodata_value": nodata_value,
        "has_nodata": has_nodata,
        "is_raster": True,
        "is_vector": False,
        "extent": extent,
        "extent_dict": extent_dict,
        "extent_ogr": extent_ogr,
        "extent_gdal_warp": extent_gdal_warp,
        "extent_gdal_warp_latlng": expanded_extents["extent_gdal_warp_latlng"],
        "extent_wkt": expanded_extents["extent_wkt"],
        "extent_latlng": expanded_extents["extent_latlng"],
        "extent_ogr_latlng": expanded_extents["extent_ogr_latlng"],
        "extent_dict_latlng": expanded_extents["extent_dict_latlng"],
        "extent_wkt_latlng": expanded_extents["extent_wkt_latlng"],
        "extent_geojson_dict": expanded_extents["extent_wkt_latlng"],
        "extent_geojson": expanded_extents["extent_geojson"],
        "extent_datasource": None,
        "extent_datasource_latlng": None,
        "extent_geom_latlng": None,
        "extent_geom": None,
    }

    def get_extent_datasource():
        extent_name = f"/vsimem/{name}_{uuid4().int}_extent.gpkg"

        driver = ogr.GetDriverByName("GPKG")
        extent_ds = driver.CreateDataSource(extent_name)
        layer = extent_ds.CreateLayer(
            "extent_ogr", original_projection, ogr.wkbPolygon
        )

        feature = ogr.Feature(layer.GetLayerDefn())
        extent_geom = ogr.CreateGeometryFromWkt(metadata["extent_wkt"], original_projection)
        feature.SetGeometry(extent_geom)
        layer.CreateFeature(feature)

        feature = None
        layer.SyncToDisk()

        return extent_name


    def get_extent_datasource_latlng():
        extent_latlng_name = f"/vsimem/{name}_{uuid4().int}_extent_latlng.gpkg"

        target_projection = osr.SpatialReference()
        target_projection.ImportFromEPSG(4326)

        driver = ogr.GetDriverByName("GPKG")
        extent_ds_latlng = driver.CreateDataSource(extent_latlng_name)
        layer = extent_ds_latlng.CreateLayer(
            "extent_latlng", target_projection, ogr.wkbPolygon
        )

        feature = ogr.Feature(layer.GetLayerDefn())
        extent_geom_latlng = ogr.CreateGeometryFromWkt(metadata["extent_wkt_latlng"], target_projection)
        feature.SetGeometry(extent_geom_latlng)
        layer.CreateFeature(feature)

        feature = None
        layer.SyncToDisk()

        return extent_latlng_name


    def get_extent_geom():
        return ogr.CreateGeometryFromWkt(metadata["extent_wkt"], original_projection)
    

    def get_extent_geom_latlng():
        target_projection = osr.SpatialReference()
        target_projection.ImportFromEPSG(4326)
        return ogr.CreateGeometryFromWkt(metadata["extent_wkt_latlng"], target_projection)


    metadata["extent_datasource"] = get_extent_datasource
    metadata["extent_datasource_latlng"] = get_extent_datasource_latlng
    metadata["extent_geom"] = get_extent_geom
    metadata["extent_geom_latlng"] = get_extent_geom_latlng

    return metadata


def raster_to_metadata(raster):
    """Reads a raster from a list of rasters, string or a dataset and returns metadata.

    Args:
        raster (list, path | Dataset): The raster to calculate metadata for.

    Returns:
        A dictionary containing metadata about the raster.
    """
    type_check(raster, [str, gdal.Dataset], "raster")

    input_list = get_raster_path(raster, return_list=True)
    return_list = []

    for readied_raster in input_list:
        if is_raster(readied_raster):
            return_list.append(
                _raster_to_metadata(readied_raster)
            )
        else:
            raise TypeError(f"Input: {readied_raster} is not a raster.")

    if isinstance(raster, list):
        return return_list

    return return_list[0]


def ready_io_raster(
    raster,
    out_path,
    *,
    overwrite=True,
    prefix="",
    postfix="",
    uuid=False,
):
    """
    Prepares a raster or a list of rasters for writing for writing.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")


    raster_list = get_raster_path(raster, return_list=True)

    if isinstance(out_path, list):
        if len(raster_list) != len(out_path):
            raise ValueError(
                "The length of raster_list must equal the length of the out_path"
            )

    # Check if folder exists and is required.
    test_out_path = [out_path] if not isinstance(out_path, list) else out_path
    for path in test_out_path:
        if isinstance(path, str):
            if "vsimem" in path:
                continue
            if os.path.basename(path) == "" and not os.path.isdir(
                os.path.abspath(path)
            ):
                raise ValueError(
                    f"Output folder does not exist. Please create first. {path}"
                )
            elif os.path.dirname(path) == "" or not os.path.isdir(
                os.path.dirname(path)
            ):
                raise ValueError(
                    f"Output folder does not exist. Please create first. {os.path.dirname(path)}"
                )

    # Generate output names
    path_list = []
    for index, in_raster in enumerate(raster_list):
        metadata = raster_to_metadata(in_raster)

        name = metadata["name"]

        path_id = "" if uuid is False else uuid4().int

        if out_path is None:
            path = f"/vsimem/{prefix}{name}{path_id}{postfix}.tif"
        elif isinstance(out_path, str):
            if folder_exists(out_path):
                path = os.path.join(out_path, f"{prefix}{name}{path_id}{postfix}.tif")
            else:
                path = out_path
        elif isinstance(out_path, list):
            if out_path[index] is None:
                path = f"/vsimem/{prefix}{name}{path_id}{postfix}.tif"
            elif isinstance(out_path[index], str):
                path = out_path[index]
            else:
                raise ValueError(f"Unable to parse out_path: {out_path}")
        else:
            raise ValueError(f"Unable to parse out_path: {out_path}")

        overwrite_required(path, overwrite)
        path_list.append(path)

    return (raster_list, path_list)


def rasters_are_aligned(
    rasters,
    *,
    same_extent=False,
    same_dtype=False,
    same_nodata=False,
    threshold=0.001,
):
    """Verifies if a list of rasters are aligned.

    Args:
        rasters (list): A list of raster, either in gdal.Dataset or a string
        refering to the dataset.

    **kwargs:
        same_extent (bool): Should all the rasters have the same extent?

        same_dtype (bool): Should all the rasters have the same data type?

        same_dtype (bool): Should all the rasters have the same data nodata value?
    Returns:
        True if rasters and aligned and optional parameters are True, False
        otherwise.
    """
    type_check(rasters, [list], "rasters")
    type_check(same_extent, [bool], "same_extent")
    type_check(same_dtype, [bool], "same_dtype")
    type_check(same_nodata, [bool], "same_nodata")

    if len(rasters) == 1:
        if not is_raster(rasters[0]):
            raise ValueError(f"Input raster is invalid. {rasters[0]}")

        return True

    base = {
        "projection": None,
        "pixel_width": None,
        "pixel_height": None,
        "x_min": None,
        "y_max": None,
        "transform": None,
        "width": None,
        "height": None,
        "datatype": None,
        "nodata_value": None,
    }

    for index, raster in enumerate(rasters):
        meta = _raster_to_metadata(raster)
        if index == 0:
            base["name"] = meta["name"]
            base["projection"] = meta["projection"]
            base["pixel_width"] = meta["pixel_width"]
            base["pixel_height"] = meta["pixel_height"]
            base["x_min"] = meta["x_min"]
            base["y_max"] = meta["y_max"]
            base["transform"] = meta["transform"]
            base["width"] = meta["width"]
            base["height"] = meta["height"]
            base["datatype"] = meta["datatype"]
            base["nodata_value"] = meta["nodata_value"]
        else:
            if meta["projection"] != base["projection"]:
                print(base["name"] + " did not match " + meta["name"] + " projection")
                return False
            if meta["pixel_width"] != base["pixel_width"]:
                if abs(meta["pixel_width"] - base["pixel_width"]) > threshold:
                    print(
                        base["name"] + " did not match " + meta["name"] + " pixel_width"
                    )
                    return False
            if meta["pixel_height"] != base["pixel_height"]:
                if abs(meta["pixel_height"] - base["pixel_height"]) > threshold:
                    print(
                        base["name"]
                        + " did not match "
                        + meta["name"]
                        + " pixel_height"
                    )
                    return False
            if meta["x_min"] != base["x_min"]:
                if abs(meta["x_min"] - base["x_min"]) > threshold:
                    print(base["name"] + " did not match " + meta["name"] + " x_min")
                    return False
            if meta["y_max"] != base["y_max"]:
                if abs(meta["y_max"] - base["y_max"]) > threshold:
                    print(base["name"] + " did not match " + meta["name"] + " y_max")
                    return False
            if same_extent:
                if meta["transform"] != base["transform"]:
                    return False
                if meta["height"] != base["height"]:
                    return False
                if meta["width"] != base["width"]:
                    return False

            if same_dtype:
                if meta["datatype"] != base["datatype"]:
                    return False

            if same_nodata:
                if meta["nodata_value"] != base["nodata_value"]:
                    return False

    return True


def _raster_to_memory(
    raster,
    *,
    memory_path=None,
    copy_if_already_in_memory=False,
):
    """INTERNAL. DO NOT USE."""

    ref = open_raster(raster)
    path = get_raster_path(ref)
    raster_driver = ref.GetDriver()
    driver_name = raster_driver.ShortName

    if not copy_if_already_in_memory and (
        path[0:8] == "/vsimem/" or driver_name == "MEM"
    ):
        return path

    options = []
    if memory_path is not None:

        if memory_path[0:8] == "/vsimem/":
            raster_name = memory_path
        else:
            raster_name = f"/vsimem/{memory_path}"

        driver_name = path_to_driver_raster(memory_path)
        if driver_name is None:
            driver_name = "GTiff"

        if driver_name == "GTiff":
            options.append("BIGTIFF=YES")

        driver = gdal.GetDriverByName(driver_name)
    else:
        metadata = raster_to_metadata(ref)
        name = metadata["name"]
        raster_name = f"/vsimem/{name}_{uuid4().int}.tif"
        driver = gdal.GetDriverByName("GTiff")

        options.append("BIGTIFF=YES")

    driver.CreateCopy(raster_name, ref, options=options)

    return raster_name


def raster_to_memory(raster, memory_path=None):
    """Takes a file path or a gdal raster dataset and copies
    it to memory.

    Args:
        file_path (path | Dataset): A path to a raster or a GDAL dataframe.

    **kwargs:
        memory_path (str | None): If a path is provided, uses the
        appropriate driver and uses the VSIMEM gdal system.
        Example: raster_to_memory(clip_ref, "clip_geom.gpkg")
        /vsimem/ is autumatically added.

    Returns:
        A gdal.Dataset copied into memory.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(memory_path, [list, str], "memory_path", allow_none=True)

    raster_list, out_paths = ready_io_raster(
        raster, out_path=memory_path, overwrite=True
    )

    results = []
    for index, in_raster in enumerate(raster_list):
        result = _raster_to_memory(in_raster, memory_path=out_paths[index])

        if not isinstance(result, dict):
            raise Exception(f"Error while parsing metadata for: {in_raster}")

        results.append(result)

    if not isinstance(raster, list):
        return results[0]

    return results


def raster_to_array(
    raster,
    *,
    bands=-1,
    filled=False,
    output_2d=False,
    extent=None,
    extent_pixels=None,
):
    """Turns a path to a raster(s) or a GDAL.Dataset(s) into a numpy
        array(s).

    Args:
        raster (list | path | Dataset): The raster(s) to convert.

    **kwargs:
        bands (list | str | int): The bands from the raster to turn
        into a numpy array. Can be "all", "ALL", a list of ints or a
        single int.

        filled (bool): If the array contains nodata values. Should the
        resulting array be a filled numpy array or a masked array?

        output_2d (bool): If True, returns only the first band in a 2D
        fashion eg. (1920x1080) instead of the default channel-last format
        (1920x1080x1)

        extent (list): A list of [xmin, ymin, xmax, ymax] to use as the
        extent of the raster. Uses coordinates

        extent_pixels (list): A list of [xmin, ymin, xmax, ymax] to use as
        the extent of the raster. Uses pixel offsets.

    Returns:
        A numpy array in the 3D channel-last format unless output_2D is
        specified.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(bands, [int, list], "bands")
    type_check(filled, [bool], "filled")
    type_check(output_2d, [bool], "output_2d")

    internal_rasters = get_raster_path(raster, return_list=True)

    if not rasters_are_aligned(internal_rasters, same_extent=True, same_dtype=False):
        raise ValueError(
            "Cannot merge rasters that are not aligned, have dissimilar extent or dtype."
        )

    layers = []
    nodata_values = []
    for in_raster in internal_rasters:
        ref = open_raster(in_raster)

        metadata = raster_to_metadata(ref)

        band_count = metadata["band_count"]
        raster_extent = metadata["extent_ogr"]
        width = metadata["width"]
        height = metadata["height"]
        pixel_width = metadata["pixel_width"]
        pixel_height = metadata["pixel_height"]

        if band_count == 0:
            raise ValueError("The input raster does not have any valid bands.")

        internal_bands = to_band_list(bands, metadata["band_count"])

        for band in internal_bands:
            band_ref = ref.GetRasterBand(band + 1)
            band_nodata_value = band_ref.GetNoDataValue()

            nodata_values.append(band_nodata_value)

            if extent_pixels is not None:
                arr = band_ref.ReadAsArray(
                    extent_pixels[0],
                    extent_pixels[1],
                    extent_pixels[2],
                    extent_pixels[3],
                )
            elif extent is not None:
                ex_min, ex_max, ey_min, ey_max = extent
                x_min, x_max, y_min, y_max = raster_extent

                if ex_min > x_max or ex_max < x_min or ey_min > y_max or ey_max < y_min:
                    raise ValueError("Extent is outside of raster.")

                if ex_min < x_min:
                    x_offset = int(0)
                else:
                    x_offset = int((ex_min - x_min) // pixel_width)

                if ex_max > x_max:
                    x_pixels = ceil(width - x_offset)
                else:
                    x_pixels = ceil(
                        int(width) - int(((x_max - ex_max) // pixel_width)) - x_offset
                    )

                if ey_min < y_min:
                    y_offset = int(0)
                else:
                    y_offset = int((ey_min - y_min) // pixel_height)

                if ey_max > y_max:
                    y_pixels = ceil(height - y_offset)
                else:
                    y_pixels = ceil(
                        int(height) - int((y_max - ey_max) // pixel_height) - y_offset
                    )

                arr = band_ref.ReadAsArray(x_offset, y_offset, x_pixels, y_pixels)
            else:
                arr = band_ref.ReadAsArray()

            if band_nodata_value is not None:
                arr = np.ma.array(arr, mask=arr == band_nodata_value)

                if filled:
                    arr = arr.filled(band_nodata_value)

            layers.append(arr)

            if output_2d:
                break

    if output_2d:
        if band_nodata_value is not None and filled is False:
            stacked = np.ma.dstack(layers)[:, :, 0]

            if list_is_all_the_same(nodata_values):
                stacked.fill_value = nodata_values[0]
            else:
                stacked.fill_value = numpy_fill_values(stacked.dtype)

            return stacked

        return np.dstack(layers)[:, :, 0]

    if band_nodata_value is not None and filled is False:
        stacked = np.ma.dstack(layers)

        if list_is_all_the_same(nodata_values):
            stacked.fill_value = nodata_values[0]
        else:
            stacked.fill_value = numpy_fill_values(stacked.dtype)

        return stacked

    return np.dstack(layers)


def _raster_to_disk(
    raster,
    out_path,
    *,
    overwrite=True,
    creation_options=None,
):
    """WARNING: INTERNAL. DO NOT USE."""
    ref = open_raster(raster)

    driver = gdal.GetDriverByName(path_to_driver_raster(out_path))

    if driver is None:
        raise Exception(f"Error while parsing driver from extension: {out_path}")

    remove_if_overwrite(out_path, overwrite)
    driver.CreateCopy(out_path, ref, options=creation_options)

    return out_path


def raster_to_disk(
    raster,
    out_path,
    *,
    overwrite=True,
    creation_options=None,
):
    """Saves or copies a raster to disk. Can be used to change datatype.
    Input is either a filepath to a raster or a GDAL.Dataset.
    The driver is infered from the file extension.

    Args:
        raster (path | Dataset): The raster to save to disk.
        out_path (path): The destination to save to.

    **kwargs:
        overwrite (bool): If the file exists, should it be overwritten?

        creation_options (list): GDAL creation options. Defaults are:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW"

        opened (bool): Should the resulting raster be opened
        or a path.

    Returns:
        The filepath for the newly created raster.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(out_path, [list, str], "out_path")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options", allow_none=True)

    if creation_options is None:
        creation_options = []

    if not os.path.dirname(os.path.abspath(out_path)):
        raise ValueError(
            f"Output folder does not exist. Please create first. {out_path}"
        )

    raster_list, path_list = ready_io_raster(raster, out_path, overwrite=overwrite)

    output = []
    for index, in_raster in enumerate(raster_list):
        path = _raster_to_disk(
            in_raster,
            path_list[index],
            overwrite=overwrite,
            creation_options=default_options(creation_options),
        )

        output.append(path)

    if isinstance(raster, list):
        return output

    return output[0]


def _raster_set_datatype(
    raster,
    dtype,
    out_path=None,
    *,
    overwrite=True,
    creation_options=None,
):
    """OBS: INTERNAL: Single output.

    Changes the datatype of a raster.
    """
    type_check(raster, [str, gdal.Dataset], "raster")
    type_check(dtype, [str], "dtype")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(creation_options, [list], "creation_options", allow_none=True)

    ref = open_raster(raster)
    metadata = raster_to_metadata(ref)

    path = out_path
    if path is None:
        name = metadata["name"]
        path = f"/vsimem/{name}_{uuid4().int}.tif"

    driver = gdal.GetDriverByName(path_to_driver_raster(path))

    remove_if_overwrite(path, overwrite)

    copy = driver.Create(
        path,
        metadata["height"],
        metadata["width"],
        metadata["band_count"],
        translate_datatypes(dtype),
        default_options(creation_options),
    )

    copy.SetProjection(metadata["projection"])
    copy.SetGeoTransform(metadata["transform"])

    array = raster_to_array(ref)

    for band_idx in range(metadata["band_count"]):
        band = copy.GetRasterBand(band_idx + 1)
        band.WriteArray(array[:, :, band_idx])
        band.SetNoDataValue(metadata["nodata_value"])

    return path


def raster_set_datatype(
    raster,
    dtype,
    out_path=None,
    *,
    overwrite=True,
    creation_options=None,
):
    """Changes the datatype of a raster.

    Args:
        raster (path | Dataset): The raster(s) to convert.

        dtype (str): The destination datatype: Can be float32, uint8 etc..

    **kwargs:
        out_path (str | None): The destination of the output. If none,
        a memory raster with a random name is generated.

        creation_options (list): A list of options for the GDAL creation. Only
        used if an outpath is specified. Defaults are:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW"

    Returns:
        A path to the newly created raster.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(dtype, [str], "dtype")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(creation_options, [list], "creation_options", allow_none=True)

    raster_list, path_list = ready_io_raster(raster, out_path, overwrite=overwrite)

    output = []
    for index, in_raster in enumerate(raster_list):
        path = _raster_set_datatype(
            in_raster,
            dtype,
            out_path=path_list[index],
            overwrite=overwrite,
            creation_options=default_options(creation_options),
        )

        output.append(path)

    if isinstance(raster, list):
        return output

    return output[0]


def array_to_raster(
    array,
    reference,
    out_path=None,
    *,
    overwrite=True,
    set_nodata="pass",
    creation_options=None,
):
    """Turns a numpy array into a GDAL dataset or exported
        as a raster using a reference raster.

    Args:
        array (np.ndarray): The numpy array to convert

        reference (path or Dataset): A reference on which to base
        the geographical extent and projection of the raster.

    **kwargs:
        out_path (path): The location to save the raster. If None is
        supplied an in memory raster is returned. filetype is infered
        from the extension.

        overwrite (bool): Specifies if the file already exists, should
        it be overwritten?

        creation_options (list): A list of options for the GDAL creation. Only
        used if an outpath is specified. Defaults are:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW"

    Returns:
        If an out_path has been specified, it returns the path to the
        newly created raster file.
    """
    type_check(array, [np.ndarray, np.ma.MaskedArray], "array")
    type_check(reference, [str, gdal.Dataset], "reference")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(set_nodata, [int, float, str], "set_nodata", allow_none=True)
    type_check(creation_options, [list], "creation_options", allow_none=True)

    # Verify the numpy array
    if (
        not isinstance(array, (np.ndarray, np.ma.MaskedArray))
        or array.size == 0
        or array.ndim < 2
        or array.ndim > 3
    ):
        raise ValueError(f"Input array is invalid {array}")

    # Parse the driver
    driver_name = "GTiff" if out_path is None else path_to_driver_raster(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    # How many bands?
    bands = 1
    if array.ndim == 3:
        bands = array.shape[2]

    overwrite_required(out_path, overwrite)

    metadata = raster_to_metadata(reference)
    reference_nodata = metadata["nodata_value"]

    # handle nodata. GDAL python throws error if conversion in not explicit.
    if reference_nodata is not None:
        reference_nodata = float(reference_nodata)
        if (reference_nodata).is_integer() is True:
            reference_nodata = int(reference_nodata)

    # Handle nodata
    input_nodata = None
    if np.ma.is_masked(array) is True:
        input_nodata = array.get_fill_value()  # type: ignore (because it's a masked array.)

    destination_dtype = numpy_to_gdal_datatype(array.dtype)

    # Weird double issue with GDAL and numpy. Cast to float or int
    if input_nodata is not None:
        input_nodata = float(input_nodata)
        if (input_nodata).is_integer() is True:
            input_nodata = int(input_nodata)

    output_name = None
    if out_path is None:
        output_name = f"/vsimem/array_to_raster_{uuid4().int}.tif"
    else:
        output_name = out_path

    if metadata["width"] != array.shape[1] or metadata["height"] != array.shape[0]:
        print("WARNING: Input array and raster are not of equal size.")

    remove_if_overwrite(out_path, overwrite)

    destination = driver.Create(
        output_name,
        array.shape[1],
        array.shape[0],
        bands,
        destination_dtype,
        default_options(creation_options),
    )

    destination.SetProjection(metadata["projection"])
    destination.SetGeoTransform(metadata["transform"])

    for band_idx in range(bands):
        band = destination.GetRasterBand(band_idx + 1)
        if bands > 1 or array.ndim == 3:
            band.WriteArray(array[:, :, band_idx])
        else:
            band.WriteArray(array)

        if set_nodata != "pass":
            if set_nodata is not None:
                band.SetNoDataValue(set_nodata)
        elif input_nodata is not None:
            band.SetNoDataValue(input_nodata)
        elif reference_nodata is not None:
            band.SetNoDataValue(reference_nodata)

    return output_name


def stack_rasters(
    rasters,
    out_path=None,
    *,
    overwrite=True,
    dtype=None,
    creation_options=None,
):
    """Stacks a list of rasters. Must be aligned."""
    type_check(rasters, [list], "rasters")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(dtype, [str], "dtype", allow_none=True)
    type_check(creation_options, [list], "creation_options", allow_none=True)

    if not rasters_are_aligned(rasters, same_extent=True):
        raise ValueError("Rasters are not aligned. Try running align_rasters.")

    overwrite_required(out_path, overwrite)

    # Ensures that all the input rasters are valid.
    raster_list = get_raster_path(rasters, return_list=True)

    if out_path is not None and path_to_ext(out_path) == ".vrt":
        raise ValueError("Please use stack_rasters_vrt to create vrt files.")

    # Parse the driver
    driver_name = "GTiff" if out_path is None else path_to_driver_raster(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = f"/vsimem/stacked_rasters_{uuid4().int}.tif"
    else:
        output_name = out_path

    raster_dtype = raster_to_metadata(raster_list[0])["datatype_gdal_raw"]

    datatype = translate_datatypes(dtype) if dtype is not None else raster_dtype

    nodata_values = []
    nodata_missmatch = False
    nodata_value = None
    total_bands = 0
    metadatas = []
    for raster in raster_list:
        metadata = raster_to_metadata(raster)
        metadatas.append(metadata)

        nodata_value = metadata["nodata_value"]
        total_bands += metadata["band_count"]

        if nodata_missmatch is False:
            for ndv in nodata_values:
                if nodata_missmatch:
                    continue

                if metadata["nodata_value"] != ndv:
                    print(
                        "WARNING: NoDataValues of input rasters do not match. Removing nodata."
                    )
                    nodata_missmatch = True

        nodata_values.append(metadata["nodata_value"])

    if nodata_missmatch:
        nodata_value = None

    remove_if_overwrite(out_path, overwrite)

    destination = driver.Create(
        output_name,
        metadatas[0]["width"],
        metadatas[0]["height"],
        total_bands,
        datatype,
        default_options(creation_options),
    )

    destination.SetProjection(metadatas[0]["projection"])
    destination.SetGeoTransform(metadatas[0]["transform"])

    bands_added = 0
    for index, raster in enumerate(raster_list):
        metadata = metadatas[index]
        band_count = metadata["band_count"]

        array = raster_to_array(raster)

        for band_idx in range(band_count):
            dst_band = destination.GetRasterBand(bands_added + 1)
            dst_band.WriteArray(array[:, :, band_idx])

            if nodata_value is not None:
                dst_band.SetNoDataValue(nodata_value)

            bands_added += 1

    return output_name


def stack_rasters_vrt(
    rasters,
    out_path,
    seperate=True,
    *,
    resample_alg="nearest",
    options=(),
    overwrite=True,
    reference=None,
    creation_options=None,
):
    """Stacks a list of rasters into a virtual rasters (.vrt)."""
    type_check(rasters, [list], "rasters")
    type_check(out_path, [str], "out_path")
    type_check(seperate, [bool], "seperate")
    type_check(resample_alg, [str], "resample_alg")
    type_check(options, [tuple], "options")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options", allow_none=True)

    resample_algorithm = translate_resample_method(resample_alg)

    if reference is not None:
        meta = raster_to_metadata(reference)
        options = gdal.BuildVRTOptions(
            resampleAlg=resample_algorithm,
            separate=seperate,
            outputBounds=[meta["x_min"], meta["y_min"], meta["x_max"], meta["y_max"]],
            xRes=meta["pixel_width"],
            yRes=meta["pixel_height"],
            targetAlignedPixels=True,
        )
    else:
        options = gdal.BuildVRTOptions(resampleAlg=resample_algorithm, separate=seperate)

    gdal.BuildVRT(out_path, rasters, options=options)

    return out_path


def rasters_intersect(raster1, raster2):
    """Checks if two rasters intersect."""
    type_check(raster1, [list, str, gdal.Dataset], "raster1")
    type_check(raster2, [list, str, gdal.Dataset], "raster2")

    meta1 = raster_to_metadata(raster1)
    meta2 = raster_to_metadata(raster2)

    return meta1["extent_geom_latlng"].Intersects(meta2["extent_geom_latlng"])


def rasters_intersection(raster1, raster2):
    """Checks if two rasters intersect."""
    type_check(raster1, [list, str, gdal.Dataset], "raster1")
    type_check(raster2, [list, str, gdal.Dataset], "raster2")

    meta1 = raster_to_metadata(raster1)
    meta2 = raster_to_metadata(raster2)

    return meta1["extent_geom_latlng"].Intersection(meta2["extent_geom_latlng"])


def copy_raster(
    raster,
    out_path,
    *,
    overwrite=True,
    dtype=None,
    creation_options=None,
    opened=False,
):
    """Copies a raster to a path. Can be both memory and disk."""
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(out_path, [list, str], "out_path")
    type_check(overwrite, [bool], "overwrite")
    type_check(dtype, [str], "dtype", allow_none=True)
    type_check(creation_options, [list], "creation_options", allow_none=True)
    type_check(opened, [bool], "opened")

    raise ValueError("Not yet implemented. Sorry")


def seperate_bands(
    raster,
    out_names=None,
    out_dir=None,
    *,
    overwrite=True,
    opened=False,
    prefix="",
    postfix="_bandID",  # add automatically
    creation_options=None,
):
    """Seperates the bands of a multiband raster."""
    type_check(raster, [str, gdal.Dataset], "raster")
    type_check(out_names, [list], "out_names", allow_none=True)
    type_check(out_dir, [str], "out_dir", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(opened, [bool], "opened")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(creation_options, [list], "creation_options", allow_none=True)

    raise ValueError("Not yet implemented. Sorry")


def create_raster(
    out_path=None,
    reference=None,
    top_left=None,
    height=None,
    width=None,
    pixel_height=None,
    pixel_width=None,
    projection=None,
    extent=None,
    dtype=None,
    filled_with=0,
    bands=1,
    nodata_value=None,
    overwrite=True,
    opened=False,
    creation_options=None,
):
    """Returns a new created raster initialised with some value."""
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(reference, [str, gdal.Dataset], "reference", allow_none=True)
    type_check(top_left, [list, tuple], "top_left", allow_none=True)
    type_check(height, [int], "height", allow_none=True)
    type_check(width, [int], "width", allow_none=True)
    type_check(pixel_height, [int, float], "pixel_height", allow_none=True)
    type_check(pixel_width, [int, float], "pixel_width", allow_none=True)
    type_check(
        projection,
        [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
        "projection",
        allow_none=True,
    )
    type_check(extent, [list], "extent", allow_none=True)
    type_check(dtype, [str], "dtype", allow_none=True)
    type_check(filled_with, [int, float], "filled_with", allow_none=True)
    type_check(bands, [int], "bands")
    type_check(nodata_value, [float, int], "nodata_value", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(opened, [bool], "opened")
    type_check(creation_options, [list], "creation_options", allow_none=True)

    raise ValueError("Not yet implemented. Sorry")


def delete_raster(raster):
    """Deletes a raster."""
    type_check(raster, [str, gdal.Dataset], "raster")

    raise ValueError("Not yet implemented. Sorry")
