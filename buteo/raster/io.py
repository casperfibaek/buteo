import sys

sys.path.append("../../")
from uuid import uuid4
from osgeo import gdal, osr, ogr
from typing import Dict, Tuple, Union, List, Any, Optional
from buteo.project_types import Metadata_raster, Metadata_raster_comp, Number
import numpy as np
import os

from buteo.utils import (
    overwrite_required,
    path_to_ext,
    remove_if_overwrite,
    type_check,
)
from buteo.gdal_utils import (
    advanced_extents,
    raster_to_reference,
    path_to_driver,
    numpy_to_gdal_datatype,
    gdal_to_numpy_datatype,
    default_options,
    ready_io_raster,
    to_band_list,
    to_path_list,
    to_raster_list,
    translate_resample_method,
    is_raster,
    translate_datatypes,
)


# TODO: copy_raster
# TODO: seperate_bands
# TODO: stack_rasters_vrt
# TODO: create_raster
# TODO: assign_projection + link in reproject.py
# TODO: delete_raster // clear memory


def raster_to_metadata(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    simple: bool = True,
) -> Union[Metadata_raster, List[Metadata_raster]]:
    """ Reads a raster from a string or a dataset and returns metadata.

    Args:
        raster (path | Dataset): The raster to save to disc.

    **kwargs:
        simple (bool): If False footprints of the raster is calculated, including
        in latlng (wgs84). Requires a reprojection check. Do not use if not required
        and performance is essential. Produces geojsons as well.

    Returns:
        A dictionary containing metadata about the raster.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(simple, [bool], "simple")

    rasters = to_raster_list(raster)

    metadatas: List[Metadata_raster] = []

    for in_raster in rasters:
        dataset = raster_to_reference(in_raster)

        raster_driver = dataset.GetDriver()

        path: str = dataset.GetDescription()
        basename: str = os.path.basename(path)
        name: str = os.path.splitext(basename)[0]
        ext: str = os.path.splitext(basename)[1]

        transform: List[Number] = dataset.GetGeoTransform()
        projection: str = dataset.GetProjection()

        original_projection: osr.SpatialReference = osr.SpatialReference()
        original_projection.ImportFromWkt(projection)
        projection_osr: osr.SpatialReference = original_projection

        width: int = dataset.RasterXSize
        height: int = dataset.RasterYSize
        band_count: int = dataset.RasterCount

        driver: str = raster_driver.ShortName

        size: List[int] = [dataset.RasterXSize, dataset.RasterYSize]
        shape: Tuple[int, int, int] = (width, height, band_count)

        pixel_width: Number = abs(transform[1])
        pixel_height: Number = abs(transform[5])

        x_min: Number = transform[0]
        y_max: Number = transform[3]
        x_max: Number = x_min + width * pixel_width + height * abs(transform[2])
        y_min: Number = y_max + width * abs(transform[4]) + height * pixel_height

        band0 = dataset.GetRasterBand(1)

        datatype_gdal_raw: int = band0.DataType
        datatype_gdal: str = gdal.GetDataTypeName(datatype_gdal_raw)

        datatype: str = gdal_to_numpy_datatype(band0.DataType)

        nodata_value: Optional[Number] = band0.GetNoDataValue()
        has_nodata = True if nodata_value is not None else False

        extent: List[Number] = [x_min, y_max, x_max, y_min]
        extent_ogr: List[Number] = [x_min, x_max, y_min, y_max]
        extent_gdal_warp: List[Number] = [x_min, y_min, x_max, y_max]

        extent_dict: Dict[Any, Number] = {
            "left": x_min,
            "top": y_max,
            "right": x_max,
            "bottom": y_min,
        }

        metadata: Metadata_raster = {
            "path": path,
            "basename": basename,
            "name": name,
            "ext": ext,
            "transform": transform,
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
            "extent_ogr": extent_ogr,
            "extent_gdal_warp": extent_gdal_warp,
            "extent_dict": extent_dict,
            "extent_wkt": None,
            "extent_datasource": None,
            "extent_geom": None,
            "extent_latlng": None,
            "extent_gdal_warp_latlng": None,
            "extent_ogr_latlng": None,
            "extent_dict_latlng": None,
            "extent_wkt_latlng": None,
            "extent_datasource_latlng": None,
            "extent_geom_latlng": None,
            "extent_geojson": None,
            "extent_geojson_dict": None,
        }

        if not simple:
            extended_extents = advanced_extents(extent_ogr, projection_osr)

            for key, value in extended_extents.items():
                metadata[key] = value # type: ignore

        metadatas.append(metadata)

    if isinstance(raster, list):
        return metadatas

    return metadatas[0]


def rasters_are_aligned(
    rasters: List[Union[str, gdal.Dataset]],
    same_extent: bool = False,
    same_dtype: bool = False,
    same_nodata: bool = False,
) -> bool:
    """ Verifies if a list of rasters are aligned.

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

    if len(rasters) == 1:
        if not is_raster(rasters[0]):
            raise ValueError(f"Input raster is invalid. {rasters[0]}")

        return True

    metas = raster_to_metadata(rasters)

    if isinstance(metas, dict):
        raise Exception("Metadata returned list.")

    base: Metadata_raster_comp = {
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

    for index, meta in enumerate(metas):
        if index == 0:
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
                return False
            if meta["pixel_width"] != base["pixel_width"]:
                return False
            if meta["pixel_height"] != base["pixel_height"]:
                return False
            if meta["x_min"] != base["x_min"]:
                return False
            if meta["y_max"] != base["y_max"]:
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


def raster_in_memory(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
) -> Union[bool, List[bool]]:
    """ Returns True if the raster(s) is hosted in memory.
    Args:
        raster (list | path | Dataset): The rasters to test.

    Returns:
        True if raster is in Memory, False otherwise.
    """
    type_check(raster, [str, gdal.Dataset], "raster")

    metadata = raster_to_metadata(raster)

    return_list: List[bool] = []
    if isinstance(metadata, dict):

        if metadata["driver"] == "MEM":
            return_list.append(True)
        elif "/vsimem/" in metadata["path"]:
            return_list.append(True)
        else:
            return_list.append(False)

    else:
        for meta in metadata:
            if meta["driver"] == "MEM":
                return_list.append(True)
            elif "/vsimem/" in meta["path"]:
                return_list.append(True)
            else:
                return_list.append(False)

    if isinstance(metadata, dict):
        return return_list[0]

    return return_list


def raster_to_memory(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    memory_path: Union[str, None] = None,
    opened: bool = False,
) -> Union[list, str, gdal.Dataset]:
    """ Takes a file path or a gdal raster dataset and copies
    it to memory. 

    Args:
        file_path (path | Dataset): A path to a raster or a GDAL dataframe.

    **kwargs:
        memory_path (str | None): If a path is provided, uses the
        appropriate driver and uses the VSIMEM gdal system.
        Example: vector_to_memory(clip_ref, "clip_geom.gpkg")
        /vsimem/ is autumatically added.
    
        opened (bool): If a memory path is specified, the default is 
        to return a path. If open is supplied. The raster is opened
        before returning.

    Returns:
        A gdal.Dataset copied into memory.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(memory_path, [str], "memory_path", allow_none=True)
    type_check(opened, [bool], "opened")

    raster_list = to_raster_list(raster)
    results = []

    for in_raster in raster_list:
        ref = raster_to_reference(in_raster)

        options = []
        driver = None
        raster_name = None
        if memory_path is not None:

            if memory_path[0:8] == "/vsimem/":
                raster_name = memory_path
            else:
                raster_name = f"/vsimem/{memory_path}"

            driver_name = path_to_driver(memory_path)
            if driver_name is None:
                driver_name = "GTiff"
            driver = gdal.GetDriverByName(driver_name)
        else:
            raster_name = f"/vsimem/mem_rast_{uuid4().int}.tif"
            driver = gdal.GetDriverByName("GTiff")
            options.append("BIGTIFF=YES")

        copy = driver.CreateCopy(raster_name, ref, options=options)

        if opened:
            results.append(copy)
        else:
            results.append(raster_name)

    if not isinstance(raster, list):
        return results[0]

    return results


def raster_to_array(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    bands: Union[int, list] = -1,
    filled: bool = False,
    merge: bool = True,
    output_2d: bool = False,
) -> np.ndarray:
    """ Turns a path to a raster(s) or a GDAL.Dataset(s) into a numpy
        array(s).

    Args:
        raster (path | Dataset): The raster(s) to convert.

    **kwargs:
        bands (str | int | list): The bands from the raster to turn
        into a numpy array. Can be "all", "ALL", a list of ints or a
        single int.

        filled (bool): If the array contains nodata values. Should the
        resulting array be a filled numpy array or a masked array?

        output_2d (bool): If True, returns only the first band in a 2D
        fashion eg. (1920x1080) instead of the default channel-last format
        (1920x1080x1)

    Returns:
        A numpy array in the 3D channel-last format unless output_2D is
        specified.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(bands, [int, list], "bands")
    type_check(filled, [bool], "filled")
    type_check(merge, [bool], "merge")
    type_check(output_2d, [bool], "output_2d")

    internal_rasters = to_raster_list(raster)

    if not rasters_are_aligned(internal_rasters, same_extent=True, same_dtype=False,):
        raise ValueError(
            "Cannot merge rasters that are not aligned, have dissimilar extent or dtype."
        )

    layers = []
    for in_raster in internal_rasters:
        ref = raster_to_reference(in_raster)

        metadata = raster_to_metadata(ref)

        if not isinstance(metadata, dict):
            raise Exception("Metadata is not a dict.")

        band_count = metadata["band_count"]

        if band_count == 0:
            raise ValueError("The input raster does not have any valid bands.")

        internal_bands = to_band_list(bands, metadata["band_count"])

        band_stack = []
        for band in internal_bands:
            band_ref = ref.GetRasterBand(band + 1)
            band_nodata_value = band_ref.GetNoDataValue()

            arr = band_ref.ReadAsArray()

            if band_nodata_value is None and filled is False:
                arr = np.ma.masked_invalid(arr)
            elif filled is False:
                arr = np.ma.masked_equal(arr, band_nodata_value)

            if merge:
                layers.append(arr)
            else:
                band_stack.append(arr)

            if output_2d:
                break

        if not merge:
            layers.append(band_stack)

    if merge:
        if output_2d:
            return np.dstack(layers)[:, :, :, 0]

        return np.dstack(layers)

    else:
        return_layers = []
        for layer in layers:
            if output_2d:
                return_layers.append(np.dstack(layer)[:, :, :, 0])
            else:
                return_layers.append(np.dstack(layer)[:, :, :, :])

        return return_layers


def raster_to_disk(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    out_path: Union[List[str], str],
    overwrite: bool = True,
    dtype: Union[str, None] = None,
    creation_options: list = [],
    opened: bool = False,
) -> Union[list, str]:
    """ Saves or copies a raster to disk. Can be used to change datatype.
    Input is either a filepath to a raster or a GDAL.Dataset.
    The driver is infered from the file extension.

    Args:
        raster (path | Dataset): The raster to save to disk.
        out_path (path): The destination to save to.

    **kwargs:
        overwrite (bool): If the file exists, should it be overwritten?

        dtype (str): The datatype of the saved raster ('float32', ...)

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
    type_check(dtype, [str], "dtype", allow_none=True)
    type_check(creation_options, [list], "creation_options")
    type_check(opened, [bool], "opened")

    if isinstance(raster, list) and not isinstance(out_path, list):
        raise ValueError("If input raster is a list, out_path must also be a list.")

    if isinstance(raster, list) and isinstance(out_path, list):
        if len(raster) != len(out_path):
            raise ValueError("raster list must have same length as out_path list.")

    raster_list, path_list = ready_io_raster(raster, out_path, overwrite)

    for index, in_raster in enumerate(raster_list):
        ref = raster_to_reference(in_raster)
        path = path_list[index]

        metadata = raster_to_metadata(ref)

        if not isinstance(metadata, dict):
            raise Exception("Metadata is not a dict.")

        remove_if_overwrite(path, overwrite)

        driver = gdal.GetDriverByName(path_to_driver(path))

        copy = None
        if dtype is not None:
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
                band.SetNodataValue(metadata["nodata_value"])
        else:
            copy = driver.CreateCopy(path, ref, options=creation_options)

        if copy is None:
            raise Exception("Error while creating copy.")

    if not isinstance(raster, list):
        if opened:
            return raster_to_reference(path_list[0])

        return path_list[0]

    if opened:
        for index, _ in enumerate(path_list):
            path_list[index] = raster_to_reference(path_list[index])

    return path_list


def raster_set_dtype(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    dtype: str,
    out_path: Union[List[str], str, None],
    overwrite: bool = True,
    opened: bool = False,
    creation_options: list = [],
) -> Union[str, gdal.Dataset]:
    """ Changes the datatype of a raster.

    Args:
        raster (path | Dataset): The raster(s) to convert.

        dtype (str): The destination datatype: Can be float32, uint8 etc..

    **kwargs:
        out_path (str | None): The destination of the output. If none,
        a memory raster with a random name is generated.

        opened (bool): Should the resulting raster be opened
        or a path.

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
    type_check(opened, [bool], "opened")
    type_check(creation_options, [list], "creation_options")

    raster_list, path_list = ready_io_raster(raster, out_path, overwrite)

    for index, in_raster in enumerate(raster_list):
        metadata = raster_to_metadata(in_raster)
        path = path_list[index]

        if not isinstance(metadata, dict):
            raise Exception("Metadata is not a dict.")

        driver = gdal.GetDriverByName(path_to_driver(path))

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

        array = raster_to_array(raster)

        for band_idx in range(metadata["band_count"]):
            band = copy.GetRasterBand(band_idx + 1)
            band.WriteArray(array[:, :, band_idx])
            band.SetNodataValue(metadata["nodata_value"])

    if opened:
        return copy

    return path


def array_to_raster(
    array: np.ndarray,
    reference: Union[str, gdal.Dataset],
    out_path: Union[str, None] = None,
    overwrite: bool = True,
    opened: bool = False,
    creation_options: list = [],
) -> Union[str, gdal.Dataset]:
    """ Turns a numpy array into a GDAL dataset or exported
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

        opened (bool): Should the resulting raster be opened
        or a path.

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
    type_check(array, [np.ndarray], "array")
    type_check(reference, [str, gdal.Dataset], "reference")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(opened, [bool], "opened")
    type_check(creation_options, [list], "creation_options")

    # Verify the numpy array
    if (
        not isinstance(array, np.ndarray)
        or array.size == 0
        or array.ndim < 2
        or array.ndim > 3
    ):
        raise ValueError(f"Input array is invalid {array}")

    # Parse the driver
    driver_name = "GTiff" if out_path is None else path_to_driver(out_path)
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

    if not isinstance(metadata, dict):
        raise Exception("Metadata is not a dict.")

    # Handle nodata
    input_nodata = None
    if np.ma.is_masked(array) is True:
        input_nodata = array.get_fill_value()

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

        if metadata["nodata_value"] == None and input_nodata is not None:
            band.SetNodataValue(input_nodata)

    if opened:
        return destination

    return output_name


def stack_rasters(
    rasters: List[Union[str, gdal.Dataset]],
    out_path: Union[str, None] = None,
    overwrite: bool = True,
    opened: bool = False,
    dtype: Union[str, None] = None,
    creation_options: list = [],
) -> Union[gdal.Dataset, str]:
    """ Stacks a list of rasters. Must be aligned.
    """
    type_check(rasters, [list], "rasters")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(dtype, [str], "dtype", allow_none=True)
    type_check(opened, [bool], "opened")
    type_check(creation_options, [list], "creation_options")

    if not rasters_are_aligned(rasters, same_extent=True):
        raise ValueError("Rasters are not aligned. Try running align_rasters.")

    overwrite_required(out_path, overwrite)

    # Ensures that all the input rasters are valid.
    in_rasters = to_raster_list(rasters)

    if out_path is not None and path_to_ext(out_path) == ".vrt":
        raise ValueError("Please use stack_rasters_vrt to create vrt files.")

    # Parse the driver
    driver_name = "GTiff" if out_path is None else path_to_driver(out_path)
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

    metadatas = raster_to_metadata(in_rasters)

    if not isinstance(metadatas, list):
        raise Exception("Metadata is the wrong format.")

    datatype = (
        translate_datatypes(dtype)
        if dtype is not None
        else metadatas[0]["datatype_gdal_raw"]
    )

    nodata_values: List[Union[int, float, None]] = []
    nodata_missmatch = False
    nodata_value = None
    total_bands = 0
    for index, raster in enumerate(in_rasters):
        metadata = metadatas[index]

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
        metadata["width"],
        metadata["height"],
        total_bands,
        datatype,
        default_options(creation_options),
    )

    destination.SetProjection(metadata["projection"])
    destination.SetGeoTransform(metadata["transform"])

    bands_added = 0
    for index, raster in enumerate(in_rasters):
        metadata = metadatas[index]
        band_count = metadata["band_count"]

        array = raster_to_array(raster)

        for band_idx in range(band_count):
            dst_band = destination.GetRasterBand(bands_added + 1)
            dst_band.WriteArray(array[:, :, band_idx])

            if nodata_value is not None:
                dst_band.SetNoDataValue(nodata_value)

            bands_added += 1

    if opened:
        return destination

    return output_name


def stack_rasters_vrt(
    rasters,
    out_path: str,
    seperate: bool = True,
    resample_alg="nearest",
    options: list = [],
    opened: bool = False,
    overwrite: bool = True,
    creation_options: list = [],
) -> str:
    """ Stacks a list of rasters into a virtual rasters (.vrt).
    """
    type_check(rasters, [list], "rasters")
    type_check(out_path, [str], "out_path")
    type_check(seperate, [bool], "seperate")
    type_check(resample_alg, [str], "resample_alg")
    type_check(options, [list], "options")
    type_check(opened, [bool], "opened")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")

    resample_algorithm = translate_resample_method(resample_alg)
    options = gdal.BuildVRTOptions(resampleAlg=resample_algorithm, separate=seperate)

    vrt = gdal.BuildVRT(out_path, rasters, options=options)

    if opened:
        return vrt

    return out_path


def copy_raster(
    raster: Union[list, str, gdal.Dataset],
    out_path: Union[list, str],
    overwrite: bool = True,
    dtype: Union[str, None] = None,
    creation_options: list = [],
    opened: bool = False,
) -> Union[list, str]:
    """ Copies a raster to a path. Can be both memory and disk.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(out_path, [list, str], "out_path")
    type_check(overwrite, [bool], "overwrite")
    type_check(dtype, [str], "dtype", allow_none=True)
    type_check(creation_options, [list], "creation_options")
    type_check(opened, [bool], "opened")

    rasters = to_raster_list(raster)

    raise ValueError("Not yet implemented. Sorry")


def seperate_bands(
    raster: Union[str, gdal.Dataset],
    out_names: Union[list] = None,
    out_dir: Union[str, None] = None,
    overwrite: bool = True,
    opened: bool = False,
    prefix: str = "",
    postfix: str = "_bandID",  # add automatically
    creation_options: list = [],
) -> list:
    """ Seperates the bands of a multiband raster.
    """
    type_check(raster, [str, gdal.Dataset], "raster")
    type_check(out_names, [list], "out_names", allow_none=True)
    type_check(out_dir, [str], "out_dir", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(opened, [bool], "opened")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(creation_options, [list], "creation_options")

    raise ValueError("Not yet implemented. Sorry")


def create_raster(
    out_path: Union[str, None] = None,
    reference: Union[str, gdal.Dataset, None] = None,
    top_left: Union[list, tuple, None] = None,
    height: Union[int, None] = None,
    width: Union[int, None] = None,
    pixel_height: Union[int, float, None] = None,
    pixel_width: Union[int, float, None] = None,
    projection: Union[
        int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference, None
    ] = None,
    extent: Union[list, None] = None,
    dtype: Union[str, None] = None,
    filled_with: Union[int, float, None] = 0,
    bands: int = 1,
    nodata_value: Union[float, int, None] = None,
    overwrite: bool = True,
    opened: bool = False,
    creation_options: list = [],
) -> Union[gdal.Dataset, str]:
    """ Returns a new created raster initialised with some value.
    """
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
    type_check(creation_options, [list], "creation_options")

    raise ValueError("Not yet implemented. Sorry")
