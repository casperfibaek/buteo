import sys; sys.path.append('../../')
from uuid import uuid4
from osgeo import gdal, osr, ogr
from typing import Union
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
    to_band_list,
    to_path_list,
    to_raster_list,
    translate_resample_method,
    is_raster,
    translate_datatypes,
)


# TODO: seperate_bands
# TODO: stack_rasters_vrt
# TODO: Create raster
# TODO: Delete raster // clear memory


def raster_to_metadata(
    raster: Union[str, gdal.Dataset],
    simple: bool=True,
) -> dict:
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
    type_check(raster, [str, gdal.Dataset], "raster")
    type_check(simple, [bool], "simple")

    try:
        raster = raster if isinstance(raster, gdal.Dataset) else gdal.Open(raster)
    except:
        raise Exception("Could not read input raster")

    if raster is None:
        raise Exception("Could not read input raster")

    raster_driver = raster.GetDriver()

    metadata = {
        "name": raster.GetDescription(),
        "path": raster.GetDescription(),
        "transform": raster.GetGeoTransform(),
        "projection": raster.GetProjection(),
        "width": raster.RasterXSize,
        "height": raster.RasterYSize,
        "bands": raster.RasterCount,
        "band_count": raster.RasterCount,
        "driver": raster_driver.ShortName,
        "driver_long": raster_driver.LongName,
    }

    basename = os.path.basename(metadata["name"])
    metadata["basename"] = os.path.splitext(basename)[0]
    metadata["ext"] = os.path.splitext(basename)[1]

    original_projection = osr.SpatialReference()
    original_projection.ImportFromWkt(metadata["projection"])
    metadata["projection_osr"] = original_projection

    metadata["size"] = [raster.RasterXSize, raster.RasterYSize]
    metadata["shape"] = (metadata["width"], metadata["height"])
    metadata["pixel_width"] = abs(metadata["transform"][1])
    metadata["pixel_height"] = abs(metadata["transform"][5])
    metadata["xres"] = metadata["pixel_width"]
    metadata["yres"] = metadata["pixel_height"]
    metadata["x_min"] = metadata["transform"][0]
    metadata["y_max"] = metadata["transform"][3]
    metadata["x_max"] = (
        metadata["x_min"]
        + metadata["width"] * metadata["transform"][1] # Handle skew
        + metadata["height"] * metadata["transform"][2]
    )
    metadata["y_min"] = (
        metadata["y_max"]
        + metadata["width"] * metadata["transform"][4] # Handle skew
        + metadata["height"] * metadata["transform"][5]
    )

    x_min = metadata["x_min"]
    y_max = metadata["y_max"]
    x_max = metadata["x_max"]
    y_min = metadata["y_min"]

    band0 = raster.GetRasterBand(1)
    metadata["dtype_gdal_raw"] = band0.DataType
    metadata["datatype_gdal_raw"] = metadata["dtype_gdal_raw"]

    metadata["dtype_gdal"] = gdal.GetDataTypeName(metadata["dtype_gdal_raw"])
    metadata["datatype_gdal"] = metadata["dtype_gdal"]

    metadata["dtype"] = gdal_to_numpy_datatype(band0.DataType)
    metadata["datatype"] = metadata["dtype"]

    metadata["nodata_value"] = band0.GetNoDataValue()

    metadata["extent"] = [x_min, y_max, x_max, y_min]
    metadata["extent_ogr"] = [x_min, x_max, y_min, y_max]
    metadata["extent_gdal_warp"] = [x_min, y_min, x_max, y_max]

    metadata["extent_dict"] = {
        "left": x_min,
        "top": y_max,
        "right": x_max,
        "bottom": y_min,
    }

    if not simple:
        advanced_extents(metadata)

    return metadata


def rasters_are_aligned(
    rasters: list,
    same_extent: bool=False,
    same_dtype: bool=False,
) -> bool:
    """ Verifies if a list of rasters are aligned.

    Args:
        rasters (list): A list of raster, either in gdal.Dataset or a string
        refering to the dataset.

    **kwargs:
        same_extent (bool): Should all the rasters have the same extent?

        same_dtype (bool): Should all the rasters have the same data type?
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

    metas = []

    for raster in rasters:
        metas.append(raster_to_metadata(raster))

    base = {}

    for index, meta in enumerate(metas):
        if index == 0:
            base["projection"] = meta["projection"]
            base["pixel_width"] = meta["pixel_width"]
            base["pixel_height"] = meta["pixel_height"]
            base["x_min"] = meta["x_min"]
            base["y_max"] = meta["y_max"]
            
            base["transform"] = meta["transform"]
            base["height"] = meta["height"]
            base["width"] = meta["width"]
            base["dtype"] = meta["nodata_value"]
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
                if meta["dtype"] != base["dtype"]:
                    return False

    return True


def raster_in_memory(
    raster: Union[str, gdal.Dataset],
) -> bool:
    """ Returns True if the raster is hosted in memory.
    Args:
        raster (list | path | Dataset): The raster to test.

    Returns:
        True if raster is in Memory, False otherwise.
    """
    type_check(raster, [str, gdal.Dataset], "raster")
    metadata = raster_to_metadata(raster)
    
    if metadata["driver"] == "MEM":
        return True
    
    if "/vsimem/" in metadata["path"]:
        return True
    
    return False


def raster_to_memory(
    raster: Union[list, str, gdal.Dataset],
    memory_path: Union[str, None]=None,
    opened: bool=False,
) -> Union[str, gdal.Dataset]:
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

        if memory_path is None or opened:
            results.append(copy)

        results.append(memory_path)

    if not isinstance(raster, list):
        return results[0]
    
    return results


def raster_to_array(
    raster: Union[list, str, gdal.Dataset],
    bands: Union[int, list]=-1,
    filled: bool=False,
    merge: bool=True,
    output_2d: bool=False,
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

    if not rasters_are_aligned(
        internal_rasters,
        same_extent=True,
        same_dtype=False,
    ):
        raise ValueError("Cannot merge rasters that are not aligned, have dissimilar extent or dtype.")

    layers = []
    for in_raster in internal_rasters:
        ref = raster_to_reference(in_raster)
        metadata = raster_to_metadata(ref)
        band_count = metadata["bands"]

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
    raster: Union[list, str, gdal.Dataset],
    out_path: Union[list, str],
    overwrite: bool=True,
    dtype: Union[str, None]=None,
    creation_options: list=[],
    opened: bool=False,
) -> str:
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

    if isinstance(raster, list) and not isinstance(out_path, list):
        raise ValueError("If input raster is a list, out_path must also be a list.")
    
    if isinstance(raster, list) and isinstance(out_path, list):
        if len(raster) != len(out_path):
            raise ValueError("raster list must have same length as out_path list.")

    raster_list = to_raster_list(raster)
    path_list = to_path_list(out_path)

    if out_path is not None:
        creation_options = default_options(creation_options)
    else:
        creation_options = ["BIGTIFF=YES"]

    for index, in_raster in enumerate(raster_list):
        ref = raster_to_reference(in_raster)
        path = path_list[index]

        metadata = raster_to_metadata(ref)

        overwrite_required(path, overwrite)
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
                creation_options,
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
    raster,
    dtype,
    out_path: Union[str, None]=None,
    opened: bool=False,
    creation_options: list=[],
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

    metadata = raster_to_metadata(raster)

    driver_name = None
    if out_path is None:
        driver_name = path_to_driver(out_path)
    else:
        driver_name = "GTiff"
    
    driver = gdal.GetDriverByName(driver_name)

    path = None
    if out_path is None:
        path = f"/vsimem/changed_dtype_{uuid4().int}.tif"
    else:
        path = out_path

    if out_path is not None:
        creation_options = default_options(creation_options)
    else:
        creation_options = ["BIGTIFF=YES"]

    copy = driver.Create(
        path,
        metadata["height"],
        metadata["width"],
        metadata["band_count"],
        translate_datatypes(dtype),
        creation_options,
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
    out_path: Union[str, None]=None,
    overwrite: bool=True,
    opened: bool=False,
    creation_options: list=[],
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
    if not isinstance(array, np.ndarray) or array.size == 0 or array.ndim < 2 or array.ndim > 3:
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

    # If the output is not memory, set compression options.
    if out_path is not None:
        creation_options = default_options(creation_options)
    else:
        creation_options = ["BIGTIFF=YES"]

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
        creation_options,
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
    rasters: Union[list],
    out_path: Union[str, None]=None,
    overwrite: bool=True,
    opened: bool=False,
    dtype: Union[str, None]=None,
    creation_options: list=[],
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

    if out_path is not None:
        creation_options = default_options(creation_options)
    else:
        creation_options = ["BIGTIFF=YES"]

    metadata = raster_to_metadata(in_rasters[0])

    datatype = translate_datatypes(dtype) if dtype is not None else metadata["datatype_gdal_raw"]

    metas = []
    total_bands = 0
    for raster in in_rasters:
        rast_meta = raster_to_metadata(raster)
        total_bands += rast_meta["band_count"]
        metas.append(rast_meta)

    remove_if_overwrite(out_path, overwrite)

    destination = driver.Create(
        output_name,
        metadata["height"],
        metadata["width"],
        total_bands,
        datatype,
        creation_options,
    )

    destination.SetProjection(metadata["projection"])
    destination.SetGeoTransform(metadata["transform"])

    bands_added = 0
    for index, raster in enumerate(in_rasters):
        rast_meta = metas[index]
        band_count = rast_meta["band_count"]

        array = raster_to_array(raster)

        for band_idx in range(band_count):
            dst_band = destination.GetRasterBand(bands_added + 1)
            dst_band.WriteArray(array[:, :, band_idx])
            dst_band.SetNodataValue(rast_meta["nodata_value"])
            
            bands_added += 1

    if opened:
        return destination
    
    return output_name


def stack_rasters_vrt(
    rasters,
    out_path: Union[str, None]=None,
    seperate: bool=True,
    resample_alg="nearest",
    options: list=[],
    opened: bool=False,
    overwrite: bool=True,
    creation_options: list=[],
) -> str:
    """ Stacks a list of rasters into a virtual rasters (.vrt).
    """
    type_check(rasters, [list], "rasters")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(seperate, [bool], "seperate")
    type_check(resample_alg, [str], "resample_alg")
    type_check(options, [list], "options")
    type_check(opened, [bool], "opened")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")

    resample_algorithm = translate_resample_method(resample_alg)

    raise ValueError("Not yet implemented.")

    # options = gdal.BuildVRTOptions(resampleAlg=gdal.GRA_NearestNeighbour, separate=True)

    # return gdal.BuildVRT(out_path, in_rasters, options=options)


def seperate_bands(
    raster: Union[str, gdal.Dataset],
    out_names: Union[list]=None,
    out_dir: Union[str, None]=None,
    overwrite: bool=True,
    opened: bool=False,
    prefix: str="",
    postfix: str="_bandID",
    creation_options: list=[],
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

    raise ValueError("Not yet implemented.")


def create_raster(
    out_path: Union[str, None]=None,
    reference: Union[str, gdal.Dataset, None]=None,
    top_left: Union[list, tuple, None]=None,
    height: Union[int, None]=None,
    width: Union[int, None]=None,
    pixel_height: Union[int, float, None]=None,
    pixel_width: Union[int, float, None]=None,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference, None]=None,
    extent: Union[list, None]=None,
    dtype: Union[str, None]=None,
    filled_with: Union[int, float, None]=0,
    bands: int=1,
    nodata_value: Union[float, int, None]=None,
    overwrite: bool=True,
    opened: bool=False,
    creation_options: list=[],
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
    type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection", allow_none=True)
    type_check(extent, [list], "extent", allow_none=True)
    type_check(dtype, [str], "dtype", allow_none=True)
    type_check(filled_with, [int, float], "filled_with", allow_none=True)
    type_check(bands, [int], "bands")
    type_check(nodata_value, [float, int], "nodata_value", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(opened, [bool], "opened")
    type_check(creation_options, [list], "creation_options")

    raise ValueError("Not yet implemented. Sorry")
