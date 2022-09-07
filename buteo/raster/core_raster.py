"""
### Basic IO functions for working with Rasters ###

This module does standard raster operations related to read, write, and metadata.

TODO:
    * Copy, seperate, expand, create, delete
    * Refactor logic in ready_io_raster
"""

# Standard library
import sys; sys.path.append("../../")
import os

# External
import numpy as np
from osgeo import gdal, osr, ogr

# Internal
from buteo.utils import bbox_utils, core_utils, gdal_utils, gdal_enums



def _open_raster(raster, *, writeable=True):
    """ **INTERNAL**. """
    assert isinstance(raster, (gdal.Dataset, str)), "raster must be a string or a gdal.Dataset"

    if isinstance(raster, gdal.Dataset):
        return raster

    if gdal_utils.is_in_memory(raster) or core_utils.file_exists(raster):

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = gdal.Open(raster, gdal.GF_Write) if writeable else gdal.Open(raster, gdal.GF_Read)
        gdal.PopErrorHandler()

        if not isinstance(opened, gdal.Dataset):
            raise ValueError(f"Input raster is not readable. Received: {raster}")

        return opened

    raise ValueError(f"Input raster does not exists. Received: {raster}")


def open_raster(raster, *, writeable=True, allow_lists=True):
    """
    Opens a raster from a path to a raster. Can be in-memory or local. If a
    gdal.Dataset is passed it is returned. Supports lists. If a list is passed
    a list is returned with the opened raster.

    ## Args:
    `raster` (_gdal.Dataset_/_str_/_list_): A path to a raster or a GDAL dataframe. </br>

    ## Kwargs:
    `writeable` (_bool_): If True, the raster is opened in write mode. (Default: **True**) </br>
    `allow_lists` (_bool_): If True, the input can be a list of rasters. Otherwise, only
    a single raster is allowed. (Default: **True**) </br>

    ## Returns:
    (_gdal.Dataset_/_list_): A gdal.Dataset or a list of gdal.Datasets.
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(writeable, [bool], "writeable")
    core_utils.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(raster, list):
        raise ValueError("Input raster must be a single raster.")

    if not allow_lists:
        return _open_raster(raster, writeable=writeable)

    list_input = core_utils.ensure_list(raster)
    list_return = []

    for in_raster in list_input:
        try:
            list_return.append(_open_raster(in_raster, writeable=writeable))
        except Exception:
            raise ValueError(f"Could not open raster: {in_raster}") from None

    if isinstance(raster, list):
        return list_return

    return list_return[0]


def _raster_to_metadata(raster):
    """ Internal. """
    assert isinstance(raster, (str, gdal.Dataset))

    dataset = open_raster(raster)

    raster_driver = dataset.GetDriver()

    path = dataset.GetDescription()
    basename = os.path.basename(path)
    split_path = os.path.splitext(basename)
    name = split_path[0]
    ext = split_path[1]

    driver = raster_driver.ShortName

    in_memory = gdal_utils.is_in_memory(raster)

    transform = dataset.GetGeoTransform()

    projection_wkt = dataset.GetProjection()
    projection_osr = osr.SpatialReference()
    projection_osr.ImportFromWkt(projection_wkt)

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

    datatype = gdal_enums.translate_gdal_dtype_to_str(datatype_gdal_raw)

    nodata_value = band0.GetNoDataValue()
    has_nodata = nodata_value is not None

    bbox_ogr = [x_min, x_max, y_min, y_max]

    bboxes = bbox_utils.additional_bboxes(bbox_ogr, projection_osr)

    metadata = {
        "path": path,
        "basename": basename,
        "name": name,
        "ext": ext,
        "transform": transform,
        "in_memory": in_memory,
        "projection_wkt": projection_wkt,
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
        "bbox": bbox_ogr,
    }

    for key, value in bboxes.items():
        metadata[key] = value

    def get_bbox_as_vector():
        return bbox_utils.convert_bbox_to_vector(bbox_ogr, projection_osr)


    def get_bbox_as_vector_latlng():
        projection_osr_latlng = osr.SpatialReference()
        projection_osr_latlng.ImportFromEPSG(4326)

        return bbox_utils.convert_bbox_to_vector(metadata["bbox_latlng"], projection_osr_latlng)


    metadata["get_bbox_vector"] = get_bbox_as_vector
    metadata["get_bbox_vector_latlng"] = get_bbox_as_vector_latlng

    return metadata


def raster_to_metadata(raster, *, allow_lists=True):
    """
    Reads a raster from a list of rasters, string or a dataset and returns metadata.

    ## Args:
    `raster` (_gdal.Dataset_/_str_/_list_): A GDAL dataframe or a path to a raster. </br>

    ## Kwargs:
    `allow_lists` (_bool_): If True, the input can be a list of rasters. Otherwise, only
    a single raster is allowed. (Default: **True**) </br>

    ## Returns:
    (_dict_/_list_): A dictionary or list of dictionaries containing metadata.
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    if not allow_lists and isinstance(raster, list):
        raise ValueError("Input raster must be a single raster.")

    if not allow_lists:
        return _raster_to_metadata(raster)

    list_input = core_utils.ensure_list(raster)
    list_return = []

    for in_raster in list_input:
        list_return.append(_raster_to_metadata(in_raster))

    if isinstance(raster, list):
        return list_return

    return list_return[0]


def rasters_are_aligned(
    rasters,
    *,
    same_extent=False,
    same_dtype=False,
    same_nodata=False,
    threshold=0.001,
):
    """
    Verifies if a list of rasters are aligned.

    ## Args:
    `rasters` (_list_): A list of raster, either in gdal.Dataset or a string
    refering to the dataset. </br>

    ## Kwargs:
    `same_extent` (_bool_): Should all the rasters have the same extent? (Default: **False**). </br>
    `same_dtype` (_bool_): Should all the rasters have the same data type? (Default: **False**)
    `same_dtype` (_bool_): Should all the rasters have the same data nodata value? (Default: **False**). </br>
    `threshold` (_float_): The threshold for the difference between the rasters. (Default: **0.001**). </br>

    ## Returns:
    (_bool_): **True** if rasters and aligned and optional parameters are True, **False** otherwise.
    """
    core_utils.type_check(rasters, [[str, gdal.Dataset]], "rasters")
    core_utils.type_check(same_extent, [bool], "same_extent")
    core_utils.type_check(same_dtype, [bool], "same_dtype")
    core_utils.type_check(same_nodata, [bool], "same_nodata")

    if len(rasters) == 1:
        if not gdal_utils.is_raster(rasters[0]):
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
        "projection_wkt": None,
        "projection_osr": None,
    }

    for index, raster in enumerate(rasters):
        meta = _raster_to_metadata(raster)
        if index == 0:
            base["name"] = meta["name"]
            base["projection_wkt"] = meta["projection_wkt"]
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
            if meta["projection_wkt"] != base["projection_wkt"]:
                if meta["projection_osr"].IsSame(base["projection_osr"]):
                    print("WARNING: " + base["name"] + " has the same projection as " + meta["name"] + " but they are written differently in WKT format. Consider using the same definition.")
                else:
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


def raster_to_array(
    raster,
    *,
    bands=-1,
    filled=False,
    bbox=None,
    pixel_offsets=None,
    stack=True,
    split=False,
):
    """
    Turns a path to a raster(s) or a GDAL.Dataset(s) into a **NumPy** array(s).

    ## Args:
    (_gdal.Dataset_/_str_/_list_): The raster(s) to convert.

    ## Kwargs:
    `bands` (_list_/_str_/_int_): The bands from the raster to turn
    into a numpy array. Can be "all", "ALL", a list of ints or a
    single int. </br>
    `filled` (_bool_): If the array contains nodata values. Should the
    resulting array be a filled numpy array or a masked array? </br>
    `bbox` (_list_): A list of `[xmin, xmax, ymin, ymax]` to use as the
    extent of the raster. Uses coordinates and the **OGR** format. </br>
    `pixel_offsets` (_list_): A list of [x_offset, y_offset, x_size, y_size] to use as
    the extent of the raster. Uses pixel offsets and the **OGR** format. </br>
    `stack` (_bool_): If True, stacks the input rasters into a single array. Only works if
    the rasters are aligned. (Default: **True**) </br>
    `split` (_bool_): If True, splits the bands of the input rasters into seperate arrays. (Default: **False**)

    ## Returns:
    (_np.ndarray_): A numpy array in the 3D channel-last format unless output_2D is
    specified. </br>
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(bands, [int, list], "bands")
    core_utils.type_check(filled, [bool], "filled")
    core_utils.type_check(bbox, [list, None], "bbox")
    core_utils.type_check(pixel_offsets, [list, None], "pixel_offsets")
    core_utils.type_check(stack, [bool], "stack")
    core_utils.type_check(split, [bool], "split")

    internal_rasters = core_utils.ensure_list(raster)

    if not gdal_utils.is_raster_list(internal_rasters):
        raise ValueError(f"An input raster is invalid. {internal_rasters}")

    internal_rasters = gdal_utils.get_path_from_dataset_list(internal_rasters, dataset_type="raster")

    if stack and not rasters_are_aligned(internal_rasters, same_extent=True, same_dtype=False):
        raise ValueError(
            "Cannot merge rasters that are not aligned, have dissimilar extent or dtype, when stack=True."
        )

    layers = []
    nodata_values = []
    for in_raster in internal_rasters:

        if not gdal_utils.is_raster(in_raster):
            raise ValueError(f"Invalid raster: {in_raster}")

        ref = open_raster(in_raster)

        metadata = raster_to_metadata(ref)
        band_count = metadata["band_count"]
        band_arrs = []

        if band_count == 0:
            raise ValueError("The input raster does not have any valid bands.")

        internal_bands = gdal_utils.to_band_list(bands, metadata["band_count"])

        for band in internal_bands:
            band_ref = ref.GetRasterBand(band + 1)
            band_nodata_value = band_ref.GetNoDataValue()

            nodata_values.append(band_nodata_value)

            if pixel_offsets is not None:
                arr = band_ref.ReadAsArray(
                    pixel_offsets[0], # x_offset
                    pixel_offsets[1], # y_offset
                    pixel_offsets[2], # x_size
                    pixel_offsets[3], # y_size
                )
            elif bbox is not None:
                if not bbox_utils.bboxes_intersect(metadata["extent_ogr"], bbox):
                    raise ValueError("Extent is outside of raster.")

                x_offset, y_offset, x_size, y_size = bbox_utils.get_pixel_offsets(metadata["transform"], bbox)

                arr = band_ref.ReadAsArray(x_offset, y_offset, x_size, y_size)
            else:
                arr = band_ref.ReadAsArray()

            if band_nodata_value is not None:
                arr = np.ma.array(arr, mask=arr == band_nodata_value)
                arr.fill_value = band_nodata_value

                if filled:
                    arr = arr.filled(band_nodata_value)

            band_arrs.append(arr)

        if split:
            layers.append(band_arrs)
        elif band_nodata_value is None:
            layers.append(np.dstack(band_arrs))
        else:
            layers.append(np.ma.dstack(band_arrs))

        ref = None

    if split:
        if stack:
            return layers

        output = []
        for layer in layers:
            for band in layer:
                output.append(band)

        return output

    if not core_utils.is_list_all_the_same(nodata_values):
        fill_value = gdal_enums.get_default_nodata_value(layers[0].dtype)
        for idx, layer in enumerate(layers):
            layer[idx].fill_value = fill_value

    output = layers

    if stack:
        if core_utils.is_list_all_val(nodata_values, None):
            output = np.dstack(layers)
        else:
            output = np.ma.dstack(layers)

    return output


def array_to_raster(
    array,
    *,
    reference,
    out_path=None,
    set_nodata="arr",
    allow_mismatches=False,
    overwrite=True,
    creation_options=None,
):
    """
    Turns a **NumPy** array into a **GDAL** dataset or exported
    as a raster using a reference raster.

    ## Args:
    `array` (_np.ndarray_): The numpy array to convert. </br>
    `reference` (_str_/_gdal.Dataset_): The reference raster to use for the output. </br>

    ## Kwargs:
    `out_path` (_path_): The destination to save to. (Default: **None**)</br>
    `set_nodata` (_bool_/_float_/_int_): Can be set to: (Default: **arr**)</br>
    `allow_mismatches` (_bool_): If True, the array can have a different shape than the reference raster.
    `overwrite` (_bool_): If the file exists, should it be overwritten? (Default: **True**) </br>
    &emsp; • **"arr"**: The nodata value will be the same as the **NumPy** array. </br>
    &emsp; • **"ref"**: The nodata value will be the same as the reference raster. </br>
    &emsp; • **"value"**: The nodata value will be the value provided. </br>
    `creation_options` (_list_): List of **GDAL** creation options. Defaults are: </br>
    &emsp; • "TILED=YES" </br>
    &emsp; • "NUM_THREADS=ALL_CPUS" </br>
    &emsp; • "BIGG_TIF=YES" </br>
    &emsp; • "COMPRESS=LZW" </br>

    ## Returns:
    (_str_): The filepath to the newly created raster(s).
    """
    core_utils.type_check(array, [np.ndarray, np.ma.MaskedArray], "array")
    core_utils.type_check(reference, [str, gdal.Dataset], "reference")
    core_utils.type_check(out_path, [str, None], "out_path")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(set_nodata, [int, float, str, None], "set_nodata")
    core_utils.type_check(creation_options, [[str], None], "creation_options")

    # Verify the numpy array
    if (
        not isinstance(array, (np.ndarray, np.ma.MaskedArray))
        or array.size == 0
        or array.ndim < 2
        or array.ndim > 3
    ):
        raise ValueError(f"Input array is invalid {array}")

    if set_nodata != "arr" and set_nodata != "ref":
        core_utils.type_check(set_nodata, [int, float], "set_nodata")

    # Parse the driver
    driver_name = "GTiff" if out_path is None else gdal_utils.path_to_driver_raster(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    # How many bands?
    bands = 1
    if array.ndim == 3:
        bands = array.shape[2]

    output_name = None
    if out_path is None:
        output_name = gdal_utils.create_memory_path("array_to_raster.tif", add_uuid=True)
    else:
        output_name = out_path

    core_utils.remove_if_required(output_name, overwrite)

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

    destination_dtype = gdal_enums.translate_str_to_gdal_dtype(array.dtype)

    # Weird double issue with GDAL and numpy. Cast to float or int
    if input_nodata is not None:
        input_nodata = float(input_nodata)
        if (input_nodata).is_integer() is True:
            input_nodata = int(input_nodata)


    if metadata["width"] != array.shape[1] or metadata["height"] != array.shape[0]:
        if not allow_mismatches:
            raise ValueError(f"Input array and raster are not of equal size. Array: {array.shape[:2]} Raster: {metadata['width'], metadata['height']}")

        print("WARNING: Input array and raster are not of equal size.")

    destination = driver.Create(
        output_name,
        array.shape[1],
        array.shape[0],
        bands,
        destination_dtype,
        gdal_utils.default_creation_options(creation_options),
    )

    destination.SetProjection(metadata["projection_wkt"])
    destination.SetGeoTransform(metadata["transform"])

    for band_idx in range(bands):
        band = destination.GetRasterBand(band_idx + 1)
        if bands > 1 or array.ndim == 3:
            band.WriteArray(array[:, :, band_idx])
        else:
            band.WriteArray(array)

        if set_nodata == "ref" and reference_nodata is not None:
            band.SetNoDataValue(reference_nodata)
        elif set_nodata == "arr" and input_nodata is not None:
            band.SetNoDataValue(input_nodata)
        elif isinstance(set_nodata, (int, float)):
            band.SetNoDataValue(set_nodata)

    destination.FlushCache()
    destination = None

    return output_name


def _raster_set_datatype(
    raster,
    dtype_str,
    out_path=None,
    *,
    overwrite=True,
    creation_options=None,
):
    """ **INTERNAL**. """
    assert isinstance(raster, (str, gdal.Dataset)), "raster must be a string or a GDAL.Dataset."
    assert isinstance(dtype_str, str), "dtype_str must be a string."
    assert len(dtype_str) > 0, "dtype_str must be a non-empty string."

    if not gdal_utils.is_raster(raster):
        raise ValueError(f"Unable to open input raster: {raster}")

    ref = open_raster(raster)
    metadata = raster_to_metadata(ref)

    path = ""
    if out_path is None:
        path = gdal_utils.create_memory_path(metadata["basename"], add_uuid=True)

    elif core_utils.folder_exists(out_path):
        path = os.path.join(out_path, os.path.basename(gdal_utils.get_path_from_dataset(ref)))

    elif core_utils.folder_exists(core_utils.path_to_folder(out_path)):
        path = out_path

    elif core_utils.is_valid_mem_path(out_path):
        path = out_path

    else:
        raise ValueError(f"Unable to find output folder: {out_path}")

    driver_name = gdal_utils.path_to_driver_raster(path)
    driver = gdal.GetDriverByName(driver_name)

    if driver is None:
        raise ValueError(f"Unable to get driver for raster: {raster}")

    core_utils.remove_if_required(path, overwrite)

    copy = driver.Create(
        path,
        metadata["height"],
        metadata["width"],
        metadata["band_count"],
        gdal_enums.translate_str_to_gdal_dtype(dtype_str),
        gdal_utils.default_creation_options(creation_options),
    )

    if copy is None:
        raise ValueError(f"Unable to create output raster: {path}")

    copy.SetProjection(metadata["projection_wkt"])
    copy.SetGeoTransform(metadata["transform"])

    array = raster_to_array(ref)

    for band_idx in range(metadata["band_count"]):
        band = copy.GetRasterBand(band_idx + 1)
        band.WriteArray(array[:, :, band_idx])

        if metadata["nodata_value"] is not None:
            band.SetNoDataValue(metadata["nodata_value"])

    ref = None

    return path


def raster_set_datatype(
    raster,
    dtype,
    out_path=None,
    *,
    overwrite=True,
    allow_lists=True,
    creation_options=None,
):
    """
    Changes the datatype of a raster.

    ## Args:
    `raster` (_str_/_gdal.Dataset_/_list_): The raster to change the datatype of. </br>
    `dtype` (_str_): The new datatype. </br>

    ## Kwargs:
    `out_path` (_path_/_list_): The destination to save to. (Default: **None**)</br>
    `overwrite` (_bool_): If the file exists, should it be overwritten? (Default: **True**) </br>
    `allow_lists` (_bool_): If True, the input can be a list of rasters. Otherwise, only single rasters
    are allowed. (Default: **True**) </br>
    `creation_options` (_list_): List of **GDAL** creation options. Defaults are: </br>
        * "TILED=YES"
        * "NUM_THREADS=ALL_CPUS"
        * "BIGG_TIF=YES"
        * "COMPRESS=LZW"

    ## Returns:
    (_str_/_list_): The filepath or list of filepaths to the newly created raster(s).
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(dtype, [str], "dtype")
    core_utils.type_check(out_path, [list, str, None], "out_path")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(allow_lists, [bool], "allow_lists")
    core_utils.type_check(creation_options, [list, None], "creation_options")

    if not allow_lists:
        if isinstance(raster, list):
            raise ValueError("allow_lists is False, but the input raster is a list.")

        return _raster_set_datatype(
            raster,
            dtype,
            out_path=out_path,
            overwrite=overwrite,
            creation_options=creation_options,
        )

    add_uuid = out_path is None

    raster_list = core_utils.ensure_list(raster)
    path_list = gdal_utils.create_output_path_list(raster_list, out_path, overwrite=overwrite, add_uuid=add_uuid)

    output = []
    for index, in_raster in enumerate(raster_list):
        path = _raster_set_datatype(
            in_raster,
            dtype,
            out_path=path_list[index],
            overwrite=overwrite,
            creation_options=gdal_utils.default_creation_options(creation_options),
        )

        output.append(path)

    if isinstance(raster, list):
        return output

    return output[0]



def stack_rasters(
    rasters,
    out_path=None,
    *,
    overwrite=True,
    dtype=None,
    creation_options=None,
):
    """
    Stacks a list of rasters. Must be aligned.

    ## Args:
    `rasters` (_list_): List of rasters to stack. </br>

    ## Kwargs
    `out_path` (_str_/_None_): The destination to save to. (Default: **None**)</br>
    `overwrite` (_bool_): If the file exists, should it be overwritten? (Default: **True**) </br>
    `dtype` (_str_): The data type of the output raster. (Default: **None**)</br>
    `creation_options` (_list_): List of **GDAL** creation options. Defaults are: </br>
    &emsp; • "TILED=YES" </br>
    &emsp; • "NUM_THREADS=ALL_CPUS" </br>
    &emsp; • "BIGG_TIF=YES" </br>
    &emsp; • "COMPRESS=LZW" </br>

    ## Returns:
    (_str_/_list_): The filepath(s) to the newly created raster(s).
    """
    core_utils.type_check(rasters, [[str, gdal.Dataset]], "rasters")
    core_utils.type_check(out_path, [str, None], "out_path")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(dtype, [str, None], "dtype")
    core_utils.type_check(creation_options, [[str], None], "creation_options")

    assert gdal_utils.is_raster_list(rasters), "Input rasters must be a list of rasters."

    if not rasters_are_aligned(rasters, same_extent=True):
        raise ValueError("Rasters are not aligned. Try running align_rasters.")

    # Ensures that all the input rasters are valid.
    raster_list = gdal_utils.get_path_from_dataset_list(rasters)

    if out_path is not None and core_utils.path_to_ext(out_path) == ".vrt":
        raise ValueError("Please use stack_rasters_vrt to create vrt files.")

    # Parse the driver
    driver_name = "GTiff" if out_path is None else gdal_utils.path_to_driver_raster(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = gdal_utils.create_memory_path("stack_rasters.tif", add_uuid=True)
    else:
        output_name = out_path

    core_utils.remove_if_required(output_name, overwrite)

    raster_dtype = raster_to_metadata(raster_list[0])["datatype_gdal_raw"]

    datatype = raster_dtype
    if dtype is not None:
        datatype = gdal_enums.translate_str_to_gdal_dtype(dtype)

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

    destination = driver.Create(
        output_name,
        metadatas[0]["width"],
        metadatas[0]["height"],
        total_bands,
        datatype,
        gdal_utils.default_creation_options(creation_options),
    )

    destination.SetProjection(metadatas[0]["projection_wkt"])
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
    options=None,
    overwrite=True,
    reference=None,
    creation_options=None,
):
    """
    Stacks a list of rasters into a virtual rasters (.vrt).

    ## Args:
    `rasters` (_list_): List of rasters to stack. </br>
    `out_path` (_str_): The destination to save to. </br>

    ## Kwargs:
    `seperate` (_bool_): If the raster bands should be seperated. (Default: **True**) </br>
    `resample_alg` (_str_): The resampling algorithm to use. (Default: **nearest**) </br>
    `options` (_list_): List of VRT options for GDAL. (Default: **()** </br>
    `overwrite` (_bool_): If the file exists, should it be overwritten? (Default: **True**) </br>
    `reference` (_str_): The reference raster to use. (Default: **None**) </br>
    `creation_options` (_list_): List of **GDAL** creation options. Defaults are: </br>
    * &emsp; • "TILED=YES"
    * &emsp; • "NUM_THREADS=ALL_CPUS"
    * &emsp; • "BIGG_TIF=YES"
    * &emsp; • "COMPRESS=LZW"

    ## Returns:
    (_str_): The filepath to the newly created VRT raster.
    """
    core_utils.type_check(rasters, [[str, gdal.Dataset]], "rasters")
    core_utils.type_check(out_path, [str], "out_path")
    core_utils.type_check(seperate, [bool], "seperate")
    core_utils.type_check(resample_alg, [str], "resample_alg")
    core_utils.type_check(options, [tuple, None], "options")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")

    resample_algorithm = gdal_enums.translate_resample_method(resample_alg)

    if reference is not None:
        meta = raster_to_metadata(reference)
        options = gdal.BuildVRTOptions(
            resampleAlg=resample_algorithm,
            separate=seperate,
            outputBounds=bbox_utils.convert_ogr_bbox_to_gdal_bbox(meta["bbox"]),
            xRes=meta["pixel_width"],
            yRes=meta["pixel_height"],
            targetAlignedPixels=True,
        )
    else:
        options = gdal.BuildVRTOptions(resampleAlg=resample_algorithm, separate=seperate)

    if options is None:
        options = ()

    vrt = gdal.BuildVRT(out_path, rasters, options=options)

    if vrt is None:
        raise ValueError(f"Error while creating VRT from rasters: {rasters}")

    return out_path


def rasters_intersect(raster1, raster2):
    """
    Checks if two rasters intersect using their latlong boundaries.

    ## Args:
    `raster1` (_str_/gdal.Dataset): The first raster. </br>
    `raster2` (_str_/gdal.Dataset): The second raster. </br>

    ## Returns:
    (_bool_): If the rasters intersect.
    """
    core_utils.type_check(raster1, [str, gdal.Dataset, [str, gdal.Dataset]], "raster1")
    core_utils.type_check(raster2, [list, str, gdal.Dataset], "raster2")

    meta1 = raster_to_metadata(raster1)
    meta2 = raster_to_metadata(raster2)

    return meta1["bbox_geom_latlng"].Intersects(meta2["bbox_geom_latlng"])


def rasters_intersection(raster1, raster2):
    """
    Get the latlng intersection of two rasters.

    ## Args:
    `raster1` (_str_/_gdal.Dataset_): The first raster. </br>
    `raster2` (_str_/_gdal.Dataset_): The second raster. </br>

    ## Returns:
    (_ogr.DataSource_): The latlng intersection of the two rasters.
    """
    core_utils.type_check(raster1, [str, gdal.Dataset, [str, gdal.Dataset]], "raster1")
    core_utils.type_check(raster2, [str, gdal.Dataset, [str, gdal.Dataset]], "raster2")

    if not rasters_intersect(raster1, raster2):
        raise ValueError("Rasters do not intersect.")

    meta1 = raster_to_metadata(raster1)
    meta2 = raster_to_metadata(raster2)

    intersection = meta1["bbox_geom_latlng"].Intersection(meta2["bbox_geom_latlng"])

    return gdal_utils.convert_geom_to_vector(intersection)


def get_overlap_fraction(raster1, raster2):
    """
    Get the fraction of the overlap between two rasters. (e.g. 0.9 for mostly overlapping rasters)

    ## Args:
    `raster1` (_str_/_gdal.Dataset_): The master raster. </br>
    `raster2` (_str_/_gdal.Dataset_): The test raster. </br>

    ## Returns:
    (_float_): A value representing the degree of overlap **(0-1)**
    """
    core_utils.type_check(raster1, [str, gdal.Dataset, [str, gdal.Dataset]], "raster1")
    core_utils.type_check(raster2, [str, gdal.Dataset, [str, gdal.Dataset]], "raster2")

    if not rasters_intersect(raster1, raster2):
        return 0.0

    meta1 = raster_to_metadata(raster1)["bbox_geom_latlng"]
    meta2 = raster_to_metadata(raster2)["bbox_geom_latlng"]

    try:
        intersection = meta1.Intersection(meta2)
    except Exception:
        return 0.0

    overlap = intersection.GetArea() / meta1.GetArea()

    return overlap


def create_raster_from_array(arr, out_path=None, pixel_size=10.0, x_min=0.0, y_max=0.0, projection="EPSG:3857", creation_options=None, overwrite=True):
    """ Create a raster from a numpy array. """
    core_utils.type_check(arr, [np.ndarray, np.ma.MaskedArray], "arr")
    core_utils.type_check(out_path, [str, None], "out_path")
    core_utils.type_check(pixel_size, [int, float, [int, float]], "pixel_size")
    core_utils.type_check(x_min, [int, float], "x_min")
    core_utils.type_check(y_max, [int, float], "y_max")
    core_utils.type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    core_utils.type_check(creation_options, [[str], None], "creation_options")
    core_utils.type_check(overwrite, [bool], "overwrite")

    assert arr.ndim == 2 or arr.ndim == 3, "Array must be 2 or 3 dimensional (3rd dimension considered bands.)"

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Parse the driver
    driver_name = "GTiff" if out_path is None else gdal_utils.path_to_driver_raster(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = gdal_utils.create_memory_path("raster_from_array.tif", add_uuid=True)
    else:
        output_name = out_path

    core_utils.remove_if_required(output_name, overwrite)

    height = arr.shape[0]
    width = arr.shape[1]
    bands = arr.shape[2]

    destination = driver.Create(
        output_name,
        width,
        height,
        bands,
        gdal_enums.translate_str_to_gdal_dtype(arr.dtype.name),
        gdal_utils.default_creation_options(creation_options),
    )

    parsed_projection = gdal_utils.parse_projection(projection, return_wkt=True)

    destination.SetProjection(parsed_projection)

    pixel_width = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[0]
    pixel_height = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[1]

    transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height] # negative for north-up

    destination.SetGeoTransform(transform)

    nodata = None
    if isinstance(arr, np.ma.MaskedArray):
        nodata = arr.fill_value

    for idx in range(0, bands):
        dst_band = destination.GetRasterBand(idx + 1)
        dst_band.WriteArray(arr[:, :, idx])

        if nodata is not None:
            dst_band.SetNoDataValue(nodata)

    return output_name
