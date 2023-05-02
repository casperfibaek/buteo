"""
### Basic IO functions for working with Rasters ###

This module does standard raster operations related to read, write, and metadata.
"""

# TODO: Copy, seperate, expand

# Standard library
import sys; sys.path.append("../../")
import os
from typing import List, Optional, Union, Tuple
from uuid import uuid4
import warnings

# External
import numpy as np
from osgeo import gdal, osr, ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_gdal_translate,
    utils_bbox,
    utils_path,
    utils_gdal_projection,
)



def _raster_open(
    raster: Union[str, gdal.Dataset],
    *,
    writeable: bool = True,
) -> gdal.Dataset:
    """ **INTERNAL**. """
    assert isinstance(raster, (gdal.Dataset, str)), "raster must be a string or a gdal.Dataset"

    if isinstance(raster, gdal.Dataset):
        return raster

    if isinstance(raster, str) and raster.startswith("/vsizip/"):
        writeable = False

    if utils_gdal._check_dataset_in_memory(raster) or utils_base.file_exists(raster):

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = gdal.Open(raster, gdal.GF_Write) if writeable else gdal.Open(raster, gdal.GF_Read)
        gdal.PopErrorHandler()

        if not isinstance(opened, gdal.Dataset):
            raise ValueError(f"Input raster is not readable. Received: {raster}")

        if opened.GetDescription() == "":
            opened.SetDescription(raster)

        if opened.GetProjectionRef() == "":
            opened.SetProjection(utils_gdal_projection._get_default_projection())
            warnings.warn(f"WARNING: Input raster {raster} has no projection. Setting to default: EPSG:4326.", UserWarning)

        return opened

    raise ValueError(f"Input raster does not exists. Received: {raster}")


def raster_open(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    *,
    writeable=True,
    allow_lists=True,
) -> Union[gdal.Dataset, List[gdal.Dataset]]:
    """
    Parameters
    ----------
    raster : Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
        A path to a raster or a GDAL dataframe.

    writeable : bool, optional
        If True, the raster is opened in write mode. Default: True.

    allow_lists : bool, optional
        If True, the input can be a list of rasters. Otherwise,
        only a single raster is allowed. Default: True.

    Returns
    -------
    Union[gdal.Dataset, List[gdal.Dataset]]
        A gdal.Dataset or a list of gdal.Datasets.
    """
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(writeable, [bool], "writeable")
    utils_base.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(raster, (list, tuple)):
        raise ValueError("Input raster must be a single raster. Not a list or tuple.")

    if not allow_lists:
        return _raster_open(raster, writeable=writeable)

    list_input = utils_base._get_variable_as_list(raster)
    list_return = []

    for in_raster in list_input:
        try:
            list_return.append(_raster_open(in_raster, writeable=writeable))
        except Exception:
            raise ValueError(f"Could not open raster: {in_raster}") from None

    if isinstance(raster, list):
        return list_return

    return list_return[0]


def _raster_get_projection(
    raster: Union[str, gdal.Dataset],
    wkt: bool = True,
) -> str:
    """
    Get the projection from a dataset, either as WKT or osr.

    Parameters
    ----------
    raster : str or gdal.Dataset
        A path to a raster or a gdal.Dataset.

    wkt : bool, optional
        If True, returns the projection as WKT. Default: True.

    Returns
    -------
    str
        The projection of the input raster in the specified format.
    """
    dataset = raster_open(raster)

    if wkt:
        return dataset.GetProjectionRef()

    return dataset.GetProjection()


def _raster_to_metadata(
    raster: Union[str, gdal.Dataset],
) -> dict:
    """ Internal. """
    utils_base.type_check(raster, [str, gdal.Dataset], "raster")

    dataset = raster_open(raster)

    raster_driver = dataset.GetDriver()

    path = dataset.GetDescription()
    basename = os.path.basename(path)
    split_path = os.path.splitext(basename)
    name = split_path[0]
    ext = split_path[1]

    driver = raster_driver.ShortName

    in_memory = utils_gdal._check_dataset_in_memory(raster)

    transform = dataset.GetGeoTransform()

    projection_wkt = dataset.GetProjection()
    projection_osr = osr.SpatialReference()
    projection_osr.ImportFromWkt(projection_wkt)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    band_count = dataset.RasterCount

    size = [dataset.RasterXSize, dataset.RasterYSize]
    shape = (height, width, band_count)

    pixel_width = abs(transform[1])
    pixel_height = abs(transform[5])

    x_min = transform[0]
    y_max = transform[3]

    x_max = x_min + width * transform[1] + height * transform[2]  # Handle skew
    y_min = y_max + width * transform[4] + height * transform[5]  # Handle skew

    band0 = dataset.GetRasterBand(1)

    datatype_gdal_raw = band0.DataType
    datatype_gdal = gdal.GetDataTypeName(datatype_gdal_raw)

    datatype = utils_gdal_translate._translate_gdal_dtype_to_str(datatype_gdal_raw)

    nodata_value = band0.GetNoDataValue()
    has_nodata = nodata_value is not None

    bbox_ogr = [x_min, x_max, y_min, y_max]

    bboxes = utils_bbox._additional_bboxes(bbox_ogr, projection_osr)

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
        "dtype": datatype,
        "dtype_gdal": datatype_gdal,
        "dtype_gdal_raw": datatype_gdal_raw,
        "datatype": datatype,
        "datatype_gdal": datatype_gdal,
        "datatype_gdal_raw": datatype_gdal_raw,
        "nodata_value": nodata_value,
        "has_nodata": has_nodata,
        "is_raster": True,
        "is_vector": False,
        "bbox": bbox_ogr,
        "extent": bbox_ogr,
    }

    for key, value in bboxes.items():
        metadata[key] = value

    def get_bbox_as_vector():
        return utils_bbox._get_vector_from_bbox(bbox_ogr, projection_osr)

    def get_bbox_as_vector_latlng():
        latlng_wkt = utils_gdal_projection._get_default_projection()
        projection_osr_latlng = osr.SpatialReference()
        projection_osr_latlng.ImportFromWkt(latlng_wkt)

        return utils_bbox._get_vector_from_bbox(metadata["bbox_latlng"], projection_osr_latlng)

    metadata["get_bbox_vector"] = get_bbox_as_vector
    metadata["get_bbox_vector_latlng"] = get_bbox_as_vector_latlng

    return metadata


def raster_to_metadata(
    raster: Union[str, gdal.Dataset, List[str], List[gdal.Dataset]],
    *,
    allow_lists: bool = True,
) -> Union[dict, List[dict]]:
    """
    Reads metadata from a raster dataset or a list of raster datasets, and returns a dictionary or a list of dictionaries
    containing metadata information for each raster.

    Parameters
    ----------
    raster : str or gdal.Dataset or list
        A path to a raster or a gdal.Dataset, or a list of paths to rasters.

    allow_lists : bool, optional
        If True, allows the input to be a list of rasters. Otherwise, only a single raster is allowed. Default: True.

    Returns
    -------
    dict or list of dict
        A dictionary or a list of dictionaries containing metadata information for each raster.
    """
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    if not allow_lists and isinstance(raster, list):
        raise ValueError("Input raster must be a single raster.")

    if not allow_lists:
        return _raster_to_metadata(raster)

    list_input = utils_base._get_variable_as_list(raster)
    list_return = []

    for in_raster in list_input:
        list_return.append(_raster_to_metadata(in_raster))

    if isinstance(raster, list):
        return list_return

    return list_return[0]


def check_are_rasters_are_aligned(
    rasters: List[Union[str, gdal.Dataset]],
    *,
    same_extent: bool = True,
    same_dtype: bool = False,
    same_nodata: bool = False,
    threshold: float = 0.001,
) -> bool:
    """
    Verifies whether a list of rasters are aligned.

    Parameters
    ----------
    rasters : list
        A list of rasters, either in gdal.Dataset or a string referring to the dataset.

    same_extent : bool, optional
        If True, all the rasters should have the same extent. Default: True.

    same_dtype : bool, optional
        If True, all the rasters should have the same data type. Default: False.

    same_nodata : bool, optional
        If True, all the rasters should have the same nodata value. Default: False.

    threshold : float, optional
        The threshold for the difference between the rasters. Default: 0.001.

    Returns
    -------
    bool
        True if rasters are aligned and optional parameters are True, False otherwise.
    """

    utils_base.type_check(rasters, [[str, gdal.Dataset]], "rasters")
    utils_base.type_check(same_extent, [bool], "same_extent")
    utils_base.type_check(same_dtype, [bool], "same_dtype")
    utils_base.type_check(same_nodata, [bool], "same_nodata")

    if len(rasters) == 1:
        if not utils_gdal._check_is_raster(rasters[0]):
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
            base["projection_osr"] = meta["projection_osr"]
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
                    warnings.warn(base["name"] + " has the same projection as " + meta["name"] + " but they are written differently in WKT format. Consider using the same definition.", UserWarning)
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


def raster_has_nodata(
    raster: Union[str, gdal.Dataset],
) -> bool:
    """
    Verifies whether a raster has any nodata values.

    Parameters
    ----------
    raster : str or gdal.Dataset
        A raster, either in gdal.Dataset or a string referring to the dataset.

    Returns
    -------
    bool
        True if raster has nodata values, False otherwise.
    """
    utils_base.type_check(raster, [str, gdal.Dataset], "raster")

    ref = raster_open(raster)
    band_count = ref.RasterCount
    for band in range(1, band_count + 1):
        band_ref = ref.GetRasterBand(band)
        if band_ref.GetNoDataValue() is not None:
            ref = None

            return True

    ref = None
    return False


def raster_has_nodata_list(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
) -> bool:
    """
    Verifies whether a list of rasters have any nodata values.

    Parameters
    ----------
    rasters : list
        A list of rasters, either in gdal.Dataset or a string referring to the dataset.

    Returns
    -------
    bool
        True if all rasters have nodata values, False otherwise.
    """
    utils_base.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    internal_rasters = utils_base._get_variable_as_list(rasters)
    assert utils_gdal._check_is_raster_list(internal_rasters), "Invalid raster list."

    has_nodata = False
    for in_raster in internal_rasters:
        if raster_has_nodata(in_raster):
            has_nodata = True
            break

    return has_nodata


def check_rasters_have_same_nodata(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
) -> bool:
    """
    Verifies whether a list of rasters have the same nodata values.

    Parameters
    ----------
    rasters : list
        A list of rasters, either in gdal.Dataset or a string referring to the dataset.

    Returns
    -------
    bool
        True if all rasters have the same nodata value, False otherwise.
    """
    utils_base.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    internal_rasters = utils_base._get_variable_as_list(rasters)
    assert utils_gdal._check_is_raster_list(internal_rasters), "Invalid raster list."

    nodata_values = []
    for in_raster in internal_rasters:
        ref = raster_open(in_raster)
        band_count = ref.RasterCount
        for band in range(1, band_count + 1):
            band_ref = ref.GetRasterBand(band)
            nodata_values.append(band_ref.GetNoDataValue())

        ref = None

    return len(set(nodata_values)) == 1


def _get_first_nodata_value(
    raster: Union[float, int, None],
):
    """
    Gets the first nodata value from a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset
        The raster to get the nodata value from.

    Returns
    -------
    float or None
        The nodata value if found, or None if not found.
    """
    utils_base.type_check(raster, [str, gdal.Dataset], "raster")

    nodata = None

    ref = raster_open(raster)
    band_count = ref.RasterCount
    for band in range(1, band_count + 1):
        band_ref = ref.GetRasterBand(band)
        nodata_value = band_ref.GetNoDataValue()
        if nodata_value is not None:
            nodata = nodata_value
            break

    ref = None
    return nodata


def raster_count_bands_list(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
) -> int:
    """
    Counts the number of bands in a list of rasters.

    Parameters
    ----------
    rasters : list
        A list of rasters, either in gdal.Dataset or a string referring to the dataset.

    Returns
    -------
    int
        The number of bands in the rasters.
    """
    utils_base.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    internal_rasters = utils_base._get_variable_as_list(rasters)
    assert utils_gdal._check_is_raster_list(internal_rasters), "Invalid raster list."

    band_count = 0
    for in_raster in internal_rasters:
        ref = raster_open(in_raster)
        band_count += ref.RasterCount
        ref = None

    return band_count


def raster_to_array(
    raster: Union[gdal.Dataset, str, List[Union[str, gdal.Dataset]]],
    *,
    bands: Union[List[int], str, int] = 'all',
    masked: Union[bool, str] = "auto",
    filled: bool = False,
    fill_value: Optional[Union[int, float]] = None,
    bbox: Optional[List[float]] = None,
    pixel_offsets: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
    cast: Optional[Union[np.dtype, str]] = None,
) -> np.ndarray:
    """
    Converts a raster or a list of rasters into a NumPy array.

    Parameters
    ----------
    raster : gdal.Dataset or str or list
        Raster(s) to convert.

    bands : list or str or int, optional
        Bands from the raster to convert to a numpy array. Can be "all", an int,
        or a list of integers, or a single integer. Please note that bands are 1-indexed.
        Default: "all".

    masked : bool or str, optional
        If the array contains nodata values, determines whether the resulting
        array should be a masked numpy array or a regular numpy array. If "auto",
        the array will be masked only if the raster has nodata values. Default: "auto".

    filled : bool, optional
        If the array contains nodata values, determines whether the resulting
        array should be a filled numpy array or a masked array. Default: False.

    fill_value : int or float, optional
        Value to fill the array with if filled is True. If None, the nodata value
        of the raster is used. Default: None.

    bbox : list, optional
        A list of `[xmin, xmax, ymin, ymax]` to use as the extent of the raster.
        Uses coordinates and the OGR format. Default: None.

    pixel_offsets : list or tuple, optional
        A list of `[x_offset, y_offset, x_size, y_size]` to use as the extent of the
        raster. Uses pixel offsets and the OGR format. Default: None.

    cast : str or dtype, optional
        A type to cast the array to. If None, the array is not cast. It is only cast
        if the array is not already the dtype. Default: None.

    Returns
    -------
    np.ndarray
        A numpy array in the 3D channel-last format.
    
    Raises
    ------
    ValueError
        If the raster is not a valid raster.
    ValueError
        If the bands are not valid.
    ValueError
        If the masked parameter is not valid.
    ValueError
        If both bbox and pixel_offsets are provided.
    
    Examples
    --------
    `Example 1: Convert a raster to a numpy array.`
    ```python
    >>> import buteo as beo
    >>> 
    >>> raster = "/path/to/raster/raster.tif"
    >>> 
    >>> # Convert a raster to a numpy array
    >>> array = beo.raster_to_array(raster)
    >>> 
    >>> array.shape, array.dtype
    >>> (100, 100, 3), dtype('uint8')
    ```
    `Example 2: Convert a raster to a numpy array with a specific band.`
    ```python
    >>> import buteo as beo
    >>> 
    >>> raster = "/path/to/raster/raster.tif"
    >>> 
    >>> # Convert a raster to a numpy array
    >>> array = beo.raster_to_array(raster, bands=[2])
    >>> 
    >>> array.shape, array.dtype
    >>> (100, 100, 1), dtype('uint8')
    ```
    `Example 3: Convert a list of rasters to a numpy array with a specific`
    ```python
    >>> # band and a specific type and filled.
    >>> from glob import glob
    >>> import buteo as beo
    >>> 
    >>> FOLDER = "/path/to/folder"
    >>> rasters = glob(FOLDER + "/*.tif")
    >>>
    >>> len(rasters)
    >>> 10
    >>> 
    >>> # Convert rasters to a numpy array
    >>> array = beo.raster_to_array(
    ...     rasters,
    ...     bands=[2],
    ...     cast="float32",
    ...     filled=True,
    ...     fill_value=0.0,
    ... )
    >>> 
    >>> # This raises an error if the 10 rasters are not aligned.
    >>> array.shape, array.dtype
    >>> (100, 100, 10), dtype('float32')
    ```
    """
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(bands, [int, [int], str], "bands")
    utils_base.type_check(filled, [bool], "filled")
    utils_base.type_check(fill_value, [int, float, None], "fill_value")
    utils_base.type_check(masked, [bool, str], "masked")
    utils_base.type_check(bbox, [list, None], "bbox")
    utils_base.type_check(pixel_offsets, [list, tuple, None], "pixel_offsets")

    if masked not in ["auto", True, False]:
        raise ValueError(f"masked must be 'auto', True, or False. {masked} was provided.")

    if bbox is not None and pixel_offsets is not None:
        raise ValueError("Cannot use both bbox and pixel_offsets.")

    internal_rasters = utils_base._get_variable_as_list(raster)

    if not utils_gdal._check_is_raster_list(internal_rasters):
        raise ValueError(f"An input raster is invalid. {internal_rasters}")

    internal_rasters = utils_gdal._get_path_from_dataset_list(internal_rasters, dataset_type="raster")

    if len(internal_rasters) > 1 and not check_are_rasters_are_aligned(internal_rasters, same_extent=True, same_dtype=False):
        raise ValueError(
            "Cannot merge rasters that are not aligned, have dissimilar extent or dtype, when stack=True."
        )

    # Read metadata
    metadata = raster_to_metadata(internal_rasters[0])
    dtype = metadata["dtype"]
    shape = metadata["shape"]

    # Determine output shape
    x_offset, y_offset, x_size, y_size = 0, 0, shape[1], shape[0]

    if pixel_offsets is not None:
        x_offset, y_offset, x_size, y_size = pixel_offsets

        if x_offset < 0 or y_offset < 0:
            raise ValueError("Pixel offsets cannot be negative.")

        if x_offset + x_size > shape[1] or y_offset + y_size > shape[0]:
            raise ValueError("Pixel offsets are outside of raster.")

    elif bbox is not None:
        if not utils_bbox._check_bboxes_intersect(metadata["bbox"], bbox):
            raise ValueError("Extent is outside of raster.")

        x_offset, y_offset, x_size, y_size = utils_bbox._get_pixel_offsets(metadata["transform"], bbox)

    if (isinstance(bands, str) and bands.lower() == "all") or bands == -1:
        output_shape = (y_size, x_size, len(internal_rasters) * shape[2])
    else:
        channels = 0
        for in_raster in internal_rasters:
            internal_bands = utils_gdal._convert_to_band_list(bands, raster_to_metadata(in_raster)["band_count"])
            channels += len(internal_bands)

        output_shape = (y_size, x_size, channels)

    # Determine nodata and value
    if masked == "auto":
        has_nodata = raster_has_nodata_list(internal_rasters)
        if has_nodata:
            masked = True
        else:
            masked = False

    output_nodata_value = None
    if masked or filled:
        output_nodata_value = _get_first_nodata_value(internal_rasters[0])

        if output_nodata_value is None:
            output_nodata_value = np.nan

        if filled and fill_value is None:
            fill_value = output_nodata_value

    # Create output array
    if masked:
        output_arr = np.ma.empty(output_shape, dtype=dtype)
        output_arr.mask = True

        if filled:
            output_arr.fill_value = fill_value
        else:
            output_arr.fill_value = output_nodata_value
    else:
        output_arr = np.empty(output_shape, dtype=dtype)


    band_idx = 0
    for in_raster in internal_rasters:

        ref = raster_open(in_raster)

        metadata = raster_to_metadata(ref)
        band_count = metadata["band_count"]

        if band_count == 0:
            raise ValueError("The input raster does not have any valid bands.")

        if bands == "all":
            bands = -1

        internal_bands = utils_gdal._convert_to_band_list(bands, metadata["band_count"])

        for band in internal_bands:
            band_ref = ref.GetRasterBand(band)
            band_nodata_value = band_ref.GetNoDataValue()

            if pixel_offsets is not None or bbox is not None:
                arr = band_ref.ReadAsArray(x_offset, y_offset, x_size, y_size)
            else:
                arr = band_ref.ReadAsArray()

            if arr.shape[0] == 0 or arr.shape[1] == 0:
                raise RuntimeWarning("The output data has no rows or columns.")

            if masked or filled:
                if band_nodata_value is not None:
                    masked_arr = np.ma.array(arr, mask=arr == band_nodata_value, copy=False)
                    masked_arr.fill_value = output_nodata_value

                    if filled:
                        arr = np.ma.getdata(masked_arr.filled(fill_value))
                    else:
                        arr = masked_arr

            output_arr[:, :, band_idx] = arr

            band_idx += 1

        ref = None

    if filled and np.ma.isMaskedArray(output_arr):
        output_arr = np.ma.getdata(output_arr.filled(fill_value))

    if cast is not None:
        output_arr = output_arr.astype(cast, copy=False)

    return output_arr


class raster_to_array_chunks:
    """
    A class for reading raster data in chunks. The array will be split into x and y
    amount of chunks in the x and y directions. The output will be the read array
    and the offsets of the chunk in the raster. The offset can be used to reconstitute
    the array into the original raster or a new raster representing the chunk,
    using the :func:`array_to_raster` function.

    Parameters
    ----------
    raster : Union[gdal.Dataset, str, List[Union[str, gdal.Dataset]]]
        The raster to read.

    chunks : int
        The number of chunks to read. The area is chunked in way that ensures
        that the chunks are as square as possible. Default: 1.

    overlap : int, optional
        The number of pixels to overlap. Default: 0.

    overlap_y : int, optional
        The number of pixels to overlap in the y direction. Default: 0.

    bands : list or str or int, optional
        The bands to read. Can be "all", an int, or a list of integers, or a single
        integer. Please note that bands are 1-indexed. Default: "all".

    masked : bool or str, optional
        Whether to return a masked array. Default: "auto".

    filled : bool, optional
        Whether to fill masked values. Default: False.

    fill_value : int or float, optional
        The value to fill masked values with. Default: None.

    cast : type or str, optional
        The data type to cast the output to. Default: None.

    Returns
    -------
    generator
        A generator that yields the raster data in chunks and the offsets of the chunk
        in the raster in a tuple.

    Examples
    --------
    ```python
    >>> # Read a raster into array via chunks.
    >>> import buteo as beo
    >>> 
    >>> raster = "/path/to/raster/raster.tif"
    >>> 
    >>> shape = beo.raster_to_metadata(raster)["shape"]
    >>> shape
    >>> (100, 100)
    >>> 
    >>> for chunk, offsets in beo.raster_to_array_chunks(raster, chunks=4):
    >>>     print(chunk.shape, offsets)
    >>>     (25, 25), [0, 0, 25, 25]
    ```
    """

    def __init__(
        self,
        raster: Union[gdal.Dataset, str, List[Union[str, gdal.Dataset]]],
        chunks: int = 1,
        *,
        overlap: int = 0,
        bands: Union[List[int], str, int] = 'all',
        masked: Union[bool, str] = "auto",
        filled: bool = False,
        fill_value: Optional[Union[int, float]] = None,
        cast: Optional[Union[np.dtype, str]] = None,
    ):
        self.raster = raster
        self.chunks = chunks
        self.overlap = overlap
        self.bands = bands
        self.masked = masked
        self.filled = filled
        self.fill_value = fill_value
        self.cast = cast
        self.current_chunk = 0

        self.shape = raster_to_metadata(self.raster)["shape"]

        assert self.chunks > 0, "The number of chunks must be greater than 0."
        assert self.overlap >= 0, "The overlap must be greater than or equal to 0."
        assert self.chunks <= self.shape[1], "The number of chunks must be less than or equal to the number of columns in the raster."
        assert self.chunks <= self.shape[0], "The number of chunks must be less than or equal to the number of rows in the raster."

        self.offsets = _get_chunk_offsets(
            self.shape,
            self.chunks,
            self.overlap,
        )

        self.total_chunks = len(self.offsets)

    def __iter__(self):
        self.current_chunk = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, List[int]]:
        if self.current_chunk >= self.total_chunks:
            raise StopIteration

        offset = self.offsets[self.current_chunk]
        self.current_chunk += 1

        return (
            raster_to_array(
                self.raster,
                bands=self.bands,
                masked=self.masked,
                filled=self.filled,
                fill_value=self.fill_value,
                pixel_offsets=offset,
                cast=self.cast,
            ),
            offset,
        )

    def __len__(self):
        return self.total_chunks


def array_to_raster(
    array: np.ndarray,
    *,
    reference: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    set_nodata: Union[bool, float, int, str] = "arr",
    allow_mismatches: bool = False,
    pixel_offsets: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
    bbox: Optional[List[float]] = None,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> str:
    """
    Turns a NumPy array into a GDAL dataset or exported as a raster using a reference raster.

    Parameters
    ----------
    array : np.ndarray
        The numpy array to convert.

    reference : str or gdal.Dataset
        The reference raster to use for the output.

    out_path : path, optional
        The destination to save to. Default: None.

    set_nodata : bool or float or int, optional
        Can be set to:
            - "arr": The nodata value will be the same as the NumPy array.
            - "ref": The nodata value will be the same as the reference raster.
            - value: The nodata value will be the value provided. Default: "arr".

    allow_mismatches : bool, optional
        If True, the array can have a different shape than the reference raster.
        Default: False.

    pixel_offsets : list or tuple, optional
        If provided, the array will be written to the reference raster at the
        specified pixel offsets. The list should be in the format [x_offset, y_offset, x_size, y_size].
        Default: None.

    bbox : list, optional
        If provided, the array will be written to the reference raster at the specified
        bounding box. The list should be in the format [min_x, min_y, max_x, max_y]. Default: None.

    overwrite : bool, optional
        If the file exists, should it be overwritten? Default: True.

    creation_options : list, optional
        List of GDAL creation options. Default: ["TILED=YES", "NUM_THREADS=ALL_CPUS",
        "BIGTIFF=YES", "COMPRESS=LZW"].

    Returns
    -------
    str
        The file path to the newly created raster(s).

    Examples
    --------
    ```python
    >>> # Create a raster from a numpy array.
    >>> import buteo as beo
    >>> 
    >>> raster = "/path/to/raster/raster.tif"
    >>> 
    >>> array = beo.raster_to_array(raster)
    >>> array = array ** 2
    >>> 
    >>> out_path = beo.array_to_raster(
    ...     array,
    ...     reference=raster,
    ...     out_path="/path/to/new/new_raster.tif"
    ... )
    >>> 
    >>> out_path
    >>> "/path/to/new/new_raster.tif"
    ```
    """
    utils_base.type_check(array, [np.ndarray, np.ma.MaskedArray], "array")
    utils_base.type_check(reference, [str, gdal.Dataset], "reference")
    utils_base.type_check(out_path, [str, None], "out_path")
    utils_base.type_check(overwrite, [bool], "overwrite")
    utils_base.type_check(pixel_offsets, [[int, float], tuple, None], "pixel_offsets")
    utils_base.type_check(allow_mismatches, [bool], "allow_mismatches")
    utils_base.type_check(set_nodata, [int, float, str, None], "set_nodata")
    utils_base.type_check(creation_options, [[str], None], "creation_options")

    # Verify the numpy array
    if (
        array.size == 0
        or array.ndim < 2
        or array.ndim > 3
    ):
        raise ValueError(f"Input array is invalid {array}")

    if set_nodata not in ["arr", "ref"]:
        utils_base.type_check(set_nodata, [int, float], "set_nodata")

    if pixel_offsets is not None:
        if len(pixel_offsets) != 4:
            raise ValueError("pixel_offsets must be a list of 4 values.")

    if pixel_offsets is not None and bbox is not None:
        raise ValueError("pixel_offsets and bbox cannot be used together.")

    # Parse the driver
    driver_name = "GTiff" if out_path is None else utils_gdal._get_raster_driver_from_path(out_path)
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
        output_name = utils_path._get_augmented_path("array_to_raster.tif", add_uuid=True, folder="/vsimem/")
    else:
        output_name = out_path

    utils_path._delete_if_required(output_name, overwrite)

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

    destination_dtype = utils_gdal_translate._translate_str_to_gdal_dtype(array.dtype)

    # Weird double issue with GDAL and numpy. Cast to float or int
    if input_nodata is not None:
        input_nodata = float(input_nodata)
        if (input_nodata).is_integer() is True:
            input_nodata = int(input_nodata)

    if (metadata["width"] != array.shape[1] or metadata["height"] != array.shape[0]) and pixel_offsets is None and bbox is None:
        if not allow_mismatches:
            raise ValueError(f"Input array and raster are not of equal size. Array: {array.shape[:2]} Raster: {metadata['width'], metadata['height']}")

        warnings.warn(f"Input array and raster are not of equal size. Array: {array.shape[:2]} Raster: {metadata['shape'][:2]}", UserWarning)

    if bbox is not None:
        pixel_offsets = utils_bbox._get_pixel_offsets(metadata["transform"], bbox)

    if pixel_offsets is not None:
        x_offset, y_offset, x_size, y_size = pixel_offsets

        if array.ndim == 3:
            array = array[:y_size, :x_size:, :] # numpy is col, row order
        else:
            array = array[:y_size, x_size]

        metadata["transform"] = (
            metadata["transform"][0] + (x_offset * metadata["pixel_width"]),
            metadata["transform"][1],
            metadata["transform"][2],
            metadata["transform"][3] - (y_offset * metadata["pixel_height"]),
            metadata["transform"][4],
            metadata["transform"][5],
        )

    destination = driver.Create(
        output_name,
        array.shape[1],
        array.shape[0],
        bands,
        destination_dtype,
        utils_gdal._get_default_creation_options(creation_options),
    )

    destination.SetProjection(metadata["projection_wkt"])
    destination.SetGeoTransform(metadata["transform"])

    for band_idx in range(bands):
        band = destination.GetRasterBand(band_idx + 1)
        band.SetColorInterpretation(gdal.GCI_Undefined)

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
    raster: Union[str, gdal.Dataset],
    dtype_str: str,
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> str:
    """ **INTERNAL**. """
    assert isinstance(raster, (str, gdal.Dataset)), "raster must be a string or a GDAL.Dataset."
    assert isinstance(dtype_str, str), "dtype_str must be a string."
    assert len(dtype_str) > 0, "dtype_str must be a non-empty string."
    assert out_path is None or isinstance(out_path, str), "out_path must be a string."

    if not utils_gdal._check_is_raster(raster):
        raise ValueError(f"Unable to open input raster: {raster}")

    ref = raster_open(raster)
    metadata = raster_to_metadata(ref)

    path = ""
    if out_path is None:
        path = utils_path._get_augmented_path("set_datatype.tif", add_uuid=True, folder="/vsimem/")

    elif utils_path._check_dir_exists(out_path):
        path = os.path.join(out_path, os.path.basename(utils_gdal._get_path_from_dataset(ref)))

    elif utils_path._check_dir_exists(utils_path._get_dir_from_path(out_path)):
        path = out_path

    elif utils_path._check_is_valid_mem_filepath(out_path):
        path = out_path

    else:
        raise ValueError(f"Unable to find output folder: {out_path}")

    driver_name = utils_gdal._get_raster_driver_from_path(path)
    driver = gdal.GetDriverByName(driver_name)

    if driver is None:
        raise ValueError(f"Unable to get driver for raster: {raster}")

    utils_path._delete_if_required(path, overwrite)

    if isinstance(dtype_str, str):
        dtype_str = dtype_str.lower()

    copy = driver.Create(
        path,
        metadata["width"],
        metadata["height"],
        metadata["band_count"],
        utils_gdal_translate._translate_str_to_gdal_dtype(dtype_str),
        utils_gdal._get_default_creation_options(creation_options),
    )

    if copy is None:
        raise ValueError(f"Unable to create output raster: {path}")

    copy.SetProjection(metadata["projection_wkt"])
    copy.SetGeoTransform(metadata["transform"])

    for band_idx in range(metadata["band_count"]):
        input_band = ref.GetRasterBand(band_idx + 1)
        output_band = copy.GetRasterBand(band_idx + 1)

        # Read the input band data and write it to the output band
        data = input_band.ReadRaster(0, 0, input_band.XSize, input_band.YSize)
        output_band.WriteRaster(0, 0, input_band.XSize, input_band.YSize, data)

        # Set the NoData value for the output band if it exists in the input band
        if input_band.GetNoDataValue() is not None:
            input_nodata = input_band.GetNoDataValue()
            if utils_gdal_translate._check_value_is_within_dtype_range(input_nodata, dtype_str):
                output_band.SetNoDataValue(input_nodata)
            else:
                warnings.warn("Input NoData value is outside the range of the output datatype. NoData value will not be set.", UserWarning)
                output_band.SetNoDataValue(None)

        # Set the color interpretation for the output band
        output_band.SetColorInterpretation(input_band.GetColorInterpretation())

    copy.FlushCache()

    ref = None
    copy = None

    return path


def raster_set_datatype(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    dtype: str,
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    overwrite: bool = True,
    allow_lists: bool = True,
    creation_options: Optional[List[str]] = None,
) -> Union[str, List[str]]:
    """
    Converts the datatype of a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset or list
        The input raster(s) for which the datatype will be changed.

    dtype : str
        The target datatype for the output raster(s).

    out_path : path or list, optional
        The output location for the processed raster(s). Default: None.

    overwrite : bool, optional
        Determines whether to overwrite existing files with the same name. Default: True.

    allow_lists : bool, optional
        Allows processing multiple rasters as a list. If set to False, only single rasters are accepted.
        Default: True.

    creation_options : list, optional
        A list of GDAL creation options for the output raster(s). Default is
        ["TILED=YES", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES", "COMPRESS=LZW"].

    Returns
    -------
    str or list
        The file path(s) of the newly created raster(s) with the specified datatype.
    """
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(dtype, [str], "dtype")
    utils_base.type_check(out_path, [list, str, None], "out_path")
    utils_base.type_check(overwrite, [bool], "overwrite")
    utils_base.type_check(allow_lists, [bool], "allow_lists")
    utils_base.type_check(creation_options, [list, None], "creation_options")

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

    raster_list = utils_base._get_variable_as_list(raster)
    path_list = utils_gdal._parse_output_data(
        raster_list,
        output_data=out_path,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_raster in enumerate(raster_list):
        path = _raster_set_datatype(
            in_raster,
            dtype,
            out_path=path_list[index],
            overwrite=overwrite,
            creation_options=utils_gdal._get_default_creation_options(creation_options),
        )

        output.append(path)

    if isinstance(raster, list):
        return output

    return output[0]


def raster_stack_list(
    rasters: List[Union[str, gdal.Dataset]],
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    dtype: Optional[str] = None,
    creation_options: Optional[List[str]] = None,
) -> Union[str, List[str]]:
    """
    Stacks a list of aligned rasters.

    Parameters
    ----------
    rasters : list
        List of rasters to stack.

    out_path : str or None, optional
        The destination to save the output raster. Default: None.

    overwrite : bool, optional
        If the file exists, should it be overwritten? Default: True.

    dtype : str, optional
        The data type of the output raster. Default: None.

    creation_options : list, optional
        A list of GDAL creation options for the output raster. Default is
        ["TILED=YES", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES", "COMPRESS=LZW"].

    Returns
    -------
    str or list
        The file path(s) to the newly created raster(s).
    """
    utils_base.type_check(rasters, [[str, gdal.Dataset]], "rasters")
    utils_base.type_check(out_path, [str, None], "out_path")
    utils_base.type_check(overwrite, [bool], "overwrite")
    utils_base.type_check(dtype, [str, None], "dtype")
    utils_base.type_check(creation_options, [[str], None], "creation_options")

    assert utils_gdal._check_is_raster_list(rasters), "Input rasters must be a list of rasters."

    if not check_are_rasters_are_aligned(rasters, same_extent=True):
        raise ValueError("Rasters are not aligned. Try running align_rasters.")

    # Ensures that all the input rasters are valid.
    raster_list = utils_gdal._get_path_from_dataset_list(rasters)

    if out_path is not None and utils_base.path_to_ext(out_path) == ".vrt":
        raise ValueError("Please use stack_rasters_vrt to create vrt files.")

    # Parse the driver
    driver_name = "GTiff" if out_path is None else utils_gdal._get_raster_driver_from_path(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = utils_path._get_output_path("stack_rasters.tif", add_uuid=True)
    else:
        output_name = out_path

    utils_path._delete_if_required(output_name, overwrite)

    raster_dtype = raster_to_metadata(raster_list[0])["datatype_gdal_raw"]

    datatype = raster_dtype
    if dtype is not None:
        datatype = utils_gdal_translate._translate_str_to_gdal_dtype(dtype)

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
                    nodata_missmatch = True
                    warnings.warn("NoDataValues of input rasters do not match. Removing nodata.", UserWarning)

        nodata_values.append(metadata["nodata_value"])

    if nodata_missmatch:
        nodata_value = None

    destination = driver.Create(
        output_name,
        metadatas[0]["width"],
        metadatas[0]["height"],
        total_bands,
        datatype,
        utils_gdal._get_default_creation_options(creation_options),
    )

    destination.SetProjection(metadatas[0]["projection_wkt"])
    destination.SetGeoTransform(metadatas[0]["transform"])

    bands_added = 0
    for idx, raster in enumerate(raster_list):
        ref = raster_open(raster)

        for band_idx in range(metadatas[idx]["band_count"]):
            target_band = destination.GetRasterBand(bands_added + 1)
            source_band = ref.GetRasterBand(band_idx + 1)

            if target_band is None or source_band is None:
                raise ValueError("Unable to get bands from raster.")

            data = source_band.ReadRaster(0, 0, source_band.XSize, source_band.YSize)
            target_band.WriteRaster(0, 0, source_band.XSize, source_band.YSize, data)

            if nodata_value is not None:
                try:
                    target_band.SetNoDataValue(nodata_value)
                except ValueError:
                    target_band.SetNoDataValue(float(nodata_value))

            target_band.SetColorInterpretation(source_band.GetColorInterpretation())

            bands_added += 1

    destination.FlushCache()
    destination = None

    return output_name


def raster_stack_vrt_list(
    rasters: List[Union[str, gdal.Dataset]],
    out_path: str,
    separate: bool = True,
    *,
    resample_alg: str = "nearest",
    nodata_src: Optional[float] = None,
    nodata_VRT: Optional[float] = None,
    nodata_hide: Optional[bool] = None,
    options: Optional[list] = None,
    overwrite: bool = True,
    reference: Optional[str] = None,
    creation_options: Optional[List[str]] = None,
) -> str:
    """
    Stacks a list of rasters into a virtual raster (.vrt).

    Parameters
    ----------
    rasters : list
        List of rasters to stack.

    out_path : str
        The destination to save the output raster.

    separate : bool, optional
        If the raster bands should be separated. Default: True.

    resample_alg : str, optional
        The resampling algorithm to use. Default: 'nearest'.

    nodata_src : float, optional
        The NoData value to use for the source rasters. Default: None.

    nodata_VRT : float, optional
        The NoData value to use for the VRT raster. Default: None.

    nodata_hide : bool, optional
        If the NoData value should be hidden. Default: None.

    options : list, optional
        List of VRT options for GDAL. Default: None.

    overwrite : bool, optional
        If the file exists, should it be overwritten? Default: True.

    reference : str, optional
        The reference raster to use. Default: None.

    creation_options : list, optional
        A list of GDAL creation options for the output raster. Default is
        ["TILED=YES", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES", "COMPRESS=LZW"].

    Returns
    -------
    str
        The file path to the newly created VRT raster.
    """
    utils_base.type_check(rasters, [[str, gdal.Dataset]], "rasters")
    utils_base.type_check(out_path, [str], "out_path")
    utils_base.type_check(separate, [bool], "separate")
    utils_base.type_check(resample_alg, [str], "resample_alg")
    utils_base.type_check(options, [tuple, None], "options")
    utils_base.type_check(overwrite, [bool], "overwrite")
    utils_base.type_check(creation_options, [[str], None], "creation_options")

    if not separate:
        master_bands = 0

        for idx, raster_input in enumerate(rasters):
            if idx == 0:
                master_bands = raster_to_metadata(raster_input)["band_count"]
            else:
                if master_bands != raster_to_metadata(raster_input)["band_count"]:
                    raise ValueError("All rasters must have the same number of bands.")

    resample_algorithm = utils_gdal_translate._translate_resample_method(resample_alg)

    if reference is not None:
        meta = raster_to_metadata(reference)
        options = gdal.BuildVRTOptions(
            resampleAlg=resample_algorithm,
            separate=separate,
            outputBounds=utils_bbox._get_gdal_bbox_from_ogr_bbox(meta["bbox"]),
            xRes=meta["pixel_width"],
            yRes=meta["pixel_height"],
            targetAlignedPixels=True,
            srcNodata=nodata_src,
            VRTNodata=nodata_VRT,
            hideNodata=nodata_hide,
        )
    else:
        options = gdal.BuildVRTOptions(
            resampleAlg=resample_algorithm,
            separate=separate,
            srcNodata=nodata_src,
            VRTNodata=nodata_VRT,
            hideNodata=nodata_hide,
        )

    if separate:
        tmp_vrt_list = []

        for idx, raster in enumerate(rasters):
            bands_in_raster = raster_to_metadata(raster)["band_count"]

            for band in range(bands_in_raster):
                tmp_vrt_path = f"/vsimem/{uuid4().int}_{idx}_{band+1}.vrt"

                tmp_vrt_code = gdal.BuildVRT(
                    tmp_vrt_path,
                    raster,
                    options=gdal.BuildVRTOptions(
                        resampleAlg=resample_algorithm,
                        separate=True,
                        srcNodata=nodata_src,
                        VRTNodata=nodata_VRT,
                        hideNodata=nodata_hide,
                        bandList=[band + 1],
                    )
                )

                tmp_vrt_list.append(tmp_vrt_path)

                if tmp_vrt_code is None:
                    raise ValueError(f"Error while creating VRT from rasters: {rasters}")

                tmp_vrt_code = None

        vrt = gdal.BuildVRT(out_path, tmp_vrt_list, options=options)

        for tmp_vrt_path in tmp_vrt_list:
            gdal.Unlink(tmp_vrt_path)

    else:
        vrt = gdal.BuildVRT(out_path, rasters, options=options)

    vrt.FlushCache()

    if vrt is None:
        raise ValueError(f"Error while creating VRT from rasters: {rasters}")

    vrt = None

    return out_path


def check_do_rasters_intersect(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
) -> bool:
    """
    Checks if two rasters intersect using their latlong boundaries.

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        The first raster.

    raster2 : str or gdal.Dataset
        The second raster.

    Returns
    -------
    bool
        True if the rasters intersect, False otherwise.
    """
    utils_base.type_check(raster1, [str, gdal.Dataset], "raster1")
    utils_base.type_check(raster2, [str, gdal.Dataset], "raster2")

    geom_1 = raster_to_metadata(raster1)["geom_latlng"]
    geom_2 = raster_to_metadata(raster2)["geom_latlng"]

    return geom_1.Intersects(geom_2)


def get_raster_intersection(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
    return_as_vector: bool = False,
) -> Union[ogr.Geometry, ogr.DataSource]:
    """
    Gets the latlng intersection of two rasters.

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        The first raster.

    raster2 : str or gdal.Dataset
        The second raster.

    return_as_vector : bool, optional
        If True, the intersection will be returned as a vector. Default: False.

    Returns
    -------
    tuple or ogr.Geometry
        If return_as_vector is False, returns a tuple `(xmin, ymin, xmax, ymax)` representing
        the intersection of the two rasters. If return_as_vector is True, returns an ogr.Geometry
        object representing the intersection.
    """
    utils_base.type_check(raster1, [str, gdal.Dataset], "raster1")
    utils_base.type_check(raster2, [str, gdal.Dataset], "raster2")

    if not check_do_rasters_intersect(raster1, raster2):
        raise ValueError("Rasters do not intersect.")

    geom_1 = raster_to_metadata(raster1)["geom_latlng"]
    geom_2 = raster_to_metadata(raster2)["geom_latlng"]

    intersection = geom_1.Intersection(geom_2)

    if return_as_vector:
        return utils_gdal.convert_geom_to_vector(intersection)

    return intersection


def get_raster_overlap_fraction(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
) -> float:
    """
    Get the fraction of the overlap between two rasters.
    (e.g. 0.9 for mostly overlapping rasters)

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        The first raster (master).

    raster2 : str or gdal.Dataset
        The second raster.

    Returns
    -------
    float
        A value between 0.0 and 1.0 representing the degree of overlap between the two rasters.
    """
    utils_base.type_check(raster1, [str, gdal.Dataset, [str, gdal.Dataset]], "raster1")
    utils_base.type_check(raster2, [str, gdal.Dataset, [str, gdal.Dataset]], "raster2")

    if not check_do_rasters_intersect(raster1, raster2):
        return 0.0

    geom_1 = raster_to_metadata(raster1)["geom_latlng"]
    geom_2 = raster_to_metadata(raster2)["geom_latlng"]

    try:
        intersection = geom_1.Intersection(geom_2)
    except RuntimeError:
        return 0.0

    overlap = intersection.GetArea() / geom_1.GetArea()

    return overlap


def raster_create_empty(
    out_path: Union[str, None] = None,
    width: int = 100,
    height: int = 100,
    pixel_size: Union[Union[float, int], List[Union[float, int]]] = 10.0,
    bands: int = 1,
    dtype: str = "uint8",
    x_min: Union[float, int] = 0.0,
    y_max: Union[float, int] = 0.0,
    nodata_value: Union[float, int, None] = None,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference] = "EPSG:3857",
    creation_options: Union[List[str], None] = None,
    overwrite: bool = True,
) -> str:
    """
    Create an empty raster.

    Parameters
    ----------
    out_path : str, optional
        The output path. If None, a temporary file will be created.

    width : int, optional
        The width of the raster in pixels. Default: 100.

    height : int, optional
        The height of the raster in pixels. Default: 100.

    pixel_size : int or float or list or tuple, optional
        The pixel size in units of the projection. Default: 10.0.

    bands : int, optional
        The number of bands in the raster. Default: 1.

    dtype : str, optional
        The data type of the raster. Default: "uint8".

    x_min : int or float, optional
        The x coordinate of the top left corner of the raster. Default: 0.0.

    y_max : int or float, optional
        The y coordinate of the top left corner of the raster. Default: 0.0.

    nodata_value : int or float or None, optional
        The nodata value of the raster. Default: None.

    projection : int or str or gdal.Dataset or ogr.DataSource or osr.SpatialReference, optional
        The projection of the raster. Default: "EPSG:3857".

    creation_options : list or None, optional
        A list of creation options. Default: None.

    overwrite : bool, optional
        If True, overwrite the output file if it exists. Default: True.

    Returns
    -------
    str
        The path to the output raster.
    """
    utils_base.type_check(out_path, [str, type(None)], "out_path")
    utils_base.type_check(width, int, "width")
    utils_base.type_check(height, int, "height")
    utils_base.type_check(pixel_size, [int, float, list, tuple], "pixel_size")
    utils_base.type_check(bands, int, "bands")
    utils_base.type_check(dtype, str, "dtype")
    utils_base.type_check(x_min, [int, float], "x_min")
    utils_base.type_check(y_max, [int, float], "y_max")
    utils_base.type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base.type_check(creation_options, [list, type(None)], "creation_options")
    utils_base.type_check(overwrite, bool, "overwrite")

     # Parse the driver
    driver_name = "GTiff" if out_path is None else utils_gdal._get_raster_driver_from_path(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = utils_path._get_output_path("raster_from_array.tif", add_uuid=True)
    else:
        output_name = out_path

    utils_path._delete_if_required(output_name, overwrite)

    destination = driver.Create(
        output_name,
        width,
        height,
        bands,
        utils_gdal_translate._translate_str_to_gdal_dtype(dtype),
        utils_gdal._get_default_creation_options(creation_options),
    )

    parsed_projection = utils_gdal_projection.parse_projection(projection, return_wkt=True)

    destination.SetProjection(parsed_projection)

    pixel_width = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[0]
    pixel_height = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[1]

    transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height] # negative for north-up

    destination.SetGeoTransform(transform)

    if nodata_value is not None:
        for band in range(1, bands + 1):
            destination.GetRasterBand(band).SetNoDataValue(nodata_value)

    destination.FlushCache()
    destination = None

    return output_name


def raster_create_from_array(
    arr: np.ndarray,
    out_path: str = None,
    pixel_size: Union[Union[float, int], List[Union[float, int]]] = 10.0,
    x_min: Union[float, int] = 0.0,
    y_max: Union[float, int] = 0.0,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference] = "EPSG:3857",
    creation_options: Union[List[str], None] = None,
    overwrite: bool = True,
) -> str:
    """ Create a raster from a numpy array.

    Parameters
    ----------
    arr : np.ndarray
        The array to convert to a raster.

    out_path : str, optional
        The output path. If None, a temporary file will be created.

    pixel_size : int or float or list or tuple, optional
        The pixel size of the output raster. Default: 10.0.

    x_min : int or float, optional
        The x coordinate of the top left corner of the output raster. Default: 0.0.

    y_max : int or float, optional
        The y coordinate of the top left corner of the output raster. Default: 0.0.

    projection : int or str or gdal.Dataset or ogr.DataSource or osr.SpatialReference, optional
        The projection of the output raster. Default: "EPSG:3857".

    creation_options : list or None, optional
        The creation options for the output raster. Default: None.

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists. Default: True.

    Returns
    -------
    str
        The path to the output raster.
    """
    utils_base.type_check(arr, [np.ndarray, np.ma.MaskedArray], "arr")
    utils_base.type_check(out_path, [str, None], "out_path")
    utils_base.type_check(pixel_size, [int, float, [int, float], tuple], "pixel_size")
    utils_base.type_check(x_min, [int, float], "x_min")
    utils_base.type_check(y_max, [int, float], "y_max")
    utils_base.type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base.type_check(creation_options, [[str], None], "creation_options")
    utils_base.type_check(overwrite, [bool], "overwrite")

    assert arr.ndim in [2, 3], "Array must be 2 or 3 dimensional (3rd dimension considered bands.)"

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Parse the driver
    driver_name = "GTiff" if out_path is None else utils_gdal._get_raster_driver_from_path(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = utils_path._get_output_path("raster_from_array.tif", add_uuid=True)
    else:
        output_name = out_path

    utils_path._delete_if_required(output_name, overwrite)

    height, width, bands = arr.shape

    destination = driver.Create(
        output_name,
        width,
        height,
        bands,
        utils_gdal_translate._translate_str_to_gdal_dtype(arr.dtype.name),
        utils_gdal._get_default_creation_options(creation_options),
    )

    parsed_projection = utils_gdal_projection.parse_projection(projection, return_wkt=True)

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


def raster_create_grid_with_coordinates(
    raster: Union[str, gdal.Dataset]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid of coordinates from a raster. Format is (x, y, xy).

    Parameters
    ----------
    raster : str or gdal.Dataset
        The raster to create the grid from.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (x, y, xy-coordinates).
    """
    utils_base.type_check(raster, [str, gdal.Dataset], "raster")

    meta = raster_to_metadata(raster)

    step_x = meta["pixel_width"]
    size_x = meta["width"]
    start_x = meta["x_min"]
    stop_x = meta["x_max"]

    step_y = -meta["pixel_height"]
    size_y = meta["height"]
    start_y = meta["y_max"]
    stop_y = meta["y_min"]

    x_adj = step_x / 2
    y_adj = step_y / 2

    x_vals = np.linspace(start_x + x_adj, stop_x - x_adj, size_x, dtype=np.float32)
    y_vals = np.linspace(start_y - y_adj, stop_y + y_adj, size_y, dtype=np.float32)

    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.dstack((xx, yy))

    return grid


# TODO: Create raster with the coordinates of the raster.

# TODO: Implement
def raster_mosaic_list(
    raster_paths: Union[str, List[str]],
    out_path: str = None,
    creation_options: Union[List[str], None] = None,
    overwrite: bool = True,
) -> str:
    """
    NOT YET IMPLEMENTED: Mosaic a list of rasters into a single raster.

    Parameters
    ----------
    raster_paths : str or list
        The list of rasters to mosaic.

    out_path : str, optional
        The output path. If None, a temporary file will be created.

    creation_options : list or None, optional
        The creation options for the output raster.

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists. Default: True.

    Returns
    -------
    None
    """
    utils_base.type_check(raster_paths, [str, [str]], "raster_paths")
    utils_base.type_check(out_path, [str, None], "out_path")
    utils_base.type_check(creation_options, [[str], None], "creation_options")
    utils_base.type_check(overwrite, [bool], "overwrite")

    if isinstance(raster_paths, str):
        raster_paths = [raster_paths]

    # raster_paths = [utils_gdal.path_to_memory(raster_path) for raster_path in raster_paths]

    # Parse the driver
    return


def split_shape_into_offsets(
    shape: Union[List[int], Tuple[int]],
    offsets_x: int = 2,
    offsets_y: int = 2,
    overlap_x: int = 0,
    overlap_y: int = 0,
) -> List[int]:
    """
    Split a shape into offsets. Usually used for splitting an image into offsets to reduce RAM needed.
    
    Parameters
    ----------
    shape : tuple or list
        The shape to split into offsets. (height, width, ...)

    offsets_x : int, optional
        The number of offsets to split the shape into in the x-direction. Default: 2.

    offsets_y : int, optional
        The number of offsets to split the shape into in the y-direction. Default: 2.

    overlap_x : int, optional
        The number of pixels to overlap in the x-direction. Default: 0.

    overlap_y : int, optional
        The number of pixels to overlap in the y-direction. Default: 0.

    Returns
    -------
    list
        The offsets. `[x_offset, y_offset, x_size, y_size]`
    """
    print("WARNING: This is deprecated and will be removed in a future update. Please use get_chunk_offsets instead.")
    height = shape[0]
    width = shape[1]

    x_remainder = width % offsets_x
    y_remainder = height % offsets_y

    x_offsets = [0]
    x_sizes = []
    for _ in range(offsets_x - 1):
        x_offsets.append(x_offsets[-1] + (width // offsets_x) - overlap_x)
    x_offsets[-1] -= x_remainder

    for idx, _ in enumerate(x_offsets):
        if idx == len(x_offsets) - 1:
            x_sizes.append(width - x_offsets[idx])
        elif idx == 0:
            x_sizes.append(x_offsets[1] + overlap_x)
        else:
            x_sizes.append(x_offsets[idx + 1] - x_offsets[idx] + overlap_x)

    y_offsets = [0]
    y_sizes = []
    for _ in range(offsets_y - 1):
        y_offsets.append(y_offsets[-1] + (height // offsets_y) - overlap_y)
    y_offsets[-1] -= y_remainder

    for idx, _ in enumerate(y_offsets):
        if idx == len(y_offsets) - 1:
            y_sizes.append(height - y_offsets[idx])
        elif idx == 0:
            y_sizes.append(y_offsets[1] + overlap_y)
        else:
            y_sizes.append(y_offsets[idx + 1] - y_offsets[idx] + overlap_y)

    offsets = []

    for idx_col, _ in enumerate(y_offsets):
        for idx_row, _ in enumerate(x_offsets):
            offsets.append([
                x_offsets[idx_row],
                y_offsets[idx_col],
                x_sizes[idx_row],
                y_sizes[idx_col],
            ])

    return offsets

def _apply_overlap_to_offsets(
    list_of_offsets: List[Tuple[int, int, int, int]],
    overlap: int,
    shape: Tuple[int, int],
) -> List[List[int]]:
    """
    Apply an overlap to a list of chunk offsets.
    
    The function adjusts the starting position and size of each chunk to apply the specified overlap.
    
    Parameters
    ----------
    list_of_offsets : List[Tuple[int, int, int, int]]
        A list of tuples containing the chunk offsets and dimensions in the format `(x_start, y_start, x_pixels, y_pixels)`.

    overlap : int
        The amount of overlap to apply to each chunk, in pixels.

    shape : Tuple[int, int]
        A tuple containing the height and width of the image: `(Height, Width)`.
        
    Returns
    -------
    List[List[int]]
        A list of lists, each containing the adjusted chunk offsets and dimensions in the format `[x_start, y_start, x_pixels, y_pixels]`.
    """
    height, width = shape
    new_offsets = []
    for start_x, start_y, size_x, size_y in list_of_offsets:
        new_start_x = max(0, start_x - (overlap // 2))
        new_start_y = max(0, start_y - (overlap // 2))

        new_size_x = size_x + overlap
        new_size_y = size_y + overlap

        # If we are over the adjust, bring it back.
        if new_size_x + new_start_x > width:
            new_size_x = new_size_x - ((new_size_x + new_start_x) - width)

        if new_size_y + new_start_y > height:
            new_size_y = new_size_y - ((new_size_y + new_start_y) - height)

        new_offsets.append([
            new_start_x, new_start_y, new_size_x, new_size_y,
        ])

    return new_offsets


def _get_chunk_offsets(
    image_shape: Tuple[int, int],
    num_chunks: int,
    overlap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate chunk offsets for dividing an image into a specified number of chunks with minimal circumference.

    The function finds the optimal configuration of chunks to minimize the circumference and ensure the whole image
    is captured.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        A tuple containing the height and width of the image. (Height, Width)

    num_chunks : int
        The number of chunks to divide the image into.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of tuples, each containing the chunk offsets and dimensions in the format: `(x_start, y_start, x_pixels, y_pixels)`.

    Raises
    ------
    ValueError
        If the number of chunks is too high for the given image size.
    """
    height, width = image_shape[:2]

    # Find the factors of num_chunks that minimize the circumference of the chunks
    min_circumference = float("inf")
    best_factors = (1, num_chunks)

    for i in range(1, num_chunks + 1):
        if num_chunks % i == 0:
            num_h_chunks = i
            num_w_chunks = num_chunks // i

            chunk_height = height // num_h_chunks
            chunk_width = width // num_w_chunks

            # Calculate the circumference of the current chunk configuration
            circumference = 2 * (chunk_height + chunk_width)

            if circumference < min_circumference:
                min_circumference = circumference
                best_factors = (num_h_chunks, num_w_chunks)

    num_h_chunks, num_w_chunks = best_factors

    # Initialize an empty list to store the chunk offsets
    chunk_offsets = []

    # Iterate through the image and create chunk offsets
    for h in range(num_h_chunks):
        for w in range(num_w_chunks):
            h_start = h * (height // num_h_chunks)
            w_start = w * (width // num_w_chunks)

            # If the current chunk is the last one in its row or column, adjust its size
            h_end = height if h == num_h_chunks - 1 else (h + 1) * (height // num_h_chunks)
            w_end = width if w == num_w_chunks - 1 else (w + 1) * (width // num_w_chunks)

            x_pixels = w_end - w_start
            y_pixels = h_end - h_start

            chunk_offsets.append((w_start, h_start, x_pixels, y_pixels))

    if overlap > 0:
        return _apply_overlap_to_offsets(chunk_offsets, overlap, image_shape)

    return chunk_offsets
