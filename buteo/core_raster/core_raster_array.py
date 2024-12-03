"""### Basic IO functions for working with Rasters. ###"""

# Standard library
from typing import List, Optional, Union, Tuple, Type

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_base,
    utils_io,
    utils_gdal,
    utils_bbox,
    utils_path,
    utils_translate,
)
from buteo.core_raster.core_raster_read import _open_raster, _read_raster_band
from buteo.core_raster.core_raster_info import get_metadata_raster



def raster_to_array(
    raster: Union[gdal.Dataset, str],
    *,
    bands: Union[List[int], str, int] = 'all',
    filled: bool = False,
    fill_value: Optional[Union[int, float]] = None,
    pixel_offsets: Optional[Tuple[int, int, int, int]] = None,
    bbox: Optional[List[float]] = None,
    cast: Optional[Union[np.dtype, str, Type[np.int32]]] = None,
) -> np.ndarray:
    """Converts a raster into a NumPy array in channel-first format (C x H x W).

    Parameters
    ----------
    raster : Union[gdal.Dataset, str]
        The raster to convert
    bands : Union[List[int], str, int], optional
        Band selection (1-based). Can be "all", int, or list of ints
    filled : bool, optional
        Fill nodata values
    fill_value : Optional[Union[int, float]], optional
        Value to fill nodata with
    pixel_offsets : Optional[Tuple[int, int, int, int]], optional
        (x_offset, y_offset, x_size, y_size) for reading subset
    bbox : Optional[List[float]], optional
        [xmin, xmax, ymin, ymax] to read. Assummed to be in the same crs as the raster.
    cast : Optional[Union[np.dtype, str]], optional
        Output data type

    Returns
    -------
    np.ndarray
        3D array in C x H x W format

    Raises
    ------
    ValueError
        If both bbox and pixel_offsets are provided
        If pixel offsets are invalid
        If bbox is outside raster extent
    TypeError
        If input types are invalid
    """
    utils_base._type_check(raster, [gdal.Dataset, str], "raster")
    utils_base._type_check(bands, [list, str, int], "bands")
    utils_base._type_check(filled, [bool], "filled")
    utils_base._type_check(fill_value, [int, float, type(None)], "fill_value")
    utils_base._type_check(pixel_offsets, [tuple, type(None)], "pixel_offsets")
    utils_base._type_check(bbox, [list, type(None)], "bbox")
    utils_base._type_check(cast, [np.dtype, str, type(None), type(np.int32)], "cast")

    if bbox is not None and pixel_offsets is not None:
        raise ValueError("Cannot specify both bbox and pixel_offsets")

    raster = _open_raster(raster)
    metadata = get_metadata_raster(raster)

    if pixel_offsets is None:
        pixel_offsets = (0, 0, metadata["width"], metadata["height"])

    if bbox is not None:
        if not utils_bbox._check_bboxes_intersect(metadata["bbox"], bbox):
            raise ValueError("bbox outside raster extent")

        pixel_offsets = utils_bbox._get_pixel_offsets(metadata["geotransform"], bbox)

    dtype = cast if cast is not None else metadata["dtype"]
    dtype = utils_translate._parse_dtype(dtype)

    if bands == "all" or bands == -1:
        bands = list(range(1, metadata["bands"] + 1))
    elif isinstance(bands, int):
        bands = [bands]

    # Read bands
    arrays = []
    for band_idx in bands:

        if not isinstance(band_idx, int):
            raise ValueError(f"Band index must be an integer, not {band_idx}")

        arr = _read_raster_band(raster, band_idx, pixel_offsets)
        if cast is not None:
            arr = utils_translate._safe_numpy_casting(arr, dtype)
        arrays.append(arr)

    output = np.stack(arrays)

    # Handle nodata values
    if metadata["nodata"]:
        nodata_value = np.array(metadata["nodata_value"], dtype=dtype)
        output = np.ma.masked_equal(output, nodata_value)

        if filled:
            fill_val = fill_value if fill_value is not None else nodata_value
            output = output.filled(fill_val)

    return output


def array_to_raster(
    array: np.ndarray,
    *,
    reference: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    set_nodata: Optional[Union[float, int, str, bool]] = "arr",
    pixel_offsets: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
    bbox: Optional[List[float]] = None,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> str:
    """Converts a NumPy array into a raster file using a reference raster.
    The default order of the array is assumed to be C x H x W for 3D arrays and H x W for 2D arrays.

    Parameters
    ----------
    array : np.ndarray
        The NumPy array to convert.
    reference : Union[str, gdal.Dataset]
        The reference raster to use for the output.
    out_path : Optional[str], optional
        The destination path to save the raster. If None, a temporary path is used.
    set_nodata : Optional[Union[float, int, str, bool]], optional
        How to set the nodata value:
            - "arr": Use the nodata value from the NumPy array.
            - "ref": Use the nodata value from the reference raster.
            - Value: Use the specified value.
            - None or False: Do not set nodata value.
    pixel_offsets : Optional[Union[List[int], Tuple[int, int, int, int]]], optional
        Pixel offsets in the format [x_offset, y_offset, x_size, y_size].
    bbox : Optional[List[float]], optional
        Bounding box [min_x, min_y, max_x, max_y] defining the area to write. Same crs as raster.
    overwrite : bool, optional
        If True, overwrites the output file if it exists.
    creation_options : Optional[List[str]], optional
        List of GDAL creation options.

    Returns
    -------
    str
        The file path to the created raster.

    Raises
    ------
    ValueError
        If both pixel_offsets and bbox are provided.
        If set_nodata is invalid.
        If the array is not 2D or 3D.
        If the reference is not a raster.
        If output path is invalid or file already exists.
        If pixel_offsets or bbox are invalid.
        If array shape does not match pixel offsets.
        If array is too large for reference raster.
    RuntimeError
        If the output raster could not be created.

    Examples
    --------
    ```python
    >>> import buteo as beo
    >>> raster = "/path/to/raster/raster.tif"
    >>> array = beo.raster_to_array(raster)
    >>> array = array ** 2
    >>> out_path = beo.array_to_raster(
    ...     array,
    ...     reference=raster,
    ...     out_path="/path/to/new/new_raster.tif"
    ... )
    >>> out_path
    '/path/to/new/new_raster.tif'
    ```
    """
    utils_base._type_check(array, [np.ndarray], "array")
    utils_base._type_check(reference, [str, gdal.Dataset], "reference")
    utils_base._type_check(out_path, [str, type(None)], "out_path")
    utils_base._type_check(set_nodata, [float, int, str, bool, type(None)], "set_nodata")
    utils_base._type_check(pixel_offsets, [list, tuple, type(None)], "pixel_offsets")
    utils_base._type_check(bbox, [list, type(None)], "bbox")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [list, type(None)], "creation_options")

    if isinstance(set_nodata, str) and set_nodata not in ["arr", "ref"]:
        raise ValueError("set_nodata must be 'arr', 'ref', a numeric value, or None/False.")

    if array.ndim not in [2, 3]:
        raise ValueError("Array must be 2D or 3D.")

    if not utils_gdal._check_is_raster(reference):
        raise ValueError("Reference is not a raster.")

    if out_path is None:
        out_path = utils_path._get_temp_filepath(name="array_to_raster", ext="tif")
    elif not utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite):
        raise ValueError(f"Output path {out_path} is not valid or already exists.")

    if pixel_offsets is not None and bbox is not None:
        raise ValueError("Cannot specify both pixel offsets and bounding box.")

    out_path = utils_path._get_augmented_path(out_path, change_ext="tif")
    metadata_ref = get_metadata_raster(reference)

    if bbox is not None:
        if not (len(bbox) == 4 and all(isinstance(val, (float, int)) for val in bbox)):
            raise ValueError("Bounding box must be a list of 4 floats or integers.")
        if not utils_bbox._check_is_valid_bbox(bbox):
            raise ValueError("Bounding box is not valid.")
        pixel_offsets = utils_bbox._get_pixel_offsets(metadata_ref["geotransform"], bbox)

    if pixel_offsets is None:
        pixel_offsets = [0, 0, metadata_ref["width"], metadata_ref["height"]]

    if not (len(pixel_offsets) == 4 and all(isinstance(val, int) for val in pixel_offsets)):
        raise ValueError("Pixel offsets must be a list of 4 integers.")

    x_start, y_start, x_size, y_size = pixel_offsets

    if x_size <= 0 or y_size <= 0:
        raise ValueError("Pixel offsets are invalid. Sizes must be greater than 0.")

    if array.ndim == 3:
        channels, height, width = array.shape
        bands = channels
    else:
        height, width = array.shape
        bands = 1

    if width != x_size or height != y_size:
        raise ValueError("Array shape does not match pixel offsets.")

    array = array[..., :y_size, :x_size]

    if set_nodata == "arr":
        destination_nodata = array.fill_value if np.ma.isMaskedArray(array) else None # type: ignore
    elif set_nodata == "ref":
        destination_nodata = metadata_ref["nodata_value"]
    elif set_nodata is None or set_nodata is False:
        destination_nodata = None
    else:
        destination_nodata = set_nodata

    if np.ma.isMaskedArray(array):
        fill_val = destination_nodata if destination_nodata is not None else array.fill_value # type: ignore
        array = array.filled(fill_val) # type: ignore

    # if destination_nodata is not None and not a float or int, convert it to float
    try:
        if destination_nodata is not None and not utils_base._check_variable_is_number_type(destination_nodata):
            destination_nodata = float(destination_nodata)
    except Exception as exc:
        raise ValueError("Nodata value is not a valid numeric") from exc

    driver = utils_gdal._get_default_driver_raster()
    destination_transform = (
        metadata_ref["geotransform"][0] + x_start * metadata_ref["pixel_width"],
        metadata_ref["geotransform"][1],
        metadata_ref["geotransform"][2],
        metadata_ref["geotransform"][3] + y_start * metadata_ref["pixel_height"],
        metadata_ref["geotransform"][4],
        metadata_ref["geotransform"][5],
    )

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    destination = driver.Create(
        out_path,
        x_size, # cols
        y_size, # rows
        bands,
        utils_translate._translate_dtype_numpy_to_gdal(array.dtype),
        utils_gdal._get_default_creation_options(creation_options),
    )

    if destination is None:
        raise RuntimeError(f"Could not create output raster: {out_path}")

    destination.SetGeoTransform(destination_transform)
    destination.SetProjection(metadata_ref["projection_wkt"])

    for band_idx in range(bands):
        band = destination.GetRasterBand(band_idx + 1)
        band.SetColorInterpretation(gdal.GCI_Undefined)

        band_array = array[band_idx, :, :] if bands > 1 else array
        band.WriteArray(band_array)

        if destination_nodata is not None:
            band.SetNoDataValue(destination_nodata)

    destination.FlushCache()
    destination = None

    return out_path


def _validata_raster_patch_parameters(
    metadata: dict,
    patch_size: Tuple[int, int],
    num_patches: int,
) -> None:
    """Validates parameters for random patch extraction.

    Parameters
    ----------
    metadata : dict
        Raster metadata dictionary
    patch_size : Tuple[int, int]
        Size of patches to extract (height, width)
    num_patches : int
        Number of patches to extract

    Raises
    ------
    ValueError
        If patch size or num_patches are invalid
    """
    patch_height, patch_width = patch_size

    if patch_height > metadata["height"] or patch_width > metadata["width"]:
        raise ValueError("Patch size is larger than raster dimensions")

    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("Patch dimensions must be positive")

    if num_patches <= 0:
        raise ValueError("Number of patches must be positive")


def raster_to_array_random_patches(
    raster: Union[gdal.Dataset, str],
    patch_size: Tuple[int, int],
    num_patches: int,
    *,
    bands: Union[List[int], str, int] = 'all',
    filled: bool = False,
    fill_value: Optional[Union[int, float]] = None,
    cast: Optional[Union[np.dtype, str, Type[np.int32]]] = None,
) -> np.ndarray:
    """Converts a raster into random patches in format (P x C x H x W).

    Parameters
    ----------
    raster : Union[gdal.Dataset, str]
        The raster to convert
    patch_size : Tuple[int, int]
        The size of patches (height, width)
    num_patches : int
        Number of patches to extract
    bands : Union[List[int], str, int], optional
        Band selection (1-based)
    filled : bool, optional
        Fill nodata values
    fill_value : Optional[Union[int, float]], optional
        Value to fill nodata with
    cast : Optional[Union[np.dtype, str]], optional
        Output data type

    Returns
    -------
    np.ndarray
        4D array in P x C x H x W format

    Raises
    ------
    ValueError
        If parameters are invalid or incompatible
    TypeError
        If input types are invalid
    """
    utils_base._type_check(raster, [gdal.Dataset, str], "raster")
    utils_base._type_check(patch_size, [tuple], "patch_size")
    utils_base._type_check(num_patches, [int], "num_patches")
    utils_base._type_check(bands, [list, str, int], "bands")
    utils_base._type_check(filled, [bool], "filled")
    utils_base._type_check(fill_value, [int, float, type(None)], "fill_value")
    utils_base._type_check(cast, [np.dtype, str, type(None), type(np.int32)], "cast")

    raster = _open_raster(raster)
    metadata = get_metadata_raster(raster)

    _validata_raster_patch_parameters(metadata, patch_size, num_patches)

    if bands == "all" or bands == -1:
        bands = list(range(1, metadata["bands"] + 1))
    elif isinstance(bands, int):
        bands = [bands]

    patch_height, patch_width = patch_size
    offsets = [
        (
            np.random.randint(0, metadata["width"] - patch_width),
            np.random.randint(0, metadata["height"] - patch_height),
            patch_width,
            patch_height
        )
        for _ in range(num_patches)
    ]

    patches = []
    for offset in offsets:
        arrays = []
        for band_idx in bands:
            if not isinstance(band_idx, int):
                raise ValueError(f"Band index must be integer, got {type(band_idx)}")

            arr = _read_raster_band(raster, band_idx, offset)
            if cast is not None:
                arr = utils_translate._safe_numpy_casting(arr, cast)
            arrays.append(arr)
        patches.append(np.stack(arrays))

    output = np.stack(patches)

    if metadata["nodata"]:
        nodata_value = np.array(metadata["nodata_value"], dtype=output.dtype)
        output = np.ma.masked_equal(output, nodata_value)

        if filled:
            if isinstance(fill_value, (float, int)):
                if not utils_translate._check_is_value_within_dtype_range(fill_value, output.dtype):
                    raise ValueError(f"Fill value {fill_value} invalid for {output.dtype}")
            fill_val = fill_value if fill_value is not None else nodata_value
            output = output.filled(fill_val)

    return output
