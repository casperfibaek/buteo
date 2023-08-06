"""
### Basic IO functions for working with Rasters ###
"""
# Standard library
import sys; sys.path.append("../../")
from typing import List, Optional, Union, Tuple, Type
import warnings

# External
import numpy as np
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
    utils_io,
    utils_translate,
    utils_projection,
)
from buteo.raster import core_raster, core_offsets



def raster_to_array(
    raster: Union[gdal.Dataset, str, List[Union[str, gdal.Dataset]]],
    *,
    bands: Union[List[int], str, int] = 'all',
    filled: bool = False,
    fill_value: Optional[Union[int, float]] = None,
    pixel_offsets: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
    bbox: Optional[List[float]] = None,
    bbox_srs: Optional[Union[str, osr.SpatialReference]] = None,
    cast: Optional[Union[np.dtype, str]] = None,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Converts a raster or a list of rasters into a NumPy array.

    Parameters
    ----------
    raster : gdal.Dataset or str or list
        Raster(s) to convert. The rasters must be aligned.

    bands : list or str or int, optional
        Bands from the raster to convert to a numpy array. Can be "all", an int,
        or a list of integers, or a single integer. Please note that bands are 1-indexed.
        Default: "all".

    filled : bool, optional
        If True, nodata values in the array will be filled with the specified fill_value.
        If False, a masked array will be created with nodata values masked. Default: False.

    fill_value : int or float, optional
        Value to fill the array with if filled is True. If None, the nodata value
        of the raster is used. Default: None.

    bbox : list, optional
        A list of `[xmin, xmax, ymin, ymax]` to use as the extent of the raster.
        Uses coordinates and the OGR format. Default: None.

    bbox_srs : str or osr.SpatialReference, optional
        The spatial reference of the bounding box. If None, the spatial reference
        of the raster is used. Default: None.

    pixel_offsets : list or tuple, optional
        A list of `[x_offset, y_offset, x_size, y_size]` to use as the extent of the
        raster. Uses pixel offsets and the OGR format. Default: None.

    cast : str or dtype, optional
        A type to cast the array to. If None, the array is not cast. It is only cast
        if the array is not already the dtype. Default: None.

    channel_last : bool, optional
        If True, the output array will have shape (height, width, channels).
        If False, the output array will have shape (channels, height, width).
        Default: True.

    Returns
    -------
    np.ndarray
        A numpy array representing the raster data.
    
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
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(bands, [int, [int], str], "bands")
    utils_base._type_check(filled, [bool], "filled")
    utils_base._type_check(fill_value, [int, float, None], "fill_value")
    utils_base._type_check(bbox, [list, None], "bbox")
    utils_base._type_check(bbox_srs, [str, osr.SpatialReference, None], "bbox_srs")
    utils_base._type_check(pixel_offsets, [list, tuple, None], "pixel_offsets")
    utils_base._type_check(cast, [np.dtype, str, type(np.int64), None], "cast")
    utils_base._type_check(channel_last, [bool], "channel_last")

    input_is_list = isinstance(raster, list)

    raster = utils_io._get_input_paths(raster, "raster")

    assert core_raster.check_rasters_are_aligned(raster), "Rasters are not aligned."

    # Read metadata
    metadata = core_raster._get_basic_metadata_raster(raster[0])
    shape = metadata["shape"]
    dtype = metadata["dtype"] if cast is None else cast
    dtype = utils_translate._parse_dtype(dtype)

    # Determine output shape
    x_offset, y_offset, x_size, y_size = 0, 0, shape[1], shape[0]

    if pixel_offsets is not None and bbox is not None:
        raise ValueError("Cannot provide both pixel offsets and a bounding box.")

    elif pixel_offsets is not None:
        x_offset, y_offset, x_size, y_size = pixel_offsets

        if x_offset < 0 or y_offset < 0:
            raise ValueError("Pixel offsets cannot be negative.")

        if x_offset + x_size > shape[1] or y_offset + y_size > shape[0]:
            raise ValueError("Pixel offsets are outside of raster.")

    elif bbox is not None:
        if bbox_srs is not None:
            bbox = utils_projection.reproject_bbox(bbox, bbox_srs, metadata["projection_wkt"])

        if not utils_bbox._check_bboxes_intersect(metadata["bbox"], bbox):
            raise ValueError("Extent is outside of raster.")

        x_offset, y_offset, x_size, y_size = utils_bbox._get_pixel_offsets(metadata["geotransform"], bbox)

        if x_offset < 0 or y_offset < 0:
            raise ValueError("Pixel offsets cannot be negative.")

    if len(raster) > 1 and bands not in ["all", -1]:
        raise ValueError("Cannot specify bands for multiple rasters.")

    bands_to_process = []
    output_shape = shape
    if (isinstance(bands, str) and bands.lower() == "all") or bands == -1:
        total_channels = core_raster._raster_count_bands_list(raster)
        output_shape = [y_size, x_size, total_channels]

        for r in raster:
            bands_in_raster = core_raster._get_basic_metadata_raster(r)["bands"]
            bands_to_process.append(
                utils_gdal._convert_to_band_list(-1, bands_in_raster),
            )
    else:
        channels = 0
        for r in raster:
            bands_in_raster = core_raster._get_basic_metadata_raster(r)["bands"]
            bands_in_raster_list = utils_gdal._convert_to_band_list(bands, bands_in_raster)
            bands_to_process.append(bands_in_raster_list)

            channels += len(bands_in_raster_list)

        output_shape = [y_size, x_size, channels]

    # Create output array
    if not core_raster._check_raster_has_nodata_list(raster) or filled:
        output_array = np.zeros(output_shape, dtype=dtype)
    else:
        output_array = np.ma.zeros(output_shape, dtype=dtype)

    # Read data
    channel = 0
    for idx, r_path in enumerate(raster):

        # We can read all at once
        if bands_to_process == "all" or bands_to_process[idx] == -1:
            r_open = core_raster._raster_open(r_path)
            data = r_open.ReadAsArray(x_offset, y_offset, x_size, y_size)

            if np.ma.isMaskedArray(data) and filled:
                if fill_value is None:
                    fill_value = r_open.GetRasterBand(1).GetNoDataValue()

                    if not utils_translate._check_is_value_within_dtype_range(fill_value, dtype):
                        warnings.warn(
                            f"Fill value {fill_value} is outside of dtype {dtype} range. "
                            "Setting fill value to 0."
                        )
                        fill_value = 0

                data = np.ma.getdata(data.filled(fill_value))

            elif filled:
                np.nan_to_num(data, nan=fill_value, copy=False)

            if cast is not None:
                data = utils_translate._safe_numpy_casting(data, dtype)
        
        # We need to read bands one by one
        else:
            for n_band in bands_to_process[idx]:
                r_open = core_raster._raster_open(r_path)

                band = r_open.GetRasterBand(n_band)
                data = band.ReadAsArray(x_offset, y_offset, x_size, y_size)

                if np.ma.isMaskedArray(data) and filled:
                    if fill_value is None:
                        fill_value = band.GetNoDataValue()

                        if not utils_translate._check_is_value_within_dtype_range(fill_value, dtype):
                            warnings.warn(
                                f"Fill value {fill_value} is outside of dtype {dtype} range. "
                                "Setting fill value to 0."
                            )
                            fill_value = 0

                    data = np.ma.getdata(data.filled(fill_value))

                elif filled:
                    np.nan_to_num(data, nan=fill_value, copy=False)

                if cast is not None:
                    output_array[:, :, channel] = utils_translate._safe_numpy_casting(data, dtype)
                else:
                    output_array[:, :, channel] = data

            channel += 1

    # Reshape array
    if not channel_last:
        output_array = np.transpose(output_array, (2, 0, 1))

    if input_is_list:
        return output_array

    return output_array


def array_to_raster(
    array: np.ndarray,
    *,
    reference: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    set_nodata: Optional[Union[float, int, str, bool]] = "arr",
    allow_mismatches: bool = False,
    pixel_offsets: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
    bbox: Optional[List[float]] = None,
    bbox_srs: Optional[Union[str, osr.SpatialReference]] = None,
    channel_last: bool = True,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
):
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

    set_nodata : str, float, int, or bool. Optional
        Can be set to:
            - "arr": The nodata value will be the same as the NumPy array.
            - "ref": The nodata value will be the same as the reference raster.
            - value: The nodata value will be the value provided. Default: "arr".
            - None/False: The nodata value will not be set.

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

    bbox_srs : str or osr.SpatialReference, optional
        The spatial reference of the bounding box. If None, the spatial reference
        of the raster is used. Default: None.

    channel_last : bool, optional
        If True, the array is in the channel-last format. If False, the array is in the
        channel-first format. Default: True.

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
    utils_base._type_check(array, [np.ndarray], "array")
    utils_base._type_check(reference, [str, gdal.Dataset], "reference")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(set_nodata, [float, int, str, bool, None], "set_nodata")
    utils_base._type_check(allow_mismatches, [bool], "allow_mismatches")
    utils_base._type_check(pixel_offsets, [list, tuple, None], "pixel_offsets")
    utils_base._type_check(bbox, [list, None], "bbox")
    utils_base._type_check(bbox_srs, [str, osr.SpatialReference, None], "bbox_srs")
    utils_base._type_check(channel_last, [bool], "channel_last")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [list, None], "creation_options")

    if isinstance(set_nodata, str) and set_nodata not in ["arr", "ref"]:
        raise ValueError("set_nodata must be either 'arr' or 'ref'.")

    if array.ndim not in [2, 3]:
        raise ValueError("Array must be 2D or 3D.")

    assert utils_gdal._check_is_raster(reference), "Reference is not a raster."

    if out_path is None:
        out_path = utils_path._get_temp_filepath(name="array_to_raster", ext="tif")
    else:
        assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), (
            f"Output path {out_path} is not valid or already exists. "
            "Set overwrite to True to overwrite existing files."
        )

    # Always output tif
    out_path = utils_path._get_augmented_path(out_path, change_ext="tif")

    if pixel_offsets is not None and bbox is not None:
        raise ValueError("Cannot specify both pixel offsets and bounding box.")

    metadata_ref = core_raster._get_basic_metadata_raster(reference)

    if pixel_offsets is None:
        pixel_offsets = [0, 0, metadata_ref["width"], metadata_ref["height"]]

    assert len(pixel_offsets) == 4 and all([isinstance(val, int) for val in pixel_offsets]), (
        "Pixel offsets must be a list of 4 integers."
    )

    if bbox is not None:
        assert len(bbox) == 4 and all([isinstance(val, (float, int)) for val in bbox]), (
            "Bounding box must be a list of 4 floats or integers."
        )
        assert utils_bbox._check_is_valid_bbox(bbox), "Bounding box is not valid."

        if bbox_srs is not None:
            bbox = utils_projection.reproject_bbox(bbox, bbox_srs, metadata_ref["projection_wkt"])

        pixel_offsets = utils_bbox._get_pixel_offsets(metadata_ref["geotransform"], bbox)

    # Handle nodata
    destination_nodata = False
    destination_nodata_value = 0.0
    if set_nodata == "arr":
        if np.ma.isMaskedArray(array):
            destination_nodata = True
            destination_nodata = array.fill_value
        else:
            destination_nodata = False
            destination_nodata = None
    elif set_nodata == "ref":
        destination_nodata = metadata_ref["nodata"]
        destination_nodata_value = metadata_ref["nodata_value"]
    elif set_nodata is None or set_nodata is False:
        destination_nodata = False
        destination_nodata_value = None
    else:
        destination_nodata = True
        destination_nodata_value = set_nodata

        if np.ma.isMaskedArray(array):
            array = np.ma.getdata(array.filled(destination_nodata_value))

    # Swap to channel_last for GDAL.
    if not channel_last:
        array = np.transpose(array, (1, 2, 0))

    x_start, y_start, x_size, y_size = pixel_offsets

    if x_size <= 0 or y_size <= 0:
        raise ValueError("Pixel offsets are invalid. Sizes must be greater than 0.")

    if array.shape[1] != x_size or array.shape[0] != y_size:
        raise ValueError("Array shape does not match pixel offsets.")

    bands = array.shape[2] if array.ndim == 3 else 1

    if not allow_mismatches and y_start + y_size > metadata_ref["height"]:
        raise ValueError("Array is too large for reference raster.")

    if not allow_mismatches and x_start + x_size > metadata_ref["width"]:
        raise ValueError("Array is too large for reference raster.")

    if array.ndim == 3:
        array = array[:y_size, :x_size:, :]
    else:
        array = array[:y_size, :x_size]

    # Create output raster
    driver = utils_gdal._get_default_driver_raster()

    destination_transform = (
        metadata_ref["geotransform"][0] + (x_start * metadata_ref["pixel_width"]),
        metadata_ref["geotransform"][1],
        metadata_ref["geotransform"][2],
        metadata_ref["geotransform"][3] - (y_start * metadata_ref["pixel_height"]),
        metadata_ref["geotransform"][4],
        metadata_ref["geotransform"][5],
    )

    utils_path._delete_if_required(out_path, overwrite)

    destination = driver.Create(
        out_path,
        x_size,
        y_size,
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

        if bands > 1 or array.ndim == 3:
            band.WriteArray(array[:, :, band_idx])
        else:
            band.WriteArray(array)

        if destination_nodata:
            if utils_base._check_variable_is_int(destination_nodata_value):
                band.SetNoDataValue(int(destination_nodata_value))
            elif utils_base._check_variable_is_float(destination_nodata_value):
                band.SetNoDataValue(float(destination_nodata_value))
            else:
                band.SetNoDataValue(np.nan)

    destination.FlushCache()
    destination = None

    return out_path


def save_dataset_to_disk(
    dataset: Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]],
    out_path: Union[str, List[str]],
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> Union[str, List[str]]:
    """
    Writes a dataset to disk. Can be a raster or a vector.

    Parameters
    ----------
    dataset : Union[gdal.Dataset, ogr.DataSource, str]
        The dataset to write to disk.

    out_path : Union[str, List[str]]
        The output path or list of output paths.
    
    prefix : str, optional
        A prefix to add to the output path. Default: "".

    suffix : str, optional
        A suffix to add to the output path. Default: "".

    add_uuid : bool, optional
        If True, a UUID will be added to the output path. Default: False.

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output path. Default: False.

    overwrite : bool, optional
        If True, the output will be overwritten if it already exists. Default: True.

    creation_options : Optional[List[str]], optional
        A list of creation options. Default: None.

    Returns
    -------
    Union[str, List[str]]
        The output path or list of output paths.
    """
    input_is_list = isinstance(dataset, list)

    input_data = utils_io._get_input_paths(dataset, "mixed")
    output_paths = utils_io._get_output_paths(
        input_data,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        overwrite=overwrite,
    )
    creation_options = utils_gdal._get_default_creation_options(creation_options)

    for idx, dataset in enumerate(input_data):
        driver = None

        # Raster
        if utils_gdal._check_is_raster(dataset):
            driver_name = utils_gdal._get_driver_name_from_path(dataset)
            driver = gdal.GetDriverByName(driver_name)
            src_ds = gdal.Open(dataset)
            utils_path._delete_if_required(output_paths[idx], overwrite)
            driver.CreateCopy(output_paths[idx], src_ds, options=creation_options)
            src_ds = None

        # Vector
        elif utils_gdal._check_is_vector(dataset):
            driver_name = utils_gdal._get_driver_name_from_path(dataset)
            driver = ogr.GetDriverByName(driver_name)
            src_ds = ogr.Open(dataset)
            utils_path._delete_if_required(output_paths[idx], overwrite)
            driver.CopyDataSource(src_ds, output_paths[idx])
            src_ds = None

        else:
            raise RuntimeError(f"Invalid dataset type: {dataset}")

        if driver is None:
            raise RuntimeError("Could not get driver for output dataset.")

    if input_is_list:
        return output_paths

    return output_paths[0]


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

    chunk_size : list or tuple, optional
        The raster can be split into chunks of a fixed size, 
        instead of splitting into a fixed number of chunks.

        The list should be in the format [x_size, y_size].
        If this is provided, the chunks parameter is ignored. Default: None.

    overlap : int, optional
        The number of pixels to overlap. Default: 0.

    bands : list or str or int, optional
        The bands to read. Can be "all", an int, or a list of integers, or a single
        integer. Please note that bands are 1-indexed. Default: "all".

    filled : bool, optional
        Whether to fill masked values. Default: False.

    fill_value : int or float, optional
        The value to fill masked values with. Default: None.

    skip : int, optional
        The number of chunks to skip when iterating. Default: 0.

    border_strategy : int, optional
        The border strategy to use when splitting the raster into chunks.
        border_strategy ignored when chunk_size and overlaps are provided.
        Only applied when chunk_size is provided. Can be 1 or 2. Default: 1.
        1. Ignore the border chunks if they do not fit the chunk size.
        2. Oversample the border chunks to fit the chunk size.
        3. Shrink the last chunk to fit the image size. (Creates uneven chunks.)

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
    >>>     (25, 25), (0, 0, 25, 25)
    ```
    """

    def __init__(
        self,
        raster: Union[gdal.Dataset, str, List[Union[str, gdal.Dataset]]],
        chunks: Optional[int] = 1,
        chunk_size: Optional[Union[List[int], Tuple[int, int]]] = None,
        *,
        overlap: int = 0,
        bands: Union[List[int], str, int] = 'all',
        filled: bool = False,
        fill_value: Optional[Union[int, float]] = None,
        skip: int = 0,
        border_strategy: int = 1,
        cast: Optional[Union[np.dtype, str]] = None,
        channel_last: bool = True,
    ):
        self.raster = raster
        self.chunks = chunks
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.bands = bands
        self.filled = filled
        self.fill_value = fill_value
        self.skip = skip
        self.border_strategy = border_strategy
        self.cast = cast
        self.current_chunk = 0
        self.channel_last = channel_last

        self.shape = core_raster._get_basic_metadata_raster(self.raster)["shape"]

        assert self.chunks > 0, "The number of chunks must be greater than 0."
        assert self.overlap >= 0, "The overlap must be greater than or equal to 0."
        assert self.chunks <= self.shape[1], "The number of chunks must be less than or equal to the number of columns in the raster."
        assert self.chunks <= self.shape[0], "The number of chunks must be less than or equal to the number of rows in the raster."
        assert self.border_strategy in [1, 2, 3], "The border strategy must be 1, 2, or 3."

        if self.chunk_size is not None:
            assert isinstance(self.chunk_size, (list, tuple)), "Chunk size must be a list or tuple."
            assert len(self.chunk_size) == 2, "Chunk size must be a list or tuple of length 2."
            assert all([isinstance(val, int) for val in self.chunk_size]), "Chunk size must be a list or tuple of integers."

            self.offsets = core_offsets._get_chunk_offsets_fixed_size(
                self.shape,
                self.chunk_size[0],
                self.chunk_size[1],
                self.border_strategy,
                self.overlap,
                channel_last=channel_last,
            )

        else:
            self.offsets = core_offsets._get_chunk_offsets(
                self.shape,
                self.chunks,
                self.overlap,
                channel_last=channel_last,
            )

        self.total_chunks = len(self.offsets)

    def __iter__(self):
        self.current_chunk = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, List[int]]:
        if self.current_chunk >= self.total_chunks:
            raise StopIteration

        offset = self.offsets[self.current_chunk]
        self.current_chunk += 1 + self.skip

        return (
            raster_to_array(
                self.raster,
                bands=self.bands,
                filled=self.filled,
                fill_value=self.fill_value,
                pixel_offsets=offset,
                cast=self.cast,
                channel_last=self.channel_last,
            ),
            offset,
        )

    def __len__(self):
        return self.total_chunks


def raster_create_empty(
    out_path: Union[str, None] = None,
    *,
    width: int = 100,
    height: int = 100,
    pixel_size: Union[Union[float, int], List[Union[float, int]]] = 10.0,
    bands: int = 1,
    dtype: Union[str, int, np.dtype] = "uint8",
    x_min: Union[float, int] = 0.0,
    y_max: Union[float, int] = 0.0,
    nodata_value: Union[float, int, None] = None,
    fill_value: Union[float, int, None] = 0.0,
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

    dtype : str, int, np.dtype, optional
        The data type of the raster. Default: "uint8".

    x_min : int or float, optional
        The x coordinate of the top left corner of the raster. Default: 0.0.

    y_max : int or float, optional
        The y coordinate of the top left corner of the raster. Default: 0.0.

    nodata_value : int or float or None, optional
        The nodata value of the raster. Default: None.

    fill_value : int or float or None, optional
        The fill value of the raster. Default: 0.0.

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
    utils_base._type_check(out_path, [str, type(None)], "out_path")
    utils_base._type_check(width, [int], "width")
    utils_base._type_check(height, [int], "height")
    utils_base._type_check(pixel_size, [int, float, list, tuple], "pixel_size")
    utils_base._type_check(bands, [int], "bands")
    utils_base._type_check(dtype, [str, int, np.dtype, type(np.int8), None], "dtype")
    utils_base._type_check(x_min, [int, float], "x_min")
    utils_base._type_check(y_max, [int, float], "y_max")
    utils_base._type_check(nodata_value, [int, float, type(None)], "nodata_value")
    utils_base._type_check(fill_value, [int, float, type(None)], "fill_value")
    utils_base._type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base._type_check(creation_options, [list, type(None)], "creation_options")
    utils_base._type_check(overwrite, bool, "overwrite")

    if out_path is not None and not utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite):
        raise ValueError(
            f"Output path {out_path} is not valid or already exists. "
            "Set overwrite to True to overwrite existing files."
        )

    if out_path is None:
        out_path = utils_path._get_temp_filepath(name="raster_create_empty", ext="tif", add_timestamp=True)

    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)

    utils_path._delete_if_required(out_path, overwrite)

    destination = driver.Create(
        out_path,
        width,
        height,
        bands,
        utils_translate._translate_dtype_numpy_to_gdal(utils_translate._parse_dtype(dtype)),
        utils_gdal._get_default_creation_options(creation_options),
    )

    parsed_projection = utils_projection.parse_projection(projection, return_wkt=True)
    destination.SetProjection(parsed_projection)

    pixel_width = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[0]
    pixel_height = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[1]

    transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height] # negative for north-up

    destination.SetGeoTransform(transform)

    if nodata_value is not None:
        for band in range(1, bands + 1):
            band_obj = destination.GetRasterBand(band)

            if nodata_value is not None:
                band_obj.SetNoDataValue(nodata_value)

            if fill_value is not None:
                band_obj.Fill(fill_value)

    destination.FlushCache()
    destination = None

    return out_path


def raster_create_from_array(
    arr: np.ndarray,
    out_path: str = None,
    pixel_size: Union[Union[float, int], List[Union[float, int]]] = 1.0,
    x_min: Union[float, int] = 0.0,
    y_max: Union[float, int] = 0.0,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference] = "EPSG:3857",
    channel_last: bool = True,
    overwrite: bool = True,
    creation_options: Union[List[str], None] = None,
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
    utils_base._type_check(arr, [np.ndarray, np.ma.MaskedArray], "arr")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(pixel_size, [int, float, [int, float], tuple], "pixel_size")
    utils_base._type_check(x_min, [int, float], "x_min")
    utils_base._type_check(y_max, [int, float], "y_max")
    utils_base._type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(overwrite, [bool], "overwrite")

    assert arr.ndim in [2, 3], "Array must be 2 or 3 dimensional (3rd dimension considered bands.)"

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if not channel_last:
        arr = np.transpose(arr, (2, 0, 1))

    if out_path is not None and not utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite):
        raise ValueError(
            f"Output path {out_path} is not valid or already exists. "
            "Set overwrite to True to overwrite existing files."
        )

    if out_path is None:
        out_path = utils_path._get_temp_filepath(name="raster_create_from_array", ext="tif", add_timestamp=True)

    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)

    utils_path._delete_if_required(out_path, overwrite)

    height, width, bands = arr.shape

    destination = driver.Create(
        out_path,
        width,
        height,
        bands,
        utils_translate._translate_dtype_numpy_to_gdal(arr.dtype),
        utils_gdal._get_default_creation_options(creation_options),
    )

    parsed_projection = utils_projection.parse_projection(projection, return_wkt=True)

    destination.SetProjection(parsed_projection)

    pixel_width = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[0]
    pixel_height = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[1]

    transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height] # negative for north-up

    destination.SetGeoTransform(transform)

    nodata = None
    nodata_value = 0.0

    if isinstance(arr, np.ma.MaskedArray):
        nodata = True
        nodata_value = arr.fill_value
        arr = np.ma.getdata(arr.filled(nodata_value))

    for idx in range(0, bands):
        dst_band = destination.GetRasterBand(idx + 1)
        dst_band.WriteArray(arr[:, :, idx])

        if nodata:
            dst_band.SetNoDataValue(nodata_value)

    return out_path


def raster_create_copy(
    raster: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Create a copy of a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset
        The raster to copy.

    out_path : str, optional
        The output path. If None, a temporary file will be created.

    Returns
    -------
    str
        The path to the output raster.
    """
    assert utils_gdal._check_is_raster(raster), "Raster is not valid."

    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            utils_gdal._get_path_from_dataset(raster),
            ext="tif",
        )
    else:
        assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), (
            f"Output path {out_path} is not valid or already exists. "
        )

    utils_path._delete_if_required(out_path, overwrite)

    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)

    src_ds = core_raster._raster_open(raster)
    dst_ds = driver.CreateCopy(out_path, src_ds) # pylint: disable=unused-variable
    dst_ds = None
    src_ds = None

    return out_path
