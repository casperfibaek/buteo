"""
### Basic IO functions for working with Rasters ###
"""
# Standard library
import sys; sys.path.append("../../")
from typing import List, Optional, Union, Tuple
import warnings

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
    utils_translate,
    utils_io,
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
    cast: Optional[Union[np.dtype, str]] = None,
    channel_last: bool = True,
) -> np.ndarray:
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(bands, [int, [int], str], "bands")
    utils_base.type_check(filled, [bool], "filled")
    utils_base.type_check(fill_value, [int, float, None], "fill_value")
    utils_base.type_check(bbox, [list, None], "bbox")
    utils_base.type_check(pixel_offsets, [list, tuple, None], "pixel_offsets")
    utils_base.type_check(cast, [np.dtype, str, None], "cast")
    utils_base.type_check(channel_last, [bool], "channel_last")

    input_is_list = True if isinstance(raster, list) else False

    raster = utils_io._get_input_paths(raster, "raster")

    assert core_raster.check_rasters_are_aligned(raster), "Rasters are not aligned."

    # Read metadata
    metadata = core_raster._get_basic_metadata_raster(raster[0])
    shape = metadata["shape"]
    dtype = metadata["dtype"] if cast is None else cast

    # Need function to check dtypes
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    elif isinstance(dtype, type(np.int64)):
        dtype = np.dtype(np.dtype)

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
                utils_gdal._convert_to_band_list(bands_in_raster, bands_in_raster),
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
    for i, r in enumerate(raster):
        bands_to_process_raster = bands_to_process[i]

        for channel, band in enumerate(bands_to_process_raster):
            r = core_raster._raster_open(r)

            band = r.GetRasterBand(band)
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

            if len(raster) == 1:
                output_array[:, :, channel] = data
            else:
                output_array[:, :, channel] = data

    # Reshape array
    if not channel_last:
        output_array = np.transpose(output_array, (2, 0, 1))

    if input_is_list:
        return output_array

    return output_array


# def raster_to_array(
#     raster: Union[gdal.Dataset, str, List[Union[str, gdal.Dataset]]],
#     *,
#     bands: Union[List[int], str, int] = 'all',
#     masked: Union[bool, str] = "auto",
#     filled: bool = False,
#     fill_value: Optional[Union[int, float]] = None,
#     bbox: Optional[List[float]] = None,
#     pixel_offsets: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
#     cast: Optional[Union[np.dtype, str]] = None,
# ) -> np.ndarray:
#     """
#     Converts a raster or a list of rasters into a NumPy array.

#     Parameters
#     ----------
#     raster : gdal.Dataset or str or list
#         Raster(s) to convert.

#     bands : list or str or int, optional
#         Bands from the raster to convert to a numpy array. Can be "all", an int,
#         or a list of integers, or a single integer. Please note that bands are 1-indexed.
#         Default: "all".

#     masked : bool or str, optional
#         If the array contains nodata values, determines whether the resulting
#         array should be a masked numpy array or a regular numpy array. If "auto",
#         the array will be masked only if the raster has nodata values. Default: "auto".

#     filled : bool, optional
#         If the array contains nodata values, determines whether the resulting
#         array should be a filled numpy array or a masked array. Default: False.

#     fill_value : int or float, optional
#         Value to fill the array with if filled is True. If None, the nodata value
#         of the raster is used. Default: None.

#     bbox : list, optional
#         A list of `[xmin, xmax, ymin, ymax]` to use as the extent of the raster.
#         Uses coordinates and the OGR format. Default: None.

#     pixel_offsets : list or tuple, optional
#         A list of `[x_offset, y_offset, x_size, y_size]` to use as the extent of the
#         raster. Uses pixel offsets and the OGR format. Default: None.

#     cast : str or dtype, optional
#         A type to cast the array to. If None, the array is not cast. It is only cast
#         if the array is not already the dtype. Default: None.

#     Returns
#     -------
#     np.ndarray
#         A numpy array in the 3D channel-last format.
    
#     Raises
#     ------
#     ValueError
#         If the raster is not a valid raster.
#     ValueError
#         If the bands are not valid.
#     ValueError
#         If the masked parameter is not valid.
#     ValueError
#         If both bbox and pixel_offsets are provided.
    
#     Examples
#     --------
#     `Example 1: Convert a raster to a numpy array.`
#     ```python
#     >>> import buteo as beo
#     >>> 
#     >>> raster = "/path/to/raster/raster.tif"
#     >>> 
#     >>> # Convert a raster to a numpy array
#     >>> array = beo.raster_to_array(raster)
#     >>> 
#     >>> array.shape, array.dtype
#     >>> (100, 100, 3), dtype('uint8')
#     ```
#     `Example 2: Convert a raster to a numpy array with a specific band.`
#     ```python
#     >>> import buteo as beo
#     >>> 
#     >>> raster = "/path/to/raster/raster.tif"
#     >>> 
#     >>> # Convert a raster to a numpy array
#     >>> array = beo.raster_to_array(raster, bands=[2])
#     >>> 
#     >>> array.shape, array.dtype
#     >>> (100, 100, 1), dtype('uint8')
#     ```
#     `Example 3: Convert a list of rasters to a numpy array with a specific`
#     ```python
#     >>> # band and a specific type and filled.
#     >>> from glob import glob
#     >>> import buteo as beo
#     >>> 
#     >>> FOLDER = "/path/to/folder"
#     >>> rasters = glob(FOLDER + "/*.tif")
#     >>>
#     >>> len(rasters)
#     >>> 10
#     >>> 
#     >>> # Convert rasters to a numpy array
#     >>> array = beo.raster_to_array(
#     ...     rasters,
#     ...     bands=[2],
#     ...     cast="float32",
#     ...     filled=True,
#     ...     fill_value=0.0,
#     ... )
#     >>> 
#     >>> # This raises an error if the 10 rasters are not aligned.
#     >>> array.shape, array.dtype
#     >>> (100, 100, 10), dtype('float32')
#     ```
#     """
#     utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
#     utils_base.type_check(bands, [int, [int], str], "bands")
#     utils_base.type_check(filled, [bool], "filled")
#     utils_base.type_check(fill_value, [int, float, None], "fill_value")
#     utils_base.type_check(masked, [bool, str], "masked")
#     utils_base.type_check(bbox, [list, None], "bbox")
#     utils_base.type_check(pixel_offsets, [list, tuple, None], "pixel_offsets")

#     if masked not in ["auto", True, False]:
#         raise ValueError(f"masked must be 'auto', True, or False. {masked} was provided.")

#     if bbox is not None and pixel_offsets is not None:
#         raise ValueError("Cannot use both bbox and pixel_offsets.")

#     internal_rasters = utils_base._get_variable_as_list(raster)

#     if not utils_gdal._check_is_raster_list(internal_rasters):
#         raise ValueError(f"An input raster is invalid. {internal_rasters}")

#     internal_rasters = utils_gdal._get_path_from_dataset_list(internal_rasters, dataset_type="raster")

#     if len(internal_rasters) > 1 and not core_raster.check_rasters_are_aligned(internal_rasters, same_extent=True, same_dtype=False):
#         raise ValueError(
#             "Cannot merge rasters that are not aligned, have dissimilar extent or dtype, when stack=True."
#         )

#     # Read metadata
#     metadata = core_raster._get_basic_metadata_raster(internal_rasters[0])
#     dtype = metadata["dtype"]
#     shape = metadata["shape"]

#     # Determine output shape
#     x_offset, y_offset, x_size, y_size = 0, 0, shape[1], shape[0]

#     if pixel_offsets is not None:
#         x_offset, y_offset, x_size, y_size = pixel_offsets

#         if x_offset < 0 or y_offset < 0:
#             raise ValueError("Pixel offsets cannot be negative.")

#         if x_offset + x_size > shape[1] or y_offset + y_size > shape[0]:
#             raise ValueError("Pixel offsets are outside of raster.")

#     elif bbox is not None:
#         if not utils_bbox._check_bboxes_intersect(metadata["bbox"], bbox):
#             raise ValueError("Extent is outside of raster.")

#         x_offset, y_offset, x_size, y_size = utils_bbox._get_pixel_offsets(metadata["transform"], bbox)

#     if (isinstance(bands, str) and bands.lower() == "all") or bands == -1:
#         output_shape = (y_size, x_size, len(internal_rasters) * shape[2])
#     else:
#         channels = 0
#         for in_raster in internal_rasters:
#             internal_bands = utils_gdal._convert_to_band_list(bands, core_raster._get_basic_metadata_raster(in_raster)["bands"])
#             channels += len(internal_bands)

#         output_shape = (y_size, x_size, channels)

#     # Determine nodata and value
#     if masked == "auto":
#         has_nodata = core_raster._check_raster_has_nodata_list(internal_rasters)
#         if has_nodata:
#             masked = True
#         else:
#             masked = False

#     output_nodata_value = None
#     if masked or filled:
#         output_nodata_value = core_raster._get_first_nodata_value(internal_rasters[0])

#         if output_nodata_value is None:
#             output_nodata_value = np.nan

#         if filled and fill_value is None:
#             fill_value = output_nodata_value

#     # Create output array
#     if masked:
#         output_arr = np.ma.empty(output_shape, dtype=dtype)
#         output_arr.mask = True

#         if filled:
#             output_arr.fill_value = fill_value
#         else:
#             output_arr.fill_value = output_nodata_value
#     else:
#         output_arr = np.empty(output_shape, dtype=dtype)


#     band_idx = 0
#     for in_raster in internal_rasters:

#         ref = core_raster._raster_open(in_raster)

#         metadata = core_raster._get_basic_metadata_raster(ref)
#         band_count = metadata["bands"]

#         if band_count == 0:
#             raise ValueError("The input raster does not have any valid bands.")

#         if bands == "all":
#             bands = -1

#         internal_bands = utils_gdal._convert_to_band_list(bands, metadata["bands"])

#         for band in internal_bands:
#             band_ref = ref.GetRasterBand(band)
#             band_nodata_value = band_ref.GetNoDataValue()

#             if pixel_offsets is not None or bbox is not None:
#                 arr = band_ref.ReadAsArray(x_offset, y_offset, x_size, y_size)
#             else:
#                 arr = band_ref.ReadAsArray()

#             if arr.shape[0] == 0 or arr.shape[1] == 0:
#                 raise RuntimeWarning("The output data has no rows or columns.")

#             if masked or filled:
#                 if band_nodata_value is not None:
#                     masked_arr = np.ma.array(arr, mask=arr == band_nodata_value, copy=False)
#                     masked_arr.fill_value = output_nodata_value

#                     if filled:
#                         arr = np.ma.getdata(masked_arr.filled(fill_value))
#                     else:
#                         arr = masked_arr

#             output_arr[:, :, band_idx] = arr

#             band_idx += 1

#         ref = None

#     if filled and np.ma.isMaskedArray(output_arr):
#         output_arr = np.ma.getdata(output_arr.filled(fill_value))

#     if cast is not None:
#         output_arr = output_arr.astype(cast, copy=False)

#     return output_arr


# class raster_to_array_chunks:
#     """
#     A class for reading raster data in chunks. The array will be split into x and y
#     amount of chunks in the x and y directions. The output will be the read array
#     and the offsets of the chunk in the raster. The offset can be used to reconstitute
#     the array into the original raster or a new raster representing the chunk,
#     using the :func:`array_to_raster` function.

#     Parameters
#     ----------
#     raster : Union[gdal.Dataset, str, List[Union[str, gdal.Dataset]]]
#         The raster to read.

#     chunks : int
#         The number of chunks to read. The area is chunked in way that ensures
#         that the chunks are as square as possible. Default: 1.

#     overlap : int, optional
#         The number of pixels to overlap. Default: 0.

#     overlap_y : int, optional
#         The number of pixels to overlap in the y direction. Default: 0.

#     bands : list or str or int, optional
#         The bands to read. Can be "all", an int, or a list of integers, or a single
#         integer. Please note that bands are 1-indexed. Default: "all".

#     masked : bool or str, optional
#         Whether to return a masked array. Default: "auto".

#     filled : bool, optional
#         Whether to fill masked values. Default: False.

#     fill_value : int or float, optional
#         The value to fill masked values with. Default: None.

#     cast : type or str, optional
#         The data type to cast the output to. Default: None.

#     Returns
#     -------
#     generator
#         A generator that yields the raster data in chunks and the offsets of the chunk
#         in the raster in a tuple.

#     Examples
#     --------
#     ```python
#     >>> # Read a raster into array via chunks.
#     >>> import buteo as beo
#     >>> 
#     >>> raster = "/path/to/raster/raster.tif"
#     >>> 
#     >>> shape = beo.raster_to_metadata(raster)["shape"]
#     >>> shape
#     >>> (100, 100)
#     >>> 
#     >>> for chunk, offsets in beo.raster_to_array_chunks(raster, chunks=4):
#     >>>     print(chunk.shape, offsets)
#     >>>     (25, 25), [0, 0, 25, 25]
#     ```
#     """

#     def __init__(
#         self,
#         raster: Union[gdal.Dataset, str, List[Union[str, gdal.Dataset]]],
#         chunks: int = 1,
#         *,
#         overlap: int = 0,
#         bands: Union[List[int], str, int] = 'all',
#         masked: Union[bool, str] = "auto",
#         filled: bool = False,
#         fill_value: Optional[Union[int, float]] = None,
#         cast: Optional[Union[np.dtype, str]] = None,
#     ):
#         self.raster = raster
#         self.chunks = chunks
#         self.overlap = overlap
#         self.bands = bands
#         self.masked = masked
#         self.filled = filled
#         self.fill_value = fill_value
#         self.cast = cast
#         self.current_chunk = 0

#         self.shape = core_raster._get_basic_metadata_raster(self.raster)["shape"]

#         assert self.chunks > 0, "The number of chunks must be greater than 0."
#         assert self.overlap >= 0, "The overlap must be greater than or equal to 0."
#         assert self.chunks <= self.shape[1], "The number of chunks must be less than or equal to the number of columns in the raster."
#         assert self.chunks <= self.shape[0], "The number of chunks must be less than or equal to the number of rows in the raster."

#         self.offsets = core_offsets._get_chunk_offsets(
#             self.shape,
#             self.chunks,
#             self.overlap,
#         )

#         self.total_chunks = len(self.offsets)

#     def __iter__(self):
#         self.current_chunk = 0
#         return self

#     def __next__(self) -> Tuple[np.ndarray, List[int]]:
#         if self.current_chunk >= self.total_chunks:
#             raise StopIteration

#         offset = self.offsets[self.current_chunk]
#         self.current_chunk += 1

#         return (
#             raster_to_array(
#                 self.raster,
#                 bands=self.bands,
#                 masked=self.masked,
#                 filled=self.filled,
#                 fill_value=self.fill_value,
#                 pixel_offsets=offset,
#                 cast=self.cast,
#             ),
#             offset,
#         )

#     def __len__(self):
#         return self.total_chunks

def array_to_raster():
    return

# def array_to_raster(
#     array: np.ndarray,
#     *,
#     reference: Union[str, gdal.Dataset],
#     out_path: Optional[str] = None,
#     set_nodata: Union[bool, float, int, str] = "arr",
#     allow_mismatches: bool = False,
#     pixel_offsets: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
#     bbox: Optional[List[float]] = None,
#     overwrite: bool = True,
#     creation_options: Optional[List[str]] = None,
# ) -> str:
#     """
#     Turns a NumPy array into a GDAL dataset or exported as a raster using a reference raster.

#     Parameters
#     ----------
#     array : np.ndarray
#         The numpy array to convert.

#     reference : str or gdal.Dataset
#         The reference raster to use for the output.

#     out_path : path, optional
#         The destination to save to. Default: None.

#     set_nodata : bool or float or int, optional
#         Can be set to:
#             - "arr": The nodata value will be the same as the NumPy array.
#             - "ref": The nodata value will be the same as the reference raster.
#             - value: The nodata value will be the value provided. Default: "arr".

#     allow_mismatches : bool, optional
#         If True, the array can have a different shape than the reference raster.
#         Default: False.

#     pixel_offsets : list or tuple, optional
#         If provided, the array will be written to the reference raster at the
#         specified pixel offsets. The list should be in the format [x_offset, y_offset, x_size, y_size].
#         Default: None.

#     bbox : list, optional
#         If provided, the array will be written to the reference raster at the specified
#         bounding box. The list should be in the format [min_x, min_y, max_x, max_y]. Default: None.

#     overwrite : bool, optional
#         If the file exists, should it be overwritten? Default: True.

#     creation_options : list, optional
#         List of GDAL creation options. Default: ["TILED=YES", "NUM_THREADS=ALL_CPUS",
#         "BIGTIFF=YES", "COMPRESS=LZW"].

#     Returns
#     -------
#     str
#         The file path to the newly created raster(s).

#     Examples
#     --------
#     ```python
#     >>> # Create a raster from a numpy array.
#     >>> import buteo as beo
#     >>> 
#     >>> raster = "/path/to/raster/raster.tif"
#     >>> 
#     >>> array = beo.raster_to_array(raster)
#     >>> array = array ** 2
#     >>> 
#     >>> out_path = beo.array_to_raster(
#     ...     array,
#     ...     reference=raster,
#     ...     out_path="/path/to/new/new_raster.tif"
#     ... )
#     >>> 
#     >>> out_path
#     >>> "/path/to/new/new_raster.tif"
#     ```
#     """
#     utils_base.type_check(array, [np.ndarray, np.ma.MaskedArray], "array")
#     utils_base.type_check(reference, [str, gdal.Dataset], "reference")
#     utils_base.type_check(out_path, [str, None], "out_path")
#     utils_base.type_check(overwrite, [bool], "overwrite")
#     utils_base.type_check(pixel_offsets, [[int, float], tuple, None], "pixel_offsets")
#     utils_base.type_check(allow_mismatches, [bool], "allow_mismatches")
#     utils_base.type_check(set_nodata, [int, float, str, None], "set_nodata")
#     utils_base.type_check(creation_options, [[str], None], "creation_options")

#     # Verify the numpy array
#     if (
#         array.size == 0
#         or array.ndim < 2
#         or array.ndim > 3
#     ):
#         raise ValueError(f"Input array is invalid {array}")

#     if set_nodata not in ["arr", "ref"]:
#         utils_base.type_check(set_nodata, [int, float], "set_nodata")

#     if pixel_offsets is not None:
#         if len(pixel_offsets) != 4:
#             raise ValueError("pixel_offsets must be a list of 4 values.")

#     if pixel_offsets is not None and bbox is not None:
#         raise ValueError("pixel_offsets and bbox cannot be used together.")

#     # Parse the driver
#     driver_name = "GTiff" if out_path is None else utils_gdal._get_raster_driver_from_path(out_path)
#     if driver_name is None:
#         raise ValueError(f"Unable to parse filetype from path: {out_path}")

#     driver = gdal.GetDriverByName(driver_name)
#     if driver is None:
#         raise ValueError(f"Error while creating driver from extension: {out_path}")

#     # How many bands?
#     bands = 1
#     if array.ndim == 3:
#         bands = array.shape[2]

#     output_name = None
#     if out_path is None:
#         output_name = utils_path._get_augmented_path_list("array_to_raster.tif", add_uuid=True, folder="/vsimem/")
#     else:
#         output_name = out_path

#     utils_path._delete_if_required(output_name, overwrite)

#     metadata = core_raster._get_basic_metadata_raster(reference)
#     reference_nodata = metadata["nodata_value"]

#     # handle nodata. GDAL python throws error if conversion in not explicit.
#     if reference_nodata is not None:
#         reference_nodata = float(reference_nodata)
#         if (reference_nodata).is_integer() is True:
#             reference_nodata = int(reference_nodata)

#     # Handle nodata
#     input_nodata = None
#     if np.ma.is_masked(array) is True:
#         input_nodata = array.get_fill_value()  # type: ignore (because it's a masked array.)

#     destination_dtype = utils_translate._translate_str_to_gdal_dtype(array.dtype)

#     # Weird double issue with GDAL and numpy. Cast to float or int
#     if input_nodata is not None:
#         input_nodata = float(input_nodata)
#         if (input_nodata).is_integer() is True:
#             input_nodata = int(input_nodata)

#     if (metadata["width"] != array.shape[1] or metadata["height"] != array.shape[0]) and pixel_offsets is None and bbox is None:
#         if not allow_mismatches:
#             raise ValueError(f"Input array and raster are not of equal size. Array: {array.shape[:2]} Raster: {metadata['width'], metadata['height']}")

#         warnings.warn(f"Input array and raster are not of equal size. Array: {array.shape[:2]} Raster: {metadata['shape'][:2]}", UserWarning)

#     if bbox is not None:
#         pixel_offsets = utils_bbox._get_pixel_offsets(metadata["transform"], bbox)

#     if pixel_offsets is not None:
#         x_offset, y_offset, x_size, y_size = pixel_offsets

#         if array.ndim == 3:
#             array = array[:y_size, :x_size:, :] # numpy is col, row order
#         else:
#             array = array[:y_size, x_size]

#         metadata["transform"] = (
#             metadata["transform"][0] + (x_offset * metadata["pixel_width"]),
#             metadata["transform"][1],
#             metadata["transform"][2],
#             metadata["transform"][3] - (y_offset * metadata["pixel_height"]),
#             metadata["transform"][4],
#             metadata["transform"][5],
#         )

#     destination = driver.Create(
#         output_name,
#         array.shape[1],
#         array.shape[0],
#         bands,
#         destination_dtype,
#         utils_gdal._get_default_creation_options(creation_options),
#     )

#     destination.SetProjection(metadata["projection_wkt"])
#     destination.SetGeoTransform(metadata["transform"])

#     for band_idx in range(bands):
#         band = destination.GetRasterBand(band_idx + 1)
#         band.SetColorInterpretation(gdal.GCI_Undefined)

#         if bands > 1 or array.ndim == 3:
#             band.WriteArray(array[:, :, band_idx])
#         else:
#             band.WriteArray(array)

#         if set_nodata == "ref" and reference_nodata is not None:
#             band.SetNoDataValue(reference_nodata)
#         elif set_nodata == "arr" and input_nodata is not None:
#             band.SetNoDataValue(input_nodata)
#         elif isinstance(set_nodata, (int, float)):
#             band.SetNoDataValue(set_nodata)

#     destination.FlushCache()
#     destination = None

#     return output_name



# # TODO: Verify this function, it looks funky.
# def save_dataset_to_disk(
#     dataset: Union[gdal.Dataset, ogr.DataSource, str],
#     out_path: Union[str, List[str]],
#     overwrite: bool = True,
#     creation_options: Optional[List[str]] = None,
#     prefix: str = "",
#     suffix: str = "",
#     add_uuid: bool = False,
# ) -> Union[str, List[str]]:
#     """
#     Writes a dataset to disk. Can be a raster or a vector.

#     Parameters
#     ----------
#     dataset : Union[gdal.Dataset, ogr.DataSource, str]
#         The dataset to write to disk.

#     out_path : Union[str, List[str]]
#         The output path or list of output paths.
    
#     overwrite : bool, optional
#         If True, the output will be overwritten if it already exists. Default: True.

#     creation_options : Optional[List[str]], optional
#         A list of creation options. Default: None.

#     prefix : str, optional
#         A prefix to add to the output path. Default: "".

#     suffix : str, optional
#         A suffix to add to the output path. Default: "".

#     add_uuid : bool, optional
#         If True, a UUID will be added to the output path. Default: False.

#     Returns
#     -------
#     Union[str, List[str]]
#         The output path or list of output paths.
#     """
#     datasets = utils_base._get_variable_as_list(dataset)
#     datasets_paths = _get_path_from_dataset_list(datasets, allow_mixed=True)
#     out_paths = utils_path._get_output_path_list(
#         datasets_paths,
#         out_path,
#         prefix=prefix,
#         suffix=suffix,
#         add_uuid=add_uuid,
#     )

#     options = None

#     for index, dataset_ in enumerate(datasets):
#         opened_dataset = None
#         dataset_type = None

#         if _check_is_raster(dataset_):
#             options = _get_default_creation_options(creation_options)
#             dataset_type = "raster"
#             if isinstance(dataset_, str):
#                 opened_dataset = gdal.Open(dataset_, 0)
#             elif isinstance(dataset_, gdal.Dataset):
#                 opened_dataset = dataset_
#             else:
#                 raise RuntimeError(f"Could not read input raster: {dataset_}")

#         elif _check_is_vector(dataset_):
#             dataset_type = "vector"
#             if isinstance(dataset_, str):
#                 opened_dataset = ogr.Open(dataset_, 0)
#             elif isinstance(dataset_, ogr.DataSource):
#                 opened_dataset = dataset_
#             else:
#                 raise RuntimeError(f"Could not read input vector: {dataset_}")

#         else:
#             raise RuntimeError(f"Invalid dataset type: {dataset_}")

#         driver_destination = None

#         if dataset_type == "raster":
#             driver_destination = gdal.GetDriverByName(_get_raster_driver_from_path(out_paths[index]))
#         else:
#             driver_destination = ogr.GetDriverByName(_get_vector_driver_from_path(out_paths[index]))

#         assert driver_destination is not None, "Could not get driver for output dataset."

#         utils_path._delete_if_required(out_paths[index], overwrite)

#         driver_destination.CreateCopy(
#             out_path[index],
#             opened_dataset,
#             options=options,
#         )

#     if isinstance(dataset, list):
#         return out_paths[0]

#     return out_paths


# def save_dataset_to_memory(
#     dataset: Union[gdal.Dataset, ogr.DataSource, str],
#     overwrite: bool = True,
#     creation_options: Optional[List[str]] = None,
#     prefix: str = "",
#     suffix: str = "",
#     add_uuid: bool = True,
# ) -> Union[str, List[str]]:
#     """
#     Writes a dataset to memory. Can be a raster or a vector.

#     Parameters
#     ----------
#     dataset : Union[gdal.Dataset, ogr.DataSource, str]
#         The dataset to write to memory.

#     overwrite : bool, optional
#         If True, the output will be overwritten if it already exists. Default: True.

#     creation_options : Optional[List[str]], optional
#         A list of creation options. Default: None.

#     prefix : str, optional
#         A prefix to add to the output path. Default: "".

#     suffix : str, optional
#         A suffix to add to the output path. Default: "".

#     add_uuid : bool, optional
#         If True, a UUID will be added to the output path. Default: False.

#     Returns
#     -------
#     Union[str, List[str]]
#         The output path or list of output paths.
#     """
#     return save_dataset_to_disk(
#         dataset,
#         out_path=None,
#         overwrite=overwrite,
#         creation_options=creation_options,
#         prefix=prefix,
#         suffix=suffix,
#         add_uuid=add_uuid,
#     )
