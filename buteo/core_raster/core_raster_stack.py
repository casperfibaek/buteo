"""### Functions for changing the datatype of a raster. ###"""

# Standard library
from typing import Union, List, Optional

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_path,
    utils_translate,
    utils_gdal,
)
from buteo.core_raster.core_raster_extent import check_rasters_are_aligned
from buteo.core_raster.core_raster_array import raster_to_array, array_to_raster
from buteo.core_raster.core_raster_info import get_metadata_raster



def raster_stack_list(
    rasters: List[Union[str, gdal.Dataset]],
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    dtype: Optional[Union[str, np.dtype, type]] = None,
    creation_options: Optional[List[str]] = None,
) -> str:
    """Stack aligned rasters into a single raster file.

    Parameters
    ----------
    rasters : List[Union[str, gdal.Dataset]]
        List of rasters to stack. These rasters must be aligned.
    out_path : Optional[str]
        The destination to save the output raster. If not provided, creates a temporary file.
    overwrite : bool
        If True, overwrite existing file.
    dtype : Optional[Union[str, np.dtype, type]]
        Output raster data type. If None, uses input raster dtype.
    creation_options : Optional[List[str]]
        GDAL creation options. Defaults to None.

    Returns
    -------
    str
        Path to the created raster.

    Raises
    ------
    ValueError
        If rasters are not aligned or invalid.
    TypeError
        If input parameters have invalid types.
    """
    utils_base._type_check(rasters, [[str, gdal.Dataset]], "rasters")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(dtype, [str, None, np.dtype, type], "dtype")
    utils_base._type_check(creation_options, [[str], None], "creation_options")

    input_data = utils_io._get_input_paths(rasters, "raster") # type: ignore

    if not utils_gdal._check_is_raster_list(input_data):
        raise ValueError("Input must be a list of valid rasters")

    if not check_rasters_are_aligned(input_data, same_bands=True): # type: ignore
        raise ValueError("Input rasters must be aligned")

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_stack", ext="tif", add_uuid=True)

    if not utils_path._check_is_valid_output_filepath(out_path):
        raise ValueError(f"Invalid output path: {out_path}")

    arrays = [raster_to_array(raster, cast=dtype) for raster in input_data]

    # stack along the first axis
    arrays_stacked = np.concatenate(arrays, axis=0)

    return array_to_raster(
        arrays_stacked,
        reference=input_data[0],
        out_path=out_path,
        overwrite=overwrite,
        creation_options=creation_options,
    )


def raster_stack_vrt_list(
    rasters: List[Union[str, gdal.Dataset]],
    out_path: Optional[str]=None,
    separate: bool = True,
    *,
    resample_alg: str = "nearest",
    nodata_src: Optional[float] = None,
    nodata_VRT: Optional[float] = None,
    nodata_hide: Optional[bool] = None,
    overwrite: bool = True,
    reference: Optional[str] = None,
) -> str:
    """Create a virtual raster (.vrt) by stacking a list of input rasters.

    The function accepts a list of rasters and creates a virtual raster stack
    by combining them. The rasters can be stacked by keeping their bands separate,
    or by merging the bands in the order of the input rasters.

    Parameters
    ----------
    rasters : List[Union[str, gdal.Dataset]]
        A list of input rasters as either file paths or GDAL datasets to be stacked.
    out_path : Optional[str], default=None
        The destination file path to save the output VRT raster. If not provided, a
        temporary file path will be generated.
    separate : bool, default=True
        If True, the raster bands will be kept separate and stacked in the order of
        the input rasters. If False, the raster bands will be merged, but all input
        rasters must have the same number of bands.
    resample_alg : str, default='nearest'
        The resampling algorithm to use when building the VRT. Accepts any algorithm
        supported by GDAL's BuildVRT function (e.g., 'nearest', 'bilinear', 'cubic').
    nodata_src : Optional[float], default=None
        The NoData value to use for the source rasters. If not provided, the NoData
        value from the input rasters will be used.
    nodata_VRT : Optional[float], default=None
        The NoData value to use for the output VRT raster. If not provided, the NoData
        value from the input rasters will be used.
    nodata_hide : Optional[bool], default=None
        If True, the NoData value will be hidden in the VRT. If not provided, the value
        will be determined by the input rasters.
    overwrite : bool, default=True
        If True and the output file exists, it will be overwritten. If False and the
        output file exists, an error will be raised.
    reference : Optional[str], default=None
        A reference raster file path or GDAL dataset to use for aligning the stacked
        rasters. If not provided, the alignment of the input rasters will be used.

    Returns
    -------
    str
        The file path to the newly created VRT raster.
    """
    # Type checking for input parameters
    utils_base._type_check(rasters, [[str, gdal.Dataset]], "rasters")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(separate, [bool], "separate")
    utils_base._type_check(resample_alg, [str], "resample_alg")
    utils_base._type_check(overwrite, [bool], "overwrite")

    # Check if the input rasters are valid
    if len(rasters) == 0:
        raise ValueError("Input rasters list is empty.")

    if not utils_gdal._check_is_raster_list(rasters):
        raise ValueError("Input rasters must be a list of valid rasters.")

    # Check if all rasters have the same number of bands when separate is False
    if not separate:
        master_bands = 0

        for idx, raster_input in enumerate(rasters):
            if idx == 0:
                master_bands = get_metadata_raster(raster_input)["bands"]
            else:
                if master_bands != get_metadata_raster(raster_input)["bands"]:
                    raise ValueError("All rasters must have the same number of bands when separate is False.")

    # Generate a temporary output file path if out_path is not provided
    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            "temp_stack",
            ext="vrt",
            add_uuid=True,
            add_timestamp=True,
        )

    # Check if the output file path is valid
    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    # Translate the input resample algorithm to a GDAL-compatible format
    resample_algorithm = utils_translate._translate_resample_method(resample_alg)

    # Build VRT options based on whether a reference raster is provided or not
    if reference is not None:
        meta = get_metadata_raster(reference)
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg=resample_algorithm,
            separate=separate,
            outputBounds=meta["bbox_gdal"],
            xRes=meta["pixel_width"],
            yRes=meta["pixel_height"],
            targetAlignedPixels=True,
            srcNodata=nodata_src,
            VRTNodata=nodata_VRT,
            hideNodata=nodata_hide,
        )
    else:
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg=resample_algorithm,
            separate=separate,
            srcNodata=nodata_src,
            VRTNodata=nodata_VRT,
            hideNodata=nodata_hide,
        )

    # Create separate VRTs for each band if separate is True
    if separate:
        tmp_vrt_list = []

        for idx, raster in enumerate(rasters):
            bands_in_raster = get_metadata_raster(raster)["bands"]

            for band in range(bands_in_raster):
                tmp_vrt_path = utils_path._get_temp_filepath(
                    "temp_vrt",
                    ext="vrt",
                    add_uuid=True,
                    add_timestamp=True,
                )

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

        vrt = gdal.BuildVRT(out_path, tmp_vrt_list, options=vrt_options)

        # Clean up temporary VRTs
        for tmp_vrt_path in tmp_vrt_list:
            gdal.Unlink(tmp_vrt_path)

    # Create a VRT by merging bands if separate is False
    else:
        vrt = gdal.BuildVRT(out_path, rasters, options=vrt_options)

    # Flush cache to ensure data is written to the output file
    vrt.FlushCache()

    # Check for errors in VRT creation
    if vrt is None:
        raise ValueError(f"Error while creating VRT from rasters: {rasters}")

    # Release the VRT object to avoid potential memory leaks
    vrt = None

    return out_path
