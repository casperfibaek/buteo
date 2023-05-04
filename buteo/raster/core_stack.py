"""
### Functions for changing the datatype of a raster. ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, List, Optional

# External
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_path,
    utils_translate,
)
from buteo.raster import core_raster, core_io



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

    input_data = utils_io._get_input_paths(rasters, "raster")

    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            "temp_stack",
            ext="tif",
            add_uuid=True,
            add_timestamp=True,
        )
    else:
        assert utils_path._check_is_valid_output_filepath(out_path), "Invalid output path."

    assert core_raster.check_rasters_are_aligned(input_data), "Rasters are not aligned."

    array = core_io.raster_to_array(input_data, cast=dtype)
    out_path = core_io.array_to_raster(
        array,
        reference=input_data[0],
        out_path=out_path,
        overwrite=overwrite,
        creation_options=creation_options,
    )

    return out_path


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
                master_bands = core_raster._get_basic_metadata_raster(raster_input)["bands"]
            else:
                if master_bands != core_raster._get_basic_metadata_raster(raster_input)["bands"]:
                    raise ValueError("All rasters must have the same number of bands.")

    resample_algorithm = utils_translate._translate_resample_method(resample_alg)

    if reference is not None:
        meta = core_raster._get_basic_metadata_raster(reference)
        options = gdal.BuildVRTOptions(
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
            bands_in_raster = core_raster._get_basic_metadata_raster(raster)["bands"]

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


# TODO: Mosaic raster(s)
# TODO: Use gdaltools?