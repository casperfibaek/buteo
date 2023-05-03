"""
### Functions for changing the datatype of a raster. ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, List, Optional
from warnings import warn
from uuid import uuid4

# External
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_gdal,
    utils_base,
    utils_path,
    utils_translate,
)
from buteo.raster import core_raster



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

    if not core_raster.check_rasters_are_aligned(rasters):
        raise ValueError("Rasters are not aligned. Try running align_rasters.")

    # Ensures that all the input rasters are valid.
    raster_list = utils_gdal._get_path_from_dataset_list(rasters)

    if out_path is not None and utils_path._get_ext_from_path(out_path) == ".vrt":
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

    raster_dtype = core_raster._get_basic_metadata_raster(raster_list[0])["dtype_gdal"]

    datatype = raster_dtype
    if dtype is not None:
        datatype = utils_translate._translate_str_to_gdal_dtype(dtype)

    nodata_values = []
    nodata_missmatch = False
    nodata_value = None
    total_bands = 0
    metadatas = []

    for raster in raster_list:
        metadata = core_raster._get_basic_metadata_raster(raster)
        metadatas.append(metadata)

        nodata_value = metadata["nodata_value"]
        total_bands += metadata["bands"]

        if nodata_missmatch is False:
            for ndv in nodata_values:
                if nodata_missmatch:
                    continue

                if metadata["nodata_value"] != ndv:
                    nodata_missmatch = True
                    warn("NoDataValues of input rasters do not match. Removing nodata.", UserWarning)

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
        ref = core_raster._raster_open(raster)

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
