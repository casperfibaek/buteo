"""### Handle nodata values in rasters. ###

A module to handle the various aspects of NODATA in raster files.
"""

# Standard library
from typing import Union, Optional, List

# External
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_translate,
)
from buteo.core_raster.core_raster_read import _open_raster
from buteo.core_raster.core_raster_info import get_metadata_raster
from buteo.core_raster.core_raster_write import raster_create_copy


def check_rasters_have_same_nodata(
    rasters: List[Union[str, gdal.Dataset]],
) -> bool:
    """Verifies whether a list of rasters have the same nodata values.

    Parameters
    ----------
    rasters : List[Union[str, gdal.Dataset]]
        A list of rasters to check.

    Returns
    -------
    bool
        True if all rasters have the same nodata value, False otherwise.

    Raises
    ------
    TypeError
        If rasters is not a list of strings or gdal.Datasets.
    ValueError
        If rasters is empty or if any raster cannot be opened.
    """
    utils_base._type_check(rasters, [[str, gdal.Dataset]], "rasters")

    if len(rasters) == 0:
        raise ValueError("Input rasters list is empty.")

    if len(rasters) == 1:
        return True

    # Open first raster and get nodata value
    first_ds = _open_raster(rasters[0])
    first_nodata = first_ds.GetRasterBand(1).GetNoDataValue()

    # Compare with remaining rasters
    for raster in rasters[1:]:
        ds = _open_raster(raster)
        if ds.GetRasterBand(1).GetNoDataValue() != first_nodata:
            return False

    return True


def _check_raster_has_nodata(raster: Union[str, gdal.Dataset]) -> bool:
    """Check if raster has nodata values for any band.

    Parameters
    ----------
    raster : str or gdal.Dataset
        Raster dataset or path to raster file

    Returns
    -------
    bool
        True if any band has nodata values, False otherwise

    Raises
    ------
    TypeError
        If raster is not str or gdal.Dataset
    ValueError
        If raster cannot be opened
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")

    dataset = _open_raster(raster, writeable=False)

    for band_idx in range(1, dataset.RasterCount + 1):
        if dataset.GetRasterBand(band_idx).GetNoDataValue() is not None:
            return True

    return False


def _raster_has_nodata(
    raster: Union[str, gdal.Dataset],
) -> bool:
    """Internal. Check if a raster or a list of rasters contain nodata values.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The raster to check for nodata values.

    Returns
    -------
    bool
        True if the raster has nodata values
    """
    assert isinstance(raster, (str, gdal.Dataset)), f"Invalid raster. {raster}"

    metadata = get_metadata_raster(raster)

    if metadata["nodata"]:
        return True

    return False


def raster_has_nodata(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
) -> Union[bool, List[bool]]:
    """Check if a raster or a list of rasters contain nodata values.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, List]
        The raster to check for nodata values.

    Returns
    -------
    Union[bool, List[bool]
        True if the raster or list of rasters contain nodata values.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    input_is_list = isinstance(raster, list)

    input_paths = utils_io._get_input_paths(raster, "raster") # type: ignore

    nodata_values = []
    for in_raster in input_paths:
        nodata_values.append(_raster_has_nodata(in_raster))

    if input_is_list:
        return nodata_values

    return nodata_values[0]


def _raster_get_nodata(
    raster: Union[str, gdal.Dataset],
) -> Union[float, int]:
    """Internal. Get the nodata value of a raster.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The raster to get nodata values from.

    Returns
    -------
    Union[float, int]
        The nodata value of the raster.
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")

    metadata = get_metadata_raster(raster)

    return metadata["nodata_value"]


def raster_get_nodata(
    raster: Union[str, gdal.Dataset, list[Union[str, gdal.Dataset]]],
) -> Union[float, int, list[Union[float, int]]]:
    """Get the nodata value of a raster or a sequence of rasters.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, Sequence]
        The raster to get nodata values from.

    Returns
    -------
    Union[float, int, None, List[Union[float, int, None]]]
        The nodata value of the raster or sequence of rasters.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    input_is_list = isinstance(raster, list)

    input_paths = utils_io._get_input_paths(raster, "raster") # type: ignore

    nodata_values = []
    for in_raster in input_paths:
        nodata_values.append(_raster_get_nodata(in_raster))

    if input_is_list:
        return nodata_values

    return nodata_values[0]


def _raster_set_nodata(
    raster: Union[str, gdal.Dataset],
    nodata: Union[float, int, None],
    out_path: Optional[str] = None,
    in_place: bool = True,
    overwrite: bool = True,
) -> Union[str, gdal.Dataset]:
    """Internal. Sets the nodata value of a single raster.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The raster to set nodata values for.
    nodata : float, int, or None
        The nodata value to set for the raster.
        If nodata is None, nodata will be removed from the raster.
    out_path : str, optional
        The destination of the changed raster, default: None
    in_place : bool, optional
        Should the raster be changed in_place or copied?, default: True

    Returns
    -------
    str
        Returns the path to the raster with nodata set.
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")
    utils_base._type_check(nodata, [float, int, None], "nodata")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(in_place, [bool], "in_place")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if in_place:
        opened = _open_raster(raster, writeable=True)
    else:
        copy = raster_create_copy(raster, out_path, overwrite=overwrite)
        opened = _open_raster(copy, writeable=True)

    bands = opened.RasterCount

    dtype = utils_translate._translate_dtype_gdal_to_numpy(opened.GetRasterBand(1).DataType)

    if nodata is not None:
        if not utils_translate._check_is_value_within_dtype_range(nodata, dtype):
            raise ValueError(f"Invalid nodata value for datatype. value: {nodata}, dtype: {dtype}")

    if nodata is None:
        for band in range(bands):
            raster_band = opened.GetRasterBand(band + 1)
            raster_band.DeleteNoDataValue()
            raster_band = None
    else:
        for band in range(bands):
            raster_band = opened.GetRasterBand(band + 1)
            raster_band.SetNoDataValue(nodata)
            raster_band = None


    opened.FlushCache()
    opened = None

    if out_path is None:
        raise ValueError("Output path is None. This should not happen.")

    if in_place:
        return raster

    return out_path


def raster_set_nodata(
    raster: Union[str, gdal.Dataset, List],
    nodata: Union[float, int, None],
    out_path: Optional[str] = None,
    in_place: bool = True,
    *,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    creation_options: Optional[List] = None,
):
    """Sets all the nodata for raster(s) to a value.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, List]
        The raster(s) to set nodata values for.
    nodata : float, int, or None
        The nodata value to set for the raster(s).
        If nodata is None, nodata will be removed from the raster(s).
    out_path : str, optional
        The destination of the changed rasters, default: None
    in_place : bool, optional
        Should the rasters be changed in_place or copied?, default: True
    overwrite : bool, optional
        Should the rasters be overwritten if they already exist? default: True
    prefix : str, optional
        Prefix to add to the output, default: ""
    suffix : str, optional
        Suffix to add to the output, default: "_nodata_set"
    creation_options : List, optional
        Creation options for the output rasters, default: None

    Returns
    -------
    Union[str, List]
        Returns the rasters with nodata set.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(nodata, [float, int, None], "nodata")
    utils_base._type_check(out_path, [list, str, None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "postfix")
    utils_base._type_check(creation_options, [[str], None], "creation_options")

    input_is_list = isinstance(raster, list)

    in_paths = utils_io._get_input_paths(raster, "raster")
    out_paths = utils_io._get_output_paths(
        in_paths, # type: ignore
        out_path,
        in_place=in_place,
        prefix=prefix,
        suffix=suffix,
    )

    if not in_place:
        utils_io._check_overwrite_policy(out_paths, overwrite)
        utils_io._delete_if_required_list(out_paths, overwrite)

    nodata_set = []
    for idx, in_raster in enumerate(in_paths):
        nodata_set.append(_raster_set_nodata(
            in_raster,
            nodata,
            out_path=out_paths[idx],
            in_place=in_place,
            overwrite=overwrite,
        ))

    if input_is_list:
        return out_paths

    return out_paths[0]
