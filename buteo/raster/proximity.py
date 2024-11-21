"""### Calculate distances on a raster. ###

Module to calculate the distance from a pixel value to other pixels.
"""

# Standard library
from typing import Union, List, Optional

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_base,
    utils_io,
    utils_path,
)
from buteo.raster import core_raster, core_raster_io
from buteo.array.distance import convolve_distance



def _raster_get_proximity(
    raster: Union[str, gdal.Dataset],
    target_value: Union[int, float] = 1,
    unit: str = "geo",
    out_path: Optional[str] = None,
    max_dist: Union[int, float] = 1000,
    add_border: bool = False,
    border_value: Union[int, float] = 0,
    inverted: bool = False,
    overwrite: bool = True,
) -> str:
    """Internal. Calculate the proximity of a raster to values."""
    assert isinstance(raster, (str, gdal.Dataset)), f"Invalid raster. {raster}"
    assert isinstance(target_value, (int, float)), f"Invalid target_value. {target_value}"
    assert isinstance(unit, str), f"Invalid unit. {unit}"

    if out_path is None:
        out_path = utils_path._get_temp_filepath("proximity_raster.tif")

    array = core_raster_io.raster_to_array(raster, filled=True)

    if unit.lower() == "geo":
        metadata = core_raster._get_basic_metadata_raster(raster)
        pixel_height = metadata["pixel_height"]
        pixel_width = metadata["pixel_width"]
    else:
        pixel_height = 1
        pixel_width = 1

    if add_border:
        padded = np.full((array.shape[0] + 2, array.shape[1] + 2, array.shape[2]), border_value, dtype=array.dtype)
        padded[1:-1, 1:-1] = array
        array = padded

    array = convolve_distance(
        array,
        target=target_value,
        maximum_distance=max_dist,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )

    if inverted:
        array = array.max() - array

    if add_border:
        array = array[1:-1, 1:-1]

    out_path = core_raster_io.array_to_raster(
        array,
        reference=raster,
        out_path=out_path,
        overwrite=overwrite,
    )

    return out_path


def raster_get_proximity(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    target_value: Union[int, float] = 1,
    unit: str = "geo",
    out_path: Optional[Union[str, List[str]]] = None,
    max_dist: Union[int, float] = 1000,
    add_border: bool = False,
    border_value: Union[int, float] = 0,
    inverted: bool = False,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> Union[str, List[str]]:
    """Calculate the proximity of input_raster to values.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, List]
        The raster(s) to use as input.

    target_value : Union[int, float], optional
        The value to use as target, default: 1.

    unit : str, optional
        The unit to use for the distance, GEO or PIXEL, default: "GEO".

    out_path : Union[str, None, List], optional
        The output path, default: None.

    max_dist : Union[int, float], optional
        The maximum distance to use, default: 1000.

    add_border : bool, optional
        If True, a border will be added to the raster, default: False.

    border_value : Union[int, float], optional
        The value to use for the border, default: 0.

    inverted : bool, optional
        If True, the target will be inversed, default: False.

    overwrite : bool, optional
        If the output path exists already, should it be overwritten?, default: True.

    prefix : str, optional
        Prefix to add to the output, default: "".

    suffix : str, optional
        Suffix to add to the output, default: "_proximity".

    add_uuid : bool, optional
        Should a uuid be added to the output path?, default: False.

    add_timestamp : bool, optional
        Should a timestamp be added to the output path?, default: False.

    Returns
    -------
    str
        A path to a raster with the calculated proximity.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(target_value, [int, float], "target_value")
    utils_base._type_check(unit, [str], "unit")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(max_dist, [int, float], "max_dist")
    utils_base._type_check(add_border, [bool], "add_border")
    utils_base._type_check(border_value, [int, float], "border_value")
    utils_base._type_check(inverted, [bool], "inverted")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    assert unit.lower() in ["geo", "pixel"], f"Invalid unit. {unit}"

    input_is_list = isinstance(raster, list)
    input_rasters = utils_io._get_input_paths(raster, "raster")
    output_rasters = utils_io._get_output_paths(
        input_rasters,
        out_path,
        in_place=False,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext="tif"
    )

    utils_path._delete_if_required_list(output_rasters, overwrite)

    output = []
    for idx, in_raster in enumerate(input_rasters):
        output.append(_raster_get_proximity(
            in_raster,
            target_value=target_value,
            unit=unit.lower(),
            out_path=output_rasters[idx],
            max_dist=max_dist,
            add_border=add_border,
            border_value=border_value,
            inverted=inverted,
            overwrite=overwrite,
        ))

    if input_is_list:
        return output

    return output[0]
