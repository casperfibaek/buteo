"""### Extract the footprints or centroids of rasters. ###"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List
import os

# External
from osgeo import gdal

# Internal
from buteo.raster import core_raster
from buteo.utils import (
    utils_base,
    utils_bbox,
    utils_path,
    utils_io,
    utils_projection,
)



def raster_get_footprints(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    latlng: bool = True,
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    out_format: str = "gpkg",
) -> Union[str, List[str], gdal.Dataset]:
    """Gets the footprints of a raster or a list of rasters.

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be shifted.

    latlng : bool, optional
        If True, the footprints are returned in lat/lon coordinates. If False, the footprints are returned in projected coordinates., default: True

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., default: None

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists., default: True

    prefix : str, optional
        The prefix to be added to the output raster name., default: ""

    suffix : str, optional
        The suffix to be added to the output raster name., default: ""

    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name., default: False

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name., default: False

    out_format : str, optional
        The output format of the raster. If None, the format is inferred from the output path., default: "gpkg"

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the shifted raster(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(latlng, [bool], "latlng")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    input_list = utils_io._get_input_paths(raster, "raster")
    output_list = utils_io._get_output_paths(
        input_list,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext=out_format,
    )

    utils_path._delete_if_required_list(output_list, overwrite)

    footprints = []
    for idx, in_raster in enumerate(input_list):
        metadata = core_raster._get_basic_metadata_raster(in_raster)
        name = os.path.splitext(os.path.basename(metadata["name"]))[0]

        # Projections
        projection_osr = metadata["projection_osr"]
        projection_latlng = utils_projection._get_default_projection_osr()

        # Bounding boxes
        bbox = metadata["bbox"]

        if latlng:
            footprint_geom = utils_bbox._get_bounds_from_bbox(
                bbox,
                projection_osr,
                wkt=False,
            )
            footprint = utils_bbox._get_vector_from_geom(
                footprint_geom,
                projection_osr=projection_latlng,
                out_path=output_list[idx],
                name=name,
            )
        else:
            footprint_geom = utils_bbox._get_geom_from_bbox(
                bbox,
            )
            footprint = utils_bbox._get_vector_from_geom(
                footprint_geom,
                projection_osr=projection_osr,
                out_path=output_list[idx],
                name=name,
            )

        footprints.append(footprint)

    if input_is_list:
        return footprints

    return footprints[0]
