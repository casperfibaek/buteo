"""
### Clip vectors to other geometries ###

Clip vector files with other geometries. Can come from rasters or vectors.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_path,
    utils_projection,
)
from buteo.raster import core_raster
from buteo.vector import core_vector
from buteo.vector.reproject import _vector_reproject



def _vector_clip(
    vector: Union[str, ogr.DataSource, gdal.Dataset],
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset],
    out_path: Optional[str] = None,
    to_extent: bool = False,
    target_projection: Optional[Union[str, int, ogr.DataSource, gdal.Dataset, osr.SpatialReference]] = None,
    preserve_fid: bool = True,
    promote_to_multi: bool = True,
) -> str:
    """ Internal. """
    input_path = utils_gdal._get_path_from_dataset(vector)

    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            input_path, add_uuid=True, prefix="", suffix="_clip",
        )

    assert utils_path._check_is_valid_filepath(input_path), "Invalid input path"

    options = []

    clear_memory = False
    geometry_to_clip = None
    if utils_gdal._check_is_vector(clip_geom):
        if to_extent:
            extent = core_vector._vector_to_metadata(clip_geom)["get_bbox_vector"]() # pylint: disable=not-callable
            geometry_to_clip = extent
            clear_memory = True
        else:
            geometry_to_clip = core_vector._vector_open(clip_geom)
    elif utils_gdal._check_is_raster(clip_geom):
        extent = core_raster._get_basic_metadata_raster(clip_geom)["get_bbox_vector"]() # pylint: disable=not-callable
        geometry_to_clip = extent
        clear_memory = True
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")

    clip_vector_path = utils_gdal._get_path_from_dataset(geometry_to_clip)
    clip_vector_reprojected = _vector_reproject(clip_vector_path, vector)

    if clear_memory:
        utils_gdal.delete_dataset_if_in_memory(clip_vector_path)

    x_min, x_max, y_min, y_max = core_vector._vector_to_metadata(clip_vector_reprojected)["extent"]

    options.append(f"-spat {x_min} {y_min} {x_max} {y_max}")

    options.append(f'-clipsrc "{clip_vector_reprojected}"')

    if promote_to_multi:
        options.append("-nlt PROMOTE_TO_MULTI")

    if preserve_fid:
        options.append("-preserve_fid")
    else:
        options.append("-unsetFid")

    if target_projection is not None:
        wkt = utils_projection.parse_projection(target_projection, return_wkt=True).replace(" ", "\\")

        options.append(f'-t_srs "{wkt}"')

    # dst  # src
    success = gdal.VectorTranslate(
        out_path,
        input_path,
        format=utils_gdal._get_vector_driver_name_from_path(out_path),
        options=" ".join(options),
    )

    utils_gdal.delete_dataset_if_in_memory(clip_vector_reprojected)

    if success != 0:
        return out_path
    else:
        raise RuntimeError("Error while clipping geometry.")


def vector_clip(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset],
    out_path: Optional[str] = None,
    to_extent: bool = False,
    target_projection: Optional[Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]] = None,
    preserve_fid: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    allow_lists: bool = True,
    overwrite: bool = True,
    promote_to_multi: bool = True,
) -> Union[str, List[str]]:
    """
    Clips a vector to a geometry.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector(s) to clip.

    clip_geom : Union[str, ogr.DataSource, gdal.Dataset]
        Vector to clip with.

    out_path : Optional[str], optional
        Output path. If None, memory vectors are created. Default: None
    
    to_extent : bool, optional
        Clip to extent. Default: False

    target_projection : Optional[Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]], optional
        Target projection. Default: None
    
    preserve_fid : bool, optional
        Preserve fid. Default: True

    prefix : str, optional
        Prefix to add to the output path. Default: ""

    suffix : str, optional
        Suffix to add to the output path. Default: ""

    add_uuid : bool, optional
        Add a uuid to the output path. Default: False

    allow_lists : bool, optional
        Allow lists as input. Default: True

    overwrite : bool, optional
        Overwrite output. Default: True

    promote_to_multi : bool, optional
        Promote to multi. Default: True

    Returns
    -------
    Union[str, List[str]]
        Path to the clipped vector(s)
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(to_extent, [bool], "to_extent")
    utils_base._type_check(target_projection, [str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int, None], "target_projection")
    utils_base._type_check(preserve_fid, [bool], "preserve_fid")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, (list, tuple)):
        raise ValueError("Lists are not allowed for vector.")

    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Invalid vector in list: {vector_list}"

    path_list = utils_io._get_output_paths(
        vector_list,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _vector_clip(
                in_vector,
                clip_geom,
                out_path=path_list[index],
                to_extent=to_extent,
                target_projection=target_projection,
                preserve_fid=preserve_fid,
                promote_to_multi=promote_to_multi,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
