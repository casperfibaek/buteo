"""### Clip vectors to other geometries. ###"""

# Standard library
from typing import Union, Optional, List
from warnings import warn

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
from buteo.vector import core_vector
from buteo.raster import core_raster
from buteo.vector.reproject import _vector_reproject



def _vector_clip(
    vector: Union[str, ogr.DataSource],
    clip_geom: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    to_extent: bool = False,
    target_projection: Optional[Union[str, int, ogr.DataSource, gdal.Dataset, osr.SpatialReference]] = None,
    preserve_fid: bool = True,
    promote_to_multi: bool = True,
    overwrite: bool = True,
    verbose: bool = False,
) -> str:
    """Internal."""
    assert isinstance(vector, (str, ogr.DataSource)), "Invalid vector input."
    assert isinstance(clip_geom, (str, ogr.DataSource)), "Invalid clip_geom input."

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_clipped", ext="gpkg")
    else:
        assert utils_path._check_is_valid_output_filepath(out_path, overwrite), "Invalid vector output path."

    gdal.UseExceptions()

    if not verbose:
        gdal.PushErrorHandler("CPLQuietErrorHandler")

    input_path = utils_gdal._get_path_from_dataset(vector)

    options = []
    to_clear = []
    geometry_to_clip = None
    if utils_gdal._check_is_raster(clip_geom):
        geometry_to_clip = core_raster.raster_to_extent(clip_geom)
        to_clear.append(geometry_to_clip)

    if utils_gdal._check_is_vector(clip_geom):
        if to_extent:
            geometry_to_clip = core_vector.vector_to_extent(clip_geom)
            to_clear.append(geometry_to_clip)
        else:
            geometry_to_clip = utils_gdal._get_path_from_dataset(clip_geom)
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")

    clip_vector_reprojected = _vector_reproject(geometry_to_clip, vector)

    if clip_vector_reprojected != geometry_to_clip:
        to_clear.append(clip_vector_reprojected)

    x_min, x_max, y_min, y_max = core_vector._get_basic_metadata_vector(clip_vector_reprojected)["bbox"]

    options.append(f"-spat {x_min} {y_min} {x_max} {y_max}")
    options.append(f'-clipsrc "{clip_vector_reprojected}"')

    if promote_to_multi:
        options.append("-nlt PROMOTE_TO_MULTI")

    if preserve_fid:
        options.append("-preserve_fid")
    else:
        options.append("-unsetFid")

    if target_projection is not None:
        wkt = utils_projection.parse_projection_wkt(target_projection).replace(" ", "\\")

        options.append(f'-t_srs "{wkt}"')

    # dst  # src
    success = gdal.VectorTranslate(
        out_path,
        input_path,
        format=utils_gdal._get_vector_driver_name_from_path(out_path),
        options=" ".join(options),
    )

    utils_gdal.delete_dataset_if_in_memory_list(to_clear)

    if not verbose:
        gdal.PopErrorHandler()

    if success != 0 and success is not None:

        opened = ogr.Open(out_path)

        if opened is None:
            raise RuntimeError("Error while clipping geometry.")

        layer = opened.GetLayer()
        features = layer.GetFeatureCount()

        if features == 0:
            warn("Error while clipping geometry. No features in output.", RuntimeWarning)

        return out_path
    else:
        raise RuntimeError("Error while clipping geometry.")


def vector_clip(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    clip_geom: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    to_extent: bool = False,
    target_projection: Optional[Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]] = None,
    preserve_fid: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
    promote_to_multi: bool = True,
    verbose: bool = False,
) -> Union[str, List[str]]:
    """Clips a vector to a geometry.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector(s) to clip.

    clip_geom : Union[str, ogr.DataSource]
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

    overwrite : bool, optional
        Overwrite output. Default: True

    promote_to_multi : bool, optional
        Promote to multi. Default: True

    verbose : bool, optional
        Print progress/warnings. Default: False

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
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(promote_to_multi, [bool], "promote_to_multi")
    utils_base._type_check(verbose, [bool], "verbose")

    input_is_list = isinstance(vector, list)
    in_paths = utils_io._get_input_paths(vector, "vector")

    out_paths = utils_io._get_output_paths(
        in_paths,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    output = []
    for idx, in_vector in enumerate(in_paths):
        output.append(
            _vector_clip(
                in_vector,
                clip_geom,
                out_path=out_paths[idx],
                to_extent=to_extent,
                target_projection=target_projection,
                preserve_fid=preserve_fid,
                promote_to_multi=promote_to_multi,
                overwrite=overwrite,
                verbose=verbose,
            )
        )

    if input_is_list:
        return output

    return output[0]
