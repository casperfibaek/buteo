"""### Reproject vectors. ###

Functions to reproject vectors. References can be both vector and raster.
"""

# Standard library
from typing import Union, Optional, List

# External
from osgeo import gdal, osr, ogr

# Internal
from buteo.utils import (
    utils_io,
    utils_gdal,
    utils_base,
    utils_path,
    utils_projection,
)



def _vector_reproject(
    vector: Union[str, ogr.DataSource],
    projection: Union[str, int, osr.SpatialReference, gdal.Dataset, ogr.DataSource],
    out_path: Optional[str] = None,
    copy_if_same: bool = False,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
):
    """Internal."""
    assert isinstance(vector, (ogr.DataSource, str)), "Invalid vector input"
    assert utils_gdal._check_is_vector(vector), "Invalid vector input"

    # The input is already in the correct projection.
    if not copy_if_same:
        original_projection = utils_projection.parse_projection(vector)

        if utils_projection._check_projections_match(original_projection, projection):
            return utils_gdal._get_path_from_dataset(vector)

    in_path = utils_gdal._get_path_from_dataset(vector)

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, add_uuid=add_uuid, prefix=prefix, suffix=suffix)

    options = []
    wkt = utils_projection.parse_projection(projection, return_wkt=True).replace(" ", "\\")

    options.append(f'-t_srs "{wkt}"')

    success = gdal.VectorTranslate(
        out_path,
        in_path,
        format=utils_gdal._get_vector_driver_name_from_path(out_path),
        options=" ".join(options),
    )

    if success != 0 and success is not None:
        return out_path
    else:
        raise RuntimeError("Error while reprojecting geometry.")


def vector_reproject(
    vector: Union[Union[str, ogr.DataSource], List[Union[str, ogr.DataSource]]],
    projection: Union[str, int, osr.SpatialReference, gdal.Dataset, ogr.DataSource],
    out_path: Optional[str] = None,
    copy_if_same: bool = False,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
):
    """Reprojects a vector given a target projection.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The vector to reproject.

    projection : Union[str, int, osr.SpatialReference, gdal.Dataset, ogr.DataSource]
        The projection is infered from the input. The input can be: WKT proj, EPSG proj, Proj, or read from a vector or raster datasource either from path or in-memory.

    out_path : Optional[str], optional
        The destination to save to. If None then the output is an in-memory raster., default: None

    copy_if_same : bool, optional
        Create a copy, even if the projections are the same., default: False

    prefix : str, optional
        The prefix to add to the output path., default: ""

    suffix : str, optional
        The suffix to add to the output path., default: ""

    add_uuid : bool, optional
        Add a uuid to the output path., default: False

    overwrite : bool, optional
        Is it possible to overwrite the out_path if it exists., default: True

    Returns
    -------
    Union[str, List[str]]
        An in-memory vector. If an out_path is given, the output is a string containing the path to the newly created vecotr.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(projection, [str, int, ogr.DataSource, gdal.Dataset, osr.SpatialReference], "projection")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(copy_if_same, [bool], "copy_if_same")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(vector, list)

    input_data = utils_io._get_input_paths(vector, "vector")
    output_data = utils_io._get_output_paths(
        input_data,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    utils_path._delete_if_required_list(output_data, overwrite)

    output = []
    for idx, in_vector in enumerate(input_data):
        output.append(
            _vector_reproject(
                in_vector,
                projection,
                out_path=output_data[idx],
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                copy_if_same=copy_if_same,
            )
        )

    if input_is_list:
        return output

    return output[0]
