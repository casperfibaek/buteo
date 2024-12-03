"""### Merge vectors. ###

Merges vectors into a single vector file.
"""

# Standard library
from typing import Union, List, Optional

# External
from osgeo import ogr
from osgeo_utils.ogrmerge import ogrmerge

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_projection,
)
from buteo.vector import core_vector
from buteo.vector.metadata import _vector_to_metadata



def vector_merge_layers(
    vectors: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[str] = None,
) -> str:
    """Merge vectors to a single geopackage.

    Parameters
    ----------
    vectors : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The input vectors.

    out_path : Optional[str], optional
        The output path, default: None.

    Returns
    -------
    str
        The output path.
    """
    utils_base._type_check(vectors, [[str, ogr.DataSource]], "vector")
    utils_base._type_check(out_path, [str, [str], None], "out_path")

    vector_list = utils_base._get_variable_as_list(vectors)

    assert utils_gdal._check_is_vector_list(vector_list), "Invalid input vector list"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector_list[0], suffix="_merged")

    driver = ogr.GetDriverByName(utils_gdal._get_vector_driver_name_from_path(out_path))

    merged_ds = driver.CreateDataSource(out_path)

    for vector in vector_list:
        ref = core_vector._vector_open(vector)
        metadata = _vector_to_metadata(ref)

        for layer in metadata["layers"]:
            name = layer["layer_name"]
            merged_ds.CopyLayer(ref.GetLayer(name), name, ["OVERWRITE=YES"])

    merged_ds.FlushCache()

    return out_path


def vector_merge_features(
    vectors: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[str] = None,
    projection: Optional[str] = None,
    single_layer: bool = True,
    overwrite: bool = True,
    skip_failures: bool = True,
) -> str:
    """Merge vectors to a single geopackage.

    Parameters
    ----------
    vectors : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The input vectors.

    out_path : Optional[str], optional
        The output path, default: None.

    projection : Optional[str], optional
        The projection of the output vector, default: None.
        If None, the projection of the first input vector is used.

    single_layer : bool, optional
        If True, all layers are merged into a single layer, default: True.

    overwrite : bool, optional
        If True, the output file is overwritten if it exists, default: True.

    skip_failures : bool, optional
        If True, failures are skipped, default: True.

    Returns
    -------
    str
        The output path.
    """
    utils_base._type_check(vectors, [[str, ogr.DataSource]], "vector")
    utils_base._type_check(out_path, [str, [str], None], "out_path")

    vector_list = utils_base._get_variable_as_list(vectors)

    assert utils_gdal._check_is_vector_list(vector_list), "Invalid input vector list"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector_list[0], suffix="_merged")

    driver_name = utils_gdal._get_driver_name_from_path(out_path)

    if projection is None:
        target_projection = utils_projection._get_projection_from_vector(vector_list[0]).ExportToWkt()
    else:
        target_projection = utils_projection.parse_projection_wkt(projection)

    success = ogrmerge(
        vector_list,
        out_path,
        single_layer=single_layer,
        driver_name=driver_name,
        t_srs=target_projection,
        overwrite_ds=overwrite,
        skip_failures=skip_failures,
        src_layer_field_name="og_name",
        src_layer_field_content="{LAYER_NAME}"
    )

    return out_path
