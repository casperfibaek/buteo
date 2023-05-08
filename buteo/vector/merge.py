"""
### Merge vectors. ###

Merges vectors into a single vector file.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, List, Optional

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base, utils_gdal, utils_path
from buteo.vector import core_vector



def vector_merge(
    vectors: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[str] = None,
    preserve_fid: bool = True,
) -> str:
    """
    Merge vectors to a single geopackage.

    Parameters
    ----------
    vectors : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The input vectors.

    out_path : Optional[str], optional
        The output path, default: None.

    preserve_fid : bool, optional
        If True, the FIDs will be preserved, default: True.

    Returns
    -------
    str
        The output path.
    """
    utils_base._type_check(vectors, [[str, ogr.DataSource]], "vector")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(preserve_fid, [bool], "preserve_fid")

    vector_list = utils_base._get_variable_as_list(vectors)

    assert utils_gdal._check_is_vector_list(vector_list), "Invalid input vector list"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector_list[0], suffix="_merged")

    driver = ogr.GetDriverByName(utils_gdal._get_vector_driver_name_from_path(out_path))

    merged_ds = driver.CreateDataSource(out_path)

    for vector in vector_list:
        ref = core_vector._vector_open(vector)
        metadata = core_vector._vector_to_metadata(ref)

        for layer in metadata["layers"]:
            name = layer["layer_name"]
            merged_ds.CopyLayer(ref.GetLayer(name), name, ["OVERWRITE=YES"])

    merged_ds.FlushCache()

    return out_path
