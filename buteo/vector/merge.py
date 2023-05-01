"""
### Merge vectors. ###

Merges vectors into a single vector file.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base, utils_gdal
from buteo.vector import core_vector



def merge_vectors(
    vectors,
    out_path=None,
    *,
    preserve_fid=True,
):
    """
    Merge vectors to a single geopackage.

    ## Args:
    `vectors` (_list_): List of vectors to merge.

    ## Kwargs:
    `out_path` (_str_): Path to output vector. (Default: **None**) </br>
    `preserve_fid` (_bool_): Preserve FIDs? (Default: **True**)

    ## Returns:
    (_str_): Path to output vector.
    """
    utils_base.type_check(vectors, [[str, ogr.DataSource]], "vector")
    utils_base.type_check(out_path, [str, [str], None], "out_path")
    utils_base.type_check(preserve_fid, [bool], "preserve_fid")

    vector_list = utils_base._get_variable_as_list(vectors)

    assert utils_gdal._check_is_vector_list(vector_list), "Invalid input vector list"

    if out_path is None:
        out_path = utils_gdal.create_memory_path(
            utils_gdal._get_path_from_dataset(vector_list[0]),
            prefix="",
            suffix="_merged",
            add_uuid=True,
        )

    driver = ogr.GetDriverByName(utils_gdal._get_vector_driver_from_path(out_path))

    merged_ds = driver.CreateDataSource(out_path)

    for vector in vector_list:
        ref = core_vector._open_vector(vector)
        metadata = core_vector._vector_to_metadata(ref)

        for layer in metadata["layers"]:
            name = layer["layer_name"]
            merged_ds.CopyLayer(ref.GetLayer(name), name, ["OVERWRITE=YES"])

    merged_ds.FlushCache()

    return out_path
