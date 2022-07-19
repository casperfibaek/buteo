"""
### Merge vectors. ###

Merges vectors into a single vector file.

TODO:
    * Add things like .vrt files. (already in core_vector...)
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr

# Internal
from buteo.utils import core_utils, gdal_utils
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
    core_utils.type_check(vectors, [[str, ogr.DataSource]], "vector")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(preserve_fid, [bool], "preserve_fid")

    vector_list = core_utils.ensure_list(vectors)

    assert gdal_utils.is_vector_list(vector_list), "Invalid input vector list"

    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            gdal_utils.get_path_from_dataset(vector_list[0]),
            prefix="",
            suffix="_merged",
            add_uuid=True,
        )

    driver = ogr.GetDriverByName(gdal_utils.path_to_driver_vector(out_path))

    merged_ds = driver.CreateDataSource(out_path)

    for vector in vector_list:
        ref = core_vector._open_vector(vector)
        metadata = core_vector._vector_to_metadata(ref)

        for layer in metadata["layers"]:
            name = layer["layer_name"]
            merged_ds.CopyLayer(ref.GetLayer(name), name, ["OVERWRITE=YES"])

    merged_ds.FlushCache()

    return out_path
