"""
Merges vectors into a single vector file.

TODO:
    - Improve documentation

"""

import sys; sys.path.append("../../") # Path: buteo/vector/merge.py
from uuid import uuid4

from osgeo import ogr

from buteo.utils.gdal_utils import path_to_driver_vector
from buteo.utils.core import path_to_ext, type_check
from buteo.vector.io import open_vector, to_vector_list, _vector_to_metadata


def merge_vectors(
    vectors,
    out_path=None,
    *,
    preserve_fid=True,
):
    """Merge vectors to a single geopackage."""
    type_check(vectors, [list], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(preserve_fid, [bool], "preserve_fid")

    vector_list = to_vector_list(vectors)

    out_driver = "GPKG"
    out_format = ".gpkg"
    out_target = f"/vsimem/clipped_{uuid4().int}{out_format}"

    if out_path is not None:
        out_target = out_path
        out_driver = path_to_driver_vector(out_path)
        out_format = path_to_ext(out_path)

    driver = ogr.GetDriverByName(out_driver)

    merged_ds = driver.CreateDataSource(out_target)

    for vector in vector_list:
        ref = open_vector(vector)
        metadata = _vector_to_metadata(ref)

        for layer in metadata["layers"]:
            name = layer["layer_name"]
            merged_ds.CopyLayer(ref.GetLayer(name), name, ["OVERWRITE=YES"])

    merged_ds.FlushCache()

    return out_target
