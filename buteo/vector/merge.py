import sys

sys.path.append("../../")
from uuid import uuid4
from typing import Union, List, Optional
from osgeo import ogr

from buteo.gdal_utils import path_to_driver
from buteo.utils import path_to_ext, type_check
from buteo.vector.io import open_vector, to_vector_list, internal_vector_to_metadata


def merge_vectors(
    vectors: List[Union[str, ogr.DataSource]],
    out_path: Optional[str] = None,
    preserve_fid: bool = True,
) -> str:
    """ Merge vectors to a single geopackage.
    """
    type_check(vectors, [list], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(preserve_fid, [bool], "preserve_fid")

    vector_list = to_vector_list(vectors)

    out_driver = "GPKG"
    out_format = ".gpkg"
    out_target = f"/vsimem/clipped_{uuid4().int}{out_format}"

    if out_path is not None:
        out_target = out_path
        out_driver = path_to_driver(out_path)
        out_format = path_to_ext(out_path)

    driver = ogr.GetDriverByName(out_driver)

    merged_ds: ogr.DataSource = driver.CreateDataSource(out_target)

    for vector in vector_list:
        ref = open_vector(vector)
        metadata = internal_vector_to_metadata(ref)

        for layer in metadata["layers"]:
            name = layer["layer_name"]
            merged_ds.CopyLayer(ref.GetLayer(name), name, ["OVERWRITE=YES"])

    merged_ds.FlushCache()

    return out_target


# TODO: Memory outputs, flips and mirrors - consider
if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"

    vector = folder + "odense_grid.gpkg"
    clip_geom = folder + "havnen.gpkg"
    out_dir = folder + "out/"

    bob = merge_vectors([vector, clip_geom])
