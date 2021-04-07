import sys; sys.path.append('../../')
from uuid import uuid4
from typing import Union
from osgeo import ogr

from buteo.gdal_utils import (
    is_vector,
    path_to_driver,
    vector_to_reference,
)
from buteo.utils import path_to_ext, type_check
from buteo.vector.io import vector_to_metadata


def merge_vectors(
    vectors: Union[ogr.DataSource, str, list],
    out_path: str=None,
    preserve_fid: bool=True,
    opened: bool=False,
):
    """ Description.
    Args:
    **kwargs:
    Returns:
        Description.
    """
    type_check(vectors, [list], "vector")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(preserve_fid, [bool], "preserve_fid")

    for vector in vectors:
        if not is_vector(vector):
            raise TypeError(f"Invalid vector input: {vector}")

    out_driver = 'GPKG'
    out_format = '.gpkg'
    out_target = f"/vsimem/clipped_{uuid4().int}{out_format}"

    if out_path is not None:
        out_target = out_path
        out_driver = path_to_driver(out_path)
        out_format = path_to_ext(out_path)

    driver = ogr.GetDriverByName(out_driver)

    merged_ds = driver.CreateDataSource(out_target)

    for vector in vectors:
        metadata = vector_to_metadata(vector)
        ref = vector_to_reference(vector)

        layers = metadata["layers"]
        for layer in layers:
            name = layer["layer_name"]
            merged_ds.CopyLayer(ref.GetLayer(name), name, ["OVERWRITE=YES"])

    if opened:
        return merged_ds

    return out_target


# TODO: Memory outputs, flips and mirrors - consider
if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"

    vector = folder + "odense_grid.gpkg"
    clip_geom = folder + "havnen.gpkg"
    out_dir = folder + "out/"

    bob = merge_vectors([vector, clip_geom])

    import pdb; pdb.set_trace()