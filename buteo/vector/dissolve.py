import sys; sys.path.append('../../')
from uuid import uuid4
from typing import Union
from osgeo import ogr

from buteo.gdal_utils import (
    is_vector,
    path_to_driver,
    vector_to_reference,
)
from buteo.utils import type_check
from buteo.vector.io import vector_to_metadata, vector_add_index


def dissolve_vector(
    vector: Union[ogr.DataSource, str, list],
    attribute: Union[list, str, None]=None,
    single_parts: bool=False,
    out_path: str=None,
    overwrite: bool=True,
    opened: bool=False,
    vector_idx: int=-1,
) -> str:
    """ Clips a vector to a geometry.
    Args:
        vector (list of vectors | path | vector): The vectors(s) to clip.

        clip_geom (list of geom | path | vector | rasters): The geometry to use
        for the clipping

    **kwargs:


    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    type_check(vector, [ogr.DataSource, str, list], "vector")
    type_check(attribute, [str], "attribute", allow_none=True)
    type_check(single_parts, [bool], "single_parts")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(opened, [bool], "opened")
    type_check(vector_idx, [int], "vector_idx")

    out_format = 'GPKG'
    out_target = f"/vsimem/union_{uuid4().int}.gpkg"

    if out_path is not None:
        out_target = out_path
        out_format = path_to_driver(out_path)

    if not is_vector(vector):
        raise TypeError(f"Invalid vector input: {vector}.")

    driver = ogr.GetDriverByName(out_format)

    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(vector, latlng_and_footprint=False)

    layers = []

    if vector_idx == -1:
        for layer in metadata["layers"]:
            layers.append({
                "name": layer["layer_name"],
                "geom": layer["geom_column"],
                "fields": layer["field_names"],
            })
    else:
        layers.append({
            "name": metadata["layers"][vector_idx]["layer_name"],
            "geom": metadata["layers"][vector_idx]["geom_column"],
            "fields": metadata["layers"][vector_idx]["field_names"]
        })

    destination = driver.CreateDataSource(out_target)

    # Check if attribute table is valid
    for layer in layers:
        if attribute is not None and attribute not in layer["fields"]:
            layer_fields = layer["fields"]
            raise ValueError(f"Invalid attribute for layer. Layers has the following fields: {layer_fields}")
        
        geom_col = layer["geom"]
        name = layer["name"]
    
        sql = None
        if attribute is None:
            sql = f"SELECT ST_Union({geom_col}) AS geom FROM {name};"
        else:
            sql = f"SELECT {attribute}, ST_Union({geom_col}) AS geom FROM {name} GROUP BY {attribute};"

        result = ref.ExecuteSQL(sql, dialect="SQLITE")
        destination.CopyLayer(result, name, ["OVERWRITE=YES"])
    
    vector_add_index(destination)

    if opened:
        return destination
    
    return out_path


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"

    vector = folder + "walls_clip.gpkg"
    out_dir = folder + "out/"

    dissolve_vector(vector, attribute="tip", out_path=out_dir + "walls_dissolved.gpkg")
