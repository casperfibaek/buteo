import sys

from mypy_extensions import TypedDict

sys.path.append("../../")
from typing import Union, List, Optional
from osgeo import ogr

from buteo.gdal_utils import path_to_driver
from buteo.utils import type_check
from buteo.vector.io import (
    open_vector,
    ready_io_vector,
    internal_vector_to_metadata,
    vector_add_index,
)


def internal_dissolve_vector(
    vector: Union[str, ogr.DataSource],
    attribute: Optional[str] = None,
    out_path: str = None,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
) -> str:
    """ Clips a vector to a geometry.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(attribute, [str], "attribute", allow_none=True)
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(add_index, [bool], "add_index")
    type_check(process_layer, [int], "process_layer")

    vector_list, path_list = ready_io_vector(vector, out_path)
    out_name = path_list[0]
    out_format = path_to_driver(out_name)

    driver = ogr.GetDriverByName(out_format)

    ref = open_vector(vector_list[0])
    metadata = internal_vector_to_metadata(ref)

    Layer_info = TypedDict(
        "Layer_info", {"name": str, "geom": str, "fields": List[str]},
    )

    layers: List[Layer_info] = []

    if process_layer == -1:
        for index in range(len(metadata["layers"])):
            layers.append(
                {
                    "name": metadata["layers"][index]["layer_name"],
                    "geom": metadata["layers"][index]["column_geom"],
                    "fields": metadata["layers"][index]["field_names"],
                }
            )
    else:
        layers.append(
            {
                "name": metadata["layers"][process_layer]["layer_name"],
                "geom": metadata["layers"][process_layer]["column_geom"],
                "fields": metadata["layers"][process_layer]["field_names"],
            }
        )

    destination: ogr.DataSource = driver.CreateDataSource(out_name)

    # Check if attribute table is valid
    for index in range(len(metadata["layers"])):
        layer = layers[index]
        if attribute is not None and attribute not in layer["fields"]:
            layer_fields = layer["fields"]
            raise ValueError(
                f"Invalid attribute for layer. Layers has the following fields: {layer_fields}"
            )

        geom_col = layer["geom"]
        name = layer["name"]

        sql = None
        if attribute is None:
            sql = f"SELECT ST_Union({geom_col}) AS geom FROM {name};"
        else:
            sql = f"SELECT {attribute}, ST_Union({geom_col}) AS geom FROM {name} GROUP BY {attribute};"

        result = ref.ExecuteSQL(sql, dialect="SQLITE")
        destination.CopyLayer(result, name, ["OVERWRITE=YES"])

    if add_index:
        vector_add_index(destination)

    destination.FlushCache()

    return out_name


def dissolve_vector(
    vector: Union[List[Union[str, ogr.DataSource]], str, ogr.DataSource],
    attribute: Optional[str] = None,
    out_path: str = None,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
) -> Union[List[str], str]:
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
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(add_index, [bool], "add_index")
    type_check(process_layer, [int], "process_layer")

    raster_list, path_list = ready_io_vector(vector, out_path)

    output = []
    for index, in_vector in enumerate(raster_list):
        output.append(
            internal_dissolve_vector(
                in_vector,
                attribute=attribute,
                out_path=path_list[index],
                overwrite=overwrite,
                add_index=add_index,
                process_layer=process_layer,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"

    vector = folder + "walls_clip.gpkg"
    out_dir = folder + "out/"

    dissolve_vector(vector, attribute="tip", out_path=out_dir + "walls_dissolved.gpkg")
