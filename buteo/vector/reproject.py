"""
Functions to reproject vectors. References can be both vector and raster.

TODO:
    - Improve documentation

"""

import sys; sys.path.append("../../") # Path: buteo/vector/reproject.py
import osgeo

from osgeo import ogr, osr, gdal

from buteo.vector.io import (
    open_vector,
    get_vector_path,
    _vector_to_memory,
    _vector_to_metadata,
    _vector_to_disk,
    ready_io_vector,
)
from buteo.utils.gdal_utils import parse_projection, path_to_driver_vector
from buteo.utils.core import remove_if_overwrite, type_check


def _reproject_vector(
    vector,
    projection,
    out_path=None,
    *,
    copy_if_same=False,
    overwrite=True,
):
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(
        projection,
        [str, int, ogr.DataSource, gdal.Dataset, osr.SpatialReference],
        "projection",
    )
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(copy_if_same, [bool], "copy_if_same")
    type_check(overwrite, [bool], "overwrite")

    vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)
    origin = open_vector(vector_list[0])
    metadata = _vector_to_metadata(origin)
    out_name = path_list[0]

    origin_projection = metadata["projection_osr"]
    target_projection = parse_projection(projection)

    if not isinstance(target_projection, osr.SpatialReference):
        raise Exception("Error ")

    if origin_projection.IsSame(target_projection):
        if copy_if_same:
            if out_path is None:
                return _vector_to_memory(origin)

            return _vector_to_disk(origin, out_name)
        else:
            return get_vector_path(vector)

    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    if int(osgeo.__version__[0]) >= 3:
        origin_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    coord_trans = osr.CoordinateTransformation(origin_projection, target_projection)

    remove_if_overwrite(out_path, overwrite)

    driver = ogr.GetDriverByName(path_to_driver_vector(out_name))
    destination: ogr.DataSource = driver.CreateDataSource(out_name)

    for layer_idx in range(len(metadata["layers"])):
        origin_layer = origin.GetLayerByIndex(layer_idx)
        origin_layer_defn = origin_layer.GetLayerDefn()

        layer_dict = metadata["layers"][layer_idx]
        layer_name = layer_dict["layer_name"]
        layer_geom_type = layer_dict["geom_type_ogr"]

        destination_layer = destination.CreateLayer(
            layer_name, target_projection, layer_geom_type
        )
        destination_layer_defn = destination_layer.GetLayerDefn()

        # Copy field definitions
        origin_layer_defn = origin_layer.GetLayerDefn()
        for i in range(0, origin_layer_defn.GetFieldCount()):
            field_defn = origin_layer_defn.GetFieldDefn(i)
            destination_layer.CreateField(field_defn)

        # Loop through the input features
        for _ in range(origin_layer.GetFeatureCount()):
            feature = origin_layer.GetNextFeature()
            geom = feature.GetGeometryRef()
            geom.Transform(coord_trans)

            new_feature = ogr.Feature(destination_layer_defn)
            new_feature.SetGeometry(geom)

            # Copy field values
            for i in range(0, destination_layer_defn.GetFieldCount()):
                new_feature.SetField(
                    destination_layer_defn.GetFieldDefn(i).GetNameRef(),
                    feature.GetField(i),
                )

            destination_layer.CreateFeature(new_feature)

        destination_layer.ResetReading()
        destination_layer = None

    destination.FlushCache()

    return out_name


def reproject_vector(
    vector,
    projection,
    out_path=None,
    *,
    copy_if_same=False,
    overwrite=True,
):
    """Reprojects a vector given a target projection.

    Args:
        vector (path | vector): The vector to reproject.

        projection (str | int | vector | raster): The projection is infered from
        the input. The input can be: WKT proj, EPSG proj, Proj, or read from a
        vector or raster datasource either from path or in-memory.

    **kwargs:
        out_path (path | None): The destination to save to. If None then
        the output is an in-memory raster.

        overwite (bool): Is it possible to overwrite the out_path if it exists.

    Returns:
        An in-memory vector. If an out_path is given, the output is a string containing
        the path to the newly created vecotr.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(
        projection,
        [str, int, ogr.DataSource, gdal.Dataset, osr.SpatialReference],
        "projection",
    )
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(copy_if_same, [bool], "copy_if_same")
    type_check(overwrite, [bool], "overwrite")

    vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _reproject_vector(
                in_vector,
                projection,
                out_path=path_list[index],
                copy_if_same=copy_if_same,
                overwrite=overwrite,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
