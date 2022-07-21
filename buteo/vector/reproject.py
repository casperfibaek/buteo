"""
### Reproject vectors. ###

Functions to reproject vectors. References can be both vector and raster.
"""

# Standard library
import sys; sys.path.append("../../")
import os

# External
import osgeo
from osgeo import gdal, osr, ogr

# Internal
from buteo.utils import gdal_utils, core_utils
from buteo.vector import core_vector



def _reproject_vector(
    vector,
    projection,
    out_path=None,
    *,
    copy_if_same=False,
    overwrite=True,
    prefix="",
    suffix="",
):
    """ Internal. """
    assert isinstance(vector, (ogr.DataSource, str)), "Invalid vector input"
    assert gdal_utils.is_vector(vector), "Invalid vector input"

    origin = core_vector._open_vector(vector)
    out_name = out_path
    if out_path is None:
        out_name = gdal_utils.create_memory_path(
            gdal_utils.get_path_from_dataset(vector),
            prefix=prefix,
            suffix=suffix,
            add_uuid=True,
        )

    metadata = core_vector._vector_to_metadata(origin)
    namesplit = os.path.splitext(os.path.basename(out_name))
    out_name = os.path.join(os.path.dirname(out_name), prefix + namesplit[0] + suffix + namesplit[1])

    origin_projection = metadata["projection_osr"]
    target_projection = gdal_utils.parse_projection(projection)

    if not isinstance(target_projection, osr.SpatialReference):
        raise Exception("Error ")

    if origin_projection.IsSame(target_projection):
        if copy_if_same:
            if out_path is None:
                return gdal_utils.save_dataset_to_memory(origin)

            return gdal_utils.save_dataset_to_disk(origin, out_name)
        else:
            return gdal_utils.get_path_from_dataset(vector)

    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    if int(osgeo.__version__[0]) >= 3:
        origin_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    coord_trans = osr.CoordinateTransformation(origin_projection, target_projection)

    core_utils.remove_if_required(out_path, overwrite)

    driver = ogr.GetDriverByName(gdal_utils.path_to_driver_vector(out_name))
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
    prefix="",
    suffix="",
    add_uuid=False,
    allow_lists=True,
    overwrite=True,
):
    """Reprojects a vector given a target projection.

    Args:
        vector (path/vector): The vector to reproject.

        projection (str/int/vector/raster): The projection is infered from
        the input. The input can be: WKT proj, EPSG proj, Proj, or read from a
        vector or raster datasource either from path or in-memory.

    **kwargs:
        out_path (path/None): The destination to save to. If None then
        the output is an in-memory raster.

        overwite (bool): Is it possible to overwrite the out_path if it exists.

    Returns:
        An in-memory vector. If an out_path is given, the output is a string containing
        the path to the newly created vecotr.
    """
    core_utils.type_check(vector, [str, ogr.DataSource], "vector")
    core_utils.type_check(projection, [str, int, ogr.DataSource, gdal.Dataset, osr.SpatialReference], "projection")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(copy_if_same, [bool], "copy_if_same")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(allow_lists, [bool], "allow_lists")
    core_utils.type_check(overwrite, [bool], "overwrite")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Lists are not allowed when allow_lists is False.")

    vector_list = core_utils.ensure_list(vector)

    assert gdal_utils.is_vector_list(vector_list), f"Invalid input vector: {vector_list}"

    path_list = gdal_utils.create_output_path_list(
        vector_list,
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    assert core_utils.is_valid_output_paths(path_list, overwrite=overwrite), "Invalid output path generated."

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _reproject_vector(
                in_vector,
                projection,
                out_path=path_list[index],
                copy_if_same=copy_if_same,
                overwrite=overwrite,
                prefix=prefix,
                suffix=suffix,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
