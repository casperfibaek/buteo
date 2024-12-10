"""### Split functions for vector layers. ###

Dissolve vectors by attributes or geometry.
"""

# Standard library
import os
from typing import Optional, Union

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base, utils_path, utils_io, utils_gdal
from buteo.core_vector.core_vector_read import _vector_get_layer, _open_vector



def vector_split_by_feature(
    vector: Union[str, ogr.DataSource],
    out_folder: Optional[str] = None,
    layer_name_or_id: Optional[Union[str, int]] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    extension: Optional[str] = None,
) -> list[str]:
    """
    Splits a vector into multiple files, each containing a single feature.
    
    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        Input vector.
    out_folder : Optional[str], optional
        Output folder, default: None.
    layer_name_or_id : Optional[Union[str, int]], optional
        Layer name or index, default: None.
    prefix : str, optional
        Prefix for output files, default: "".
    suffix : str, optional
        Suffix for output files, default: "".
    extension : Optional[str], optional
        Output file extension, default: None.
        This can also be used for converting the output format.

    Returns
    -------
    list[str]
        Output paths.
    """
    utils_base._type_check(vector, [ogr.DataSource, str], "vector")
    utils_base._type_check(out_folder, [str, None], "out_folder")
    utils_base._type_check(layer_name_or_id, [str, int, None], "layer_name_or_id")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")

    if out_folder is not None and not utils_path._check_dir_exists(out_folder):
        raise FileNotFoundError("out_folder does not exist.")

    input_path = utils_io._get_input_paths(vector, input_type="vector")[0]

    opened = _open_vector(input_path, writeable=False)
    layer = _vector_get_layer(opened, layer_name_or_id)[0]
    out_paths = []

    target_ext = utils_path._get_ext_from_path(input_path)
    if extension is not None:
        target_ext = utils_path._get_ext_from_path(utils_path._get_changed_path_ext(input_path, extension))

    out_path_name = os.path.splitext(os.path.basename(utils_gdal._get_path_from_dataset(input_path)))[0]

    out_driver = ogr.GetDriverByName(utils_gdal._get_vector_driver_name_from_path("example." + target_ext))
    layer_name = layer.GetName()
    layer_crs = layer.GetSpatialRef()
    layer_geom_type = layer.GetGeomType()

    layer.ResetReading()
    feature_count = layer.GetFeatureCount()
    for _ in range(feature_count):
        feature = layer.GetNextFeature()
        feature_id = feature.GetFID()
        out_path_fid_name = f"{out_path_name}_fid-{feature_id}"

        if out_folder is None:
            out_path = utils_path._get_temp_filepath(
                out_path_fid_name,
                ext=target_ext,
                prefix=prefix,
                suffix=suffix,
                add_uuid=True,
                add_timestamp=True,
            )
        else:
            out_path_base = utils_path._parse_path(os.path.join(out_folder, out_path_fid_name) + "." + target_ext)
            out_path = utils_path._get_augmented_path(
                out_path_base,
                prefix=prefix,
                suffix=suffix,
            )

        out_ds = out_driver.CreateDataSource(out_path)
        out_layer = out_ds.CreateLayer(
            layer_name,
            srs=layer_crs,
            geom_type=layer_geom_type,
        )

        # Copy field definitions from input layer to output layer
        layer_defn = layer.GetLayerDefn()
        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            out_layer.CreateField(field_defn)

        # Create a new feature for the output layer
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        out_feature.SetGeometry(feature.GetGeometryRef())
        for i in range(feature.GetFieldCount()):
            out_feature.SetField(i, feature.GetField(i))

        out_layer.CreateFeature(out_feature)
        out_feature = None
        out_layer.SyncToDisk()

        out_ds.FlushCache()
        out_ds, out_layer, feature = (None, None, None)

        out_paths.append(out_path)

    return out_paths


def vector_split_by_attribute(
    vector: Union[str, ogr.DataSource],
    attribute: str,
    out_folder: Optional[str] = None,
    layer_name_or_id: Optional[Union[str, int]] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    extension: Optional[str] = None,
) -> list[str]:
    """
    Splits a vector into multiple files based on an attribute.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        Input vector.
    attribute : str
        The attribute name to split by.
    out_folder : Optional[str], optional
        Output folder, default: None.
    layer_name_or_id : Optional[Union[str, int]], optional
        Layer name or index, default: None.
    prefix : str, optional
        Prefix for output files, default: "".
    suffix : str, optional
        Suffix for output files, default: "".
    extension : Optional[str], optional
        Output file extension, default: None.
        This can also be used for converting the output format.

    Returns
    -------
    list[str]
        Output paths.
    """
    utils_base._type_check(vector, [ogr.DataSource, str], "vector")
    utils_base._type_check(attribute, [str], "attribute")
    utils_base._type_check(out_folder, [str, None], "out_folder")
    utils_base._type_check(layer_name_or_id, [str, int, None], "layer_name_or_id")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")

    if out_folder is not None and not utils_path._check_dir_exists(out_folder):
        raise FileNotFoundError("out_folder does not exist.")

    input_path = utils_io._get_input_paths(vector, input_type="vector")[0]
    datasource = _open_vector(input_path, writeable=False)
    layer = _vector_get_layer(datasource, layer_name_or_id)[0]

    features_by_value = {}
    for feature in layer:
        value = feature.GetField(attribute)
        if value not in features_by_value:
            features_by_value[value] = []
        features_by_value[value].append(feature.Clone())

    target_ext = utils_path._get_ext_from_path(input_path)
    if extension is not None:
        target_ext = utils_path._get_ext_from_path(
            utils_path._get_changed_path_ext(input_path, extension)
        )

    driver_name = utils_gdal._get_vector_driver_name_from_path("example." + target_ext)
    out_driver = ogr.GetDriverByName(driver_name)
    layer_name = layer.GetName()
    layer_crs = layer.GetSpatialRef()
    layer_geom_type = layer.GetGeomType()
    layer_defn = layer.GetLayerDefn()

    out_paths = []
    for value, features in features_by_value.items():
        safe_value = str(value).replace(" ", "_").replace(os.sep, "_")
        out_name = f"{prefix}{layer_name}_{attribute}_{safe_value}{suffix}.{target_ext}"

        if out_folder is None:
            out_path = utils_path._get_temp_filepath(
                out_name, add_uuid=True, add_timestamp=True
            )
        else:
            out_path = os.path.join(out_folder, out_name)

        out_ds = out_driver.CreateDataSource(out_path)
        out_layer = out_ds.CreateLayer(
            layer_name, srs=layer_crs, geom_type=layer_geom_type
        )

        for i in range(layer_defn.GetFieldCount()):
            field_defn = layer_defn.GetFieldDefn(i)
            out_layer.CreateField(field_defn)

        for feature in features:
            out_layer.CreateFeature(feature)
            feature = None

        out_ds.FlushCache()
        out_ds, out_layer = None, None
        out_paths.append(out_path)

    return out_paths
