"""Module for vector validation functions."""

# Standard library
from typing import Union, Optional, List

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base, utils_io, utils_gdal
from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer
from buteo.core_vector.core_vector_validation import check_vector_has_invalid_geometry



def _vector_fix_geometry(
    vector: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    *,
    check_invalid_first: bool = True,
    layer_name_or_id: Optional[Union[str, int]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = False,
) -> str:
    """Attempts to fix invalid geometries in a vector.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    out_path : str, optional
        The output path. Default: None (in-memory is created)
    check_invalid_first : bool, optional
        If True, checks if the input vector has invalid geometries before fixing.
        If True and no invalid geometries are found, or the vector is empty, returns the input vector.
        if False, skips the check. Default: True
    layer_name_or_id : str or int, optional
        The name or index of the layer to fix. Default: None (fixes all layers)
    prefix : str, optional
        Prefix to add to output path. Default: ""
    suffix : str, optional
        Suffix to add to output path. Default: ""
    add_uuid : bool, optional
        If True, adds a UUID to the output path. Default: False
    add_timestamp : bool, optional
        If True, adds a timestamp to the output path. Default: False
    overwrite : bool, optional
        If True, overwrites existing files. Default: False

    Returns
    -------
    str
        The path to the fixed vector
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")
    utils_base._type_check(check_invalid_first, [bool], "check_invalid_first")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if check_invalid_first and not check_vector_has_invalid_geometry(vector, layer_name_or_id=layer_name_or_id):
        if isinstance(vector, ogr.DataSource):
            return vector.GetName()
        return vector

    in_paths = utils_io._get_input_paths(vector, "vector") # type: ignore
    out_paths = utils_io._get_output_paths(
        in_paths, # type: ignore
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    in_path = in_paths[0]
    out_path = out_paths[0]

    ref = _open_vector(in_path)
    layers = _vector_get_layer(ref, layer_name_or_id)

    # Create output data source
    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)
    out_ds = driver.CreateDataSource(out_path)

    # Iterate over layers
    for in_layer in layers:
        # Create output layer
        out_layer = out_ds.CreateLayer(
            in_layer.GetName(),
            srs=in_layer.GetSpatialRef(),
            geom_type=in_layer.GetGeomType()
        )

        # Copy fields from input layer to output layer
        in_layer_defn = in_layer.GetLayerDefn()
        for i in range(in_layer_defn.GetFieldCount()):
            field_defn = in_layer_defn.GetFieldDefn(i)
            out_layer.CreateField(field_defn)

        # Get the output layer's feature definition
        out_layer_defn = out_layer.GetLayerDefn()

        # Iterate over features in input layer
        in_layer.ResetReading()
        for in_feature in in_layer:
            geom = in_feature.GetGeometryRef()
            if geom and not geom.IsValid():
                geom = geom.MakeValid()
                if not geom or not geom.IsValid():
                    continue  # Skip invalid geometries that cannot be fixed

            # Create new feature
            out_feature = ogr.Feature(out_layer_defn)
            out_feature.SetFrom(in_feature)
            out_feature.SetGeometry(geom)
            out_layer.CreateFeature(out_feature)
            out_feature = None

        out_layer = None

    out_ds.FlushCache()

    # Close data sources
    ref = None
    out_ds = None

    return out_path


def vector_fix_geometry(
    vector: Union[str, ogr.DataSource, list[Union[str, ogr.DataSource]]],
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    check_invalid_first: bool = True,
    layer_name_or_id: Optional[Union[str, int]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = False,
) -> Union[str, List[str]]:
    """Attempts to fix invalid geometries in a vector.

    Parameters
    ----------
    vector : str or ogr.DataSource or list[str or ogr.DataSource]
        A path to a vector or a list of paths to vectors or an OGR datasource or a list of OGR datasources
    out_path : str or list[str], optional
        The output path. Default: None (in-memory is created)
    check_invalid_first : bool, optional
        If True, checks if the input vector has invalid geometries before fixing.
        If True and no invalid geometries are found, or the vector is empty, returns the input vector.
        if False, skips the check. Default: True
    layer_name_or_id : str or int, optional
        The name or index of the layer to fix. Default: None (fixes all layers)
    prefix : str, optional
        Prefix to add to output path. Default: ""
    suffix : str, optional
        Suffix to add to output path. Default: ""
    add_uuid : bool, optional
        If True, adds a UUID to the output path. Default: False
    add_timestamp : bool, optional
        If True, adds a timestamp to the output path. Default: False
    overwrite : bool, optional
        If True, overwrites existing files. Default: False

    Returns
    -------
    str or list[str]
        The path(s) to the fixed vector
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [type(None), str, int], "layer_name_or_id")
    utils_base._type_check(check_invalid_first, [bool], "check_invalid_first")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(vector, list)

    in_paths = utils_io._get_input_paths(vector, "vector") # type: ignore
    out_paths = utils_io._get_output_paths(
        in_paths, # type: ignore
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    fixed_geometries = []
    for in_path, out_path in zip(in_paths, out_paths):
        fixed_geometry = _vector_fix_geometry(
            in_path,
            out_path,
            check_invalid_first=check_invalid_first,
            layer_name_or_id=layer_name_or_id,
            prefix=prefix,
            suffix=suffix,
            add_uuid=add_uuid,
            add_timestamp=add_timestamp,
            overwrite=overwrite,
        )
        fixed_geometries.append(fixed_geometry)

    if input_is_list:
        return fixed_geometries

    return fixed_geometries[0]
