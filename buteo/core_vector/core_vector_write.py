# Standard library
import os
from typing import Union, Optional, List, Dict, Any, Callable, Tuple

# External
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
    utils_projection,
)



def vector_copy(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[Union[str, List[str]]] = None,
) -> Union[str, List[str]]:
    """Creates a copy of a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    out_path : Optional[str], optional
        The path to the output vector. If None, will create a new file in the same directory as the input vector. Default: None.

    Returns
    -------
    out_path : str
        Path to the output vector.
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(out_path, (str, list, type(None))), "out_path must be a string, list, or None."

    input_is_list = isinstance(vector, list)

    in_paths = utils_io._get_input_paths(vector, "vector")
    out_paths = utils_io._get_output_paths(in_paths, out_path)

    utils_path._delete_if_required_list(out_paths, overwrite=True)

    output = []
    for idx, in_vector in enumerate(in_paths):
        ref = _vector_open(in_vector)

        driver_name = utils_gdal._get_vector_driver_name_from_path(out_paths[idx])
        driver = ogr.GetDriverByName(driver_name)

        destination = driver.CreateDataSource(out_paths[idx])
        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()
            destination.CopyLayer(layer, layer.GetName(), ["OVERWRITE=YES"])

        destination.FlushCache()
        destination = None
        ref = None

        output.append(out_paths[idx])

    if input_is_list:
        return output

    return output[0]


def vector_from_bbox(
    bbox: List[Union[float, int]],
    projection: Union[str, osr.SpatialReference, None] = None,
    out_path: Optional[str] = None,
) -> str:
    """Creates a vector from a bounding box.

    Parameters
    ----------
    bbox : List[float, int]
        The bounding box to create the vector from. Must be in the format [x_min, x_max, y_min, y_max].

    projection : Union[str, osr.SpatialReference, None], optional
        The projection of the bounding box. If None, the default (latlng) projection is used. Default: None.

    out_path : Optional[str], optional
        The path to the output vector. If None, will create a new file in the same directory as the input vector. Default: None.
    """
    assert utils_bbox._check_is_valid_bbox(bbox), "bbox is not a valid bounding box."

    if projection is None:
        proj = utils_projection._get_default_projection_osr()
    else:
        proj = utils_projection.parse_projection(projection)

    if utils_projection._projection_is_latlng(proj):
        bbox = [bbox[2], bbox[3], bbox[0], bbox[1]]

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_bbox.gpkg", add_timestamp=True, add_uuid=True)

    if not utils_path._check_is_valid_output_filepath(out_path):
        raise ValueError(f"Invalid output path: {out_path}")

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    datasource = driver.CreateDataSource(out_path)
    layer = datasource.CreateLayer("bbox", geom_type=ogr.wkbPolygon, srs=proj)
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(utils_bbox._get_geom_from_bbox(bbox))

    layer.CreateFeature(feature)
    layer.SyncToDisk()

    datasource = None
    layer = None
    feature = None

    return out_path


def vector_from_wkt(
    wkt: str,
    projection: Union[str, osr.SpatialReference, int, None] = None,
    out_path: Optional[str] = None,
) -> str:
    """Creates a vector file from a wkt string.

    Parameters
    ----------
    wkt : str
        A string with the Well-known-text representation of a vector.

    projection : Union[str, osr.SpatialReference, int, None], optional
        The projection of the points. If None, the default (latlng) projection is used. Default: None.

    out_path : Optional[str], optional
        The path to the output vector. If None, will create a new file in the same directory as the input vector. Default: None.

    Returns
    -------
    str
        The path to the created vector file.
    """
    assert isinstance(wkt, str), "wkt must be a string."
    assert len(wkt) > 0, "wkt must have at least one character."

    if projection is None:
        proj = utils_projection._get_default_projection_osr()
    else:
        proj = utils_projection.parse_projection(projection)

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_geom.gpkg", add_timestamp=True, add_uuid=True)

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    datasource = driver.CreateDataSource(out_path)

    geom_type = ogr.CreateGeometryFromWkt(wkt).GetGeometryType()

    layer = datasource.CreateLayer("temp_geom", geom_type=geom_type, srs=proj)
    layer_defn = layer.GetLayerDefn()

    feature = ogr.Feature(layer_defn)
    feature.SetGeometry(ogr.CreateGeometryFromWkt(wkt))

    layer.CreateFeature(feature)
    layer.SyncToDisk()

    datasource = None
    layer = None
    feature = None

    return out_path


def vector_from_points(
    points: List[List[Union[float, int]]],
    projection: Union[str, osr.SpatialReference, int, None] = None,
    out_path: Optional[str] = None,
    reverse_xy_order: bool = False,
) -> str:
    """Creates a point vector from a list of points.

    Parameters
    ----------
    points : List[List[float, int]]
        The points to create the vector from. Must be in the format [[x1, y1], [x2, y2], ...].

    projection : Union[str, osr.SpatialReference, int, None], optional
        The projection of the points. If None, the default (latlng) projection is used. Default: None.

    out_path : Optional[str], optional
        The path to the output vector. If None, will create a new file in the same directory as the input vector. Default: None.
    """
    assert isinstance(points, list), "points must be a list."
    assert len(points) > 0, "points must have at least one point."
    for point in points:
        assert isinstance(point, list), "points must be a list of lists."
        assert len(point) == 2, "points must be a list of lists of length 2."

    if projection is None:
        proj = utils_projection._get_default_projection_osr()
    else:
        proj = utils_projection.parse_projection(projection)

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_points.gpkg", add_timestamp=True, add_uuid=True)

    if not utils_path._check_is_valid_output_filepath(out_path):
        raise ValueError(f"Invalid output path: {out_path}")

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    datasource = driver.CreateDataSource(out_path)
    layer = datasource.CreateLayer("points", geom_type=ogr.wkbPoint, srs=proj)
    layer_defn = layer.GetLayerDefn()

    for point in points:
        feature = ogr.Feature(layer_defn)
        geom = ogr.Geometry(ogr.wkbPoint)
        if reverse_xy_order:
            geom.AddPoint(point[1], point[0])
        else:
            geom.AddPoint(point[0], point[1])
        feature.SetGeometry(geom)

        layer.CreateFeature(feature)
        feature = None

    layer.GetExtent()
    layer.SyncToDisk()

    datasource = None
    layer = None

    return out_path


def vector_save_to_disk(
    vector: Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]],
    out_path: Union[str, List[str]],
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """Writes a vector dataset to disk. Can be a single vector or a list of vectors.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]]
        The vector dataset to write to disk.
    out_path : Union[str, List[str]]
        The output path or list of output paths.
    prefix : str, optional
        A prefix to add to the output path. Default: "".
    suffix : str, optional
        A suffix to add to the output path. Default: "".
    add_uuid : bool, optional
        If True, a UUID will be added to the output path. Default: False.
    add_timestamp : bool, optional
        If True, a timestamp will be added to the output path. Default: False.
    overwrite : bool, optional
        If True, the output will be overwritten if it already exists. Default: True.

    Returns
    -------
    Union[str, List[str]]
        The output path or list of output paths.

    Raises
    ------
    ValueError
        If the vector is invalid.
    RuntimeError
        If the driver for the output vector could not be obtained.
    """
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

    for idx, ds in enumerate(in_paths):
        if not utils_gdal._check_is_vector(ds):
            raise ValueError(f"Invalid vector dataset: {ds}")

        driver_name = utils_gdal._get_driver_name_from_path(ds)
        driver = ogr.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"Could not get OGR driver for vector: {driver_name}")

        src_ds = ogr.Open(ds)
        if src_ds is None:
            raise ValueError(f"Unable to open vector dataset: {ds}")

        utils_io._delete_if_required(out_paths[idx], overwrite)
        driver.CopyDataSource(src_ds, out_paths[idx])
        src_ds = None

    return out_paths if input_is_list else out_paths[0]


def vector_set_crs(
    vector: Union[str, ogr.DataSource],
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    out_path: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Sets the projection of a vector dataset.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The input vector dataset.

    projection : Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The projection to set for the vector dataset.

    out_path : Optional[str], optional
        The output path. If None, overwrites the input vector if overwrite is True.

    overwrite : bool, optional
        If True, allows overwriting existing files.

    Returns
    -------
    str
        The path to the vector with the updated projection.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")

    in_vector_path = utils_io._get_input_paths(vector, "vector")
    if out_path is None:
        out_path = in_vector_path if overwrite else utils_path._get_temp_filepath("vector.gpkg")
    else:
        out_path = utils_io._get_output_paths(out_path)

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    srs = utils_projection.parse_projection(projection)
    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"Could not get OGR driver for: {driver_name}")

    src_ds = ogr.Open(in_vector_path)
    if src_ds is None:
        raise ValueError(f"Unable to open vector dataset: {in_vector_path}")

    dst_ds = driver.CreateDataSource(out_path)
    if dst_ds is None:
        raise RuntimeError(f"Could not create output datasource: {out_path}")

    for layer_index in range(src_ds.GetLayerCount()):
        src_layer = src_ds.GetLayerByIndex(layer_index)
        geom_type = src_layer.GetGeomType()
        layer_name = src_layer.GetName()
        dst_layer = dst_ds.CreateLayer(layer_name, srs, geom_type=geom_type)
        if dst_layer is None:
            raise RuntimeError(f"Could not create layer: {layer_name}")

        src_layer_defn = src_layer.GetLayerDefn()
        for i in range(src_layer_defn.GetFieldCount()):
            field_defn = src_layer_defn.GetFieldDefn(i)
            dst_layer.CreateField(field_defn)

        dst_layer_defn = dst_layer.GetLayerDefn()
        src_layer.ResetReading()

        for src_feat in src_layer:
            geom = src_feat.GetGeometryRef()
            dst_feat = ogr.Feature(dst_layer_defn)
            dst_feat.SetFrom(src_feat)
            dst_feat.SetGeometry(geom.Clone())
            dst_layer.CreateFeature(dst_feat)
            dst_feat = None

    src_ds = None
    dst_ds = None

    return out_path

# extract layer
# convert3D_to_2D
# convert2D_to_3D
