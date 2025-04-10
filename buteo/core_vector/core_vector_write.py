""" Module for writing vector data to disk or memory. """
# Standard library
from typing import Union, Optional, List, Dict, Any, Tuple # Added Dict, Any, Tuple
import json
import os # Added os import

# External
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    # utils_bbox, # Removed old import - This line is now removed
    utils_path,
    utils_projection,
    utils_translate
)
# Import necessary bbox functions from their new locations
from buteo.bbox.validation import _check_is_valid_bbox
from buteo.bbox.conversion import _get_geom_from_bbox

from buteo.core_vector.core_vector_read import _open_vector, _vector_get_layer



def vector_create_copy(
    vector: Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]],
    out_path: Union[str, List[str]],
    layer_names_or_ids: Optional[Union[str, int, List[Union[str, int]]]] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """Copies a vector dataset to a new location. Can be a single vector or a list of vectors.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]]
        The vector datasets to copy.
    out_path : Union[str, List[str]]
        The output path or list of output paths.
    layer_names_or_ids : Optional[Union[str, int, List[Union[str, int]]]], optional
        The layer names or ids to copy. If None, all layers will be copied. Default is None.
    prefix : str, optional
        A prefix to add to the output path. Default is an empty string.
    suffix : str, optional
        A suffix to add to the output path. Default is an empty string.
    add_uuid : bool, optional
        If True, a UUID will be added to the output path. Default is False.
    add_timestamp : bool, optional
        If True, a timestamp will be added to the output path. Default is False.
    overwrite : bool, optional
        If True, the output will be overwritten if it already exists. Default is True.

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
    utils_base._type_check(vector, [ogr.DataSource, str, list], "vector") # Allow list directly
    utils_base._type_check(out_path, [str, list], "out_path") # Allow list directly
    utils_base._type_check(layer_names_or_ids, [str, int, list, None], "layer_names_or_ids") # Allow list directly
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(vector, list)
    in_paths = utils_io._get_input_paths(vector, "vector")
    out_paths = utils_io._get_output_paths(
        in_paths,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    for idx, in_path in enumerate(in_paths):
        if not utils_gdal._check_is_vector(in_path):
            raise ValueError(f"Invalid vector dataset: {in_path}")

        out_vector_path = out_paths[idx]
        driver_name = utils_gdal._get_vector_driver_name_from_path(out_vector_path)
        driver = ogr.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"Could not get OGR driver for output format: {driver_name}")

        src_ds = _open_vector(in_path)
        if src_ds is None:
            raise ValueError(f"Unable to open vector dataset: {in_path}")

        if layer_names_or_ids is None:
            # Copy entire datasource
            dst_ds = driver.CopyDataSource(src_ds, out_vector_path)
            if dst_ds is None:
                raise RuntimeError(f"Could not copy data source to: {out_vector_path}")
            dst_ds = None
        else:
            # Copy specified layers
            dst_ds = driver.CreateDataSource(out_vector_path)
            if dst_ds is None:
                raise RuntimeError(f"Could not create output datasource: {out_vector_path}")

            target_layers = layer_names_or_ids
            if not isinstance(target_layers, list):
                target_layers = [target_layers]

            for layer_name_or_id in target_layers:
                layers = _vector_get_layer(src_ds, layer_name_or_id)
                for src_layer in layers:
                    layer_name = src_layer.GetName()
                    geom_type = src_layer.GetGeomType()
                    srs = src_layer.GetSpatialRef()

                    dst_layer = dst_ds.CreateLayer(layer_name, srs, geom_type=geom_type)
                    if dst_layer is None:
                        raise RuntimeError(f"Could not create layer: {layer_name}")

                    # Copy fields
                    src_layer_defn = src_layer.GetLayerDefn()
                    for i in range(src_layer_defn.GetFieldCount()):
                        field_defn = src_layer_defn.GetFieldDefn(i)
                        dst_layer.CreateField(field_defn)

                    # Copy features
                    src_layer.ResetReading()
                    for src_feat in src_layer:
                        dst_feat = ogr.Feature(dst_layer.GetLayerDefn())
                        dst_feat.SetFrom(src_feat)
                        dst_layer.CreateFeature(dst_feat)
                        dst_feat = None # Destroy feature

                    dst_layer = None # Destroy layer

            dst_ds.FlushCache()
            dst_ds = None # Destroy datasource

        src_ds = None # Destroy source datasource

    return out_paths if input_is_list else out_paths[0]



def vector_create_empty_copy(
    vector: Union[ogr.DataSource, str],
    out_path: Optional[str] = None,
    layer_names_or_ids: Optional[Union[str, int, List[Union[str, int]]]] = None,
    geom_type: Optional[int] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = True,
) -> str:
    """
    Creates an empty copy of a vector dataset. No features are copied, only the structure.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str]
        The input vector dataset.
    out_path : Optional[str], optional
        The output path. If None, a temporary file will be created. Default is None.
    layer_names_or_ids : Optional[Union[str, int, List[Union[str, int]]]], optional
        The layer names or ids to copy. If None, all layers will be copied. Default is None.
    geom_type : Optional[int], optional
        The geometry type of the output. If None, the same as the input will be used. Default is None.
    prefix : str, optional
        A prefix to add to the output path. Default is an empty string.
    suffix : str, optional
        A suffix to add to the output path. Default is an empty string.
    add_uuid : bool, optional
        If True, a UUID will be added to the output path. Default is False.
    add_timestamp : bool, optional
        If True, a timestamp will be added to the output path. Default is False.
    overwrite : bool, optional
        If True, the output will be overwritten if it already exists. Default is True.

    Returns
    -------
    str
        The output path.
    """
    utils_base._type_check(vector, [ogr.DataSource, str], "vector")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(layer_names_or_ids, [str, int, list, None], "layer_names_or_ids") # Allow list
    utils_base._type_check(geom_type, [int, None], "geom_type")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(overwrite, [bool], "overwrite")

    in_path = utils_io._get_input_paths(vector, "vector")[0]
    out_path_processed = utils_io._get_output_paths( # Renamed variable
        in_path,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
    )[0]

    utils_io._check_overwrite_policy([out_path_processed], overwrite)
    utils_io._delete_if_required(out_path_processed, overwrite)

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path_processed)
    driver = ogr.GetDriverByName(driver_name)

    if driver is None:
        raise RuntimeError(f"Could not get OGR driver for output format: {driver_name}")

    src_ds = _open_vector(in_path, writeable=False)
    if src_ds is None:
        raise ValueError(f"Unable to open vector dataset: {in_path}")

    dst_ds = driver.CreateDataSource(out_path_processed)
    if dst_ds is None:
        raise RuntimeError(f"Could not create output datasource: {out_path_processed}")

    target_layers = layer_names_or_ids
    if target_layers is None:
        layers = _vector_get_layer(src_ds, layer_name_or_id=None)
    elif not isinstance(target_layers, list):
        layers = _vector_get_layer(src_ds, layer_name_or_id=target_layers)
    else: # Is list
        layers = [_vector_get_layer(src_ds, layer_name_or_id=layer_id)[0] for layer_id in target_layers]

    for src_layer in layers:
        layer_name = src_layer.GetName()
        srs = src_layer.GetSpatialRef()

        if geom_type is None:
            target_geom_type = src_layer.GetGeomType()
        else:
            if utils_translate._check_geom_is_geomtype(geom_type):
                target_geom_type = utils_translate._convert_geomtype_to_wkb(geom_type)
            elif utils_translate._check_geom_is_wkbgeom(geom_type):
                target_geom_type = geom_type
            else:
                raise ValueError(f"Invalid geometry type: {geom_type}")

        dst_layer = dst_ds.CreateLayer(layer_name, srs=srs, geom_type=target_geom_type)
        if dst_layer is None:
            raise RuntimeError(f"Could not create layer: {layer_name}")

        # Copy fields
        src_layer_defn = src_layer.GetLayerDefn()
        for i in range(src_layer_defn.GetFieldCount()):
            field_defn = src_layer_defn.GetFieldDefn(i)
            dst_layer.CreateField(field_defn)

        dst_layer.SyncToDisk()
        dst_layer = None
        src_layer = None

    src_ds = None
    dst_ds = None

    return out_path_processed


def vector_create_from_bbox(
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
        The path to the output vector. If None, an in-memory vector will be created. Default: None.

    Returns
    -------
    str
        The path to the created vector file.
    """
    utils_base._type_check(bbox, [list], "bbox")
    utils_base._type_check(projection, [str, osr.SpatialReference, None], "projection")
    utils_base._type_check(out_path, [str, None], "out_path")

    # Use imported function
    if not _check_is_valid_bbox(bbox):
        raise ValueError("bbox is not a valid bounding box.")

    if projection is None:
        proj = utils_projection._get_default_projection_osr()
    else:
        proj = utils_projection.parse_projection(projection)

    # Bbox format for _get_geom_from_bbox is [xmin, xmax, ymin, ymax]
    # No need to reorder based on projection here.
    # if utils_projection._projection_is_latlng(proj):
    #     bbox = [bbox[2], bbox[3], bbox[0], bbox[1]] # This reordering was incorrect

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_bbox.gpkg", add_timestamp=True, add_uuid=True) # Changed default to gpkg

    if not utils_path._check_is_valid_output_filepath(out_path):
        raise ValueError(f"Invalid output path: {out_path}")

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)
    if driver is None: raise RuntimeError(f"Could not get driver: {driver_name}")

    datasource = driver.CreateDataSource(out_path)
    if datasource is None: raise RuntimeError(f"Could not create datasource: {out_path}")

    layer = datasource.CreateLayer("bbox", geom_type=ogr.wkbPolygon, srs=proj)
    if layer is None: raise RuntimeError("Could not create layer.")

    feature = ogr.Feature(layer.GetLayerDefn())
    # Use imported function
    feature.SetGeometry(_get_geom_from_bbox(bbox))

    if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
        raise RuntimeError("Could not create feature.")

    layer.SyncToDisk()

    datasource = None
    layer = None
    feature = None

    return out_path


def vector_create_from_wkt(
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
        The path to the output vector. If None, an in-memory vector will be created. Default: None.

    Returns
    -------
    str
        The path to the created vector file.
    """
    utils_base._type_check(wkt, [str], "wkt")
    utils_base._type_check(projection, [str, osr.SpatialReference, int, None], "projection")
    utils_base._type_check(out_path, [str, None], "out_path")

    if len(wkt) == 0:
        raise ValueError("wkt must have at least one character.")

    if projection is None:
        proj = utils_projection._get_default_projection_osr()
    else:
        proj = utils_projection.parse_projection(projection)

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_geom.gpkg", add_timestamp=True, add_uuid=True) # Changed default to gpkg

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)
    if driver is None: raise RuntimeError(f"Could not get driver: {driver_name}")

    datasource = driver.CreateDataSource(out_path)
    if datasource is None: raise RuntimeError(f"Could not create datasource: {out_path}")

    geom = ogr.CreateGeometryFromWkt(wkt)
    if geom is None: raise ValueError("Could not create geometry from WKT.")
    geom_type = geom.GetGeometryType()

    layer = datasource.CreateLayer("temp_geom", geom_type=geom_type, srs=proj)
    if layer is None: raise RuntimeError("Could not create layer.")

    layer_defn = layer.GetLayerDefn()
    feature = ogr.Feature(layer_defn)
    feature.SetGeometry(geom)

    if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
        raise RuntimeError("Could not create feature.")

    layer.SyncToDisk()

    datasource = None
    layer = None
    feature = None

    return out_path


def vector_create_from_points(
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
        The path to the output vector. If None, an in-memory vector will be created. Default: None.
    """
    utils_base._type_check(points, [list], "points")
    utils_base._type_check(projection, [str, osr.SpatialReference, int, None], "projection")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(reverse_xy_order, [bool], "reverse_xy_order")

    assert len(points) > 0, "points must have at least one point."
    for point in points:
        assert isinstance(point, list), "points must be a list of lists."
        assert len(point) == 2, "points must be a list of lists of length 2."

    if projection is None:
        proj = utils_projection._get_default_projection_osr()
    else:
        proj = utils_projection.parse_projection(projection)

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_points.gpkg", add_timestamp=True, add_uuid=True) # Changed default to gpkg

    if not utils_path._check_is_valid_output_filepath(out_path):
        raise ValueError(f"Invalid output path: {out_path}")

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)
    if driver is None: raise RuntimeError(f"Could not get driver: {driver_name}")

    datasource = driver.CreateDataSource(out_path)
    if datasource is None: raise RuntimeError(f"Could not create datasource: {out_path}")

    layer = datasource.CreateLayer("points", geom_type=ogr.wkbPoint, srs=proj)
    if layer is None: raise RuntimeError("Could not create layer.")

    layer_defn = layer.GetLayerDefn()

    for point in points:
        feature = ogr.Feature(layer_defn)
        geom = ogr.Geometry(ogr.wkbPoint)
        if reverse_xy_order:
            geom.AddPoint(point[1], point[0])
        else:
            geom.AddPoint(point[0], point[1])
        feature.SetGeometry(geom)

        if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
             raise RuntimeError("Could not create feature.")
        feature = None

    layer.GetExtent()
    layer.SyncToDisk()

    datasource = None
    layer = None

    return out_path


def vector_create_from_geojson(
    geojson: Union[str, dict],
    projection: Union[str, osr.SpatialReference, int, None] = None,
    out_path: Optional[str] = None,
) -> str:
    """Creates a vector file from a GeoJSON string or dictionary.

    Parameters
    ----------
    geojson : Union[str, dict]
        The GeoJSON as a string or dictionary.

    projection : Union[str, osr.SpatialReference, int, None], optional
        The projection of the GeoJSON. If None, the default (latlng) projection is used. Default: None.

    out_path : Optional[str], optional
        The path to the output vector. If None, an in-memory vector will be created. Default: None.

    Returns
    -------
    str
        The path to the created vector file.
    """
    utils_base._type_check(geojson, [str, dict], "geojson")
    utils_base._type_check(projection, [str, osr.SpatialReference, int, None], "projection")
    utils_base._type_check(out_path, [str, None], "out_path")

    if isinstance(geojson, dict):
        try:
            geojson = json.dumps(geojson)
        except RuntimeError as e:
            raise ValueError(f"Could not convert GeoJSON to string: {e}") from e

    if projection is None:
        proj = utils_projection._get_default_projection_osr()
    else:
        proj = utils_projection.parse_projection(projection)

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_from_geojson.gpkg", add_timestamp=True, add_uuid=True) # Changed default to gpkg

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)
    if driver is None: raise RuntimeError(f"Could not get driver: {driver_name}")

    datasource = driver.CreateDataSource(out_path)
    if datasource is None: raise RuntimeError(f"Could not create datasource: {out_path}")

    try:
        geojson_ds = ogr.Open(geojson)
    except RuntimeError as e:
        raise ValueError(f"Could not open GeoJSON string: {e}") from e

    if geojson_ds is None:
        raise ValueError("Could not open GeoJSON string.")

    # Copy all layers from the GeoJSON to the output datasource
    for i in range(geojson_ds.GetLayerCount()):
        src_layer = geojson_ds.GetLayer(i)
        layer_name = src_layer.GetName()
        geom_type = src_layer.GetGeomType()
        dst_layer = datasource.CreateLayer(layer_name, proj, geom_type=geom_type)

        if dst_layer is None:
            raise RuntimeError(f"Could not create layer: {layer_name}")

        src_layer_defn = src_layer.GetLayerDefn()

        for field_idx in range(src_layer_defn.GetFieldCount()): # Renamed loop variable
            field_defn = src_layer_defn.GetFieldDefn(field_idx)
            dst_layer.CreateField(field_defn)

        src_layer.ResetReading()

        for src_feat in src_layer:
            dst_feat = ogr.Feature(dst_layer.GetLayerDefn())
            dst_feat.SetFrom(src_feat)
            if dst_layer.CreateFeature(dst_feat) != ogr.OGRERR_NONE:
                 raise RuntimeError("Could not create feature.")
            dst_feat = None

    datasource.SyncToDisk()
    datasource = None

    return out_path


def vector_set_crs(
    vector: Union[str, ogr.DataSource],
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    out_path: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Sets the CRS of a vector dataset by creating a new dataset with the specified projection.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The input vector dataset.

    projection : Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The projection to set for the vector dataset.

    out_path : Optional[str], optional
        The path to the output vector. If None and overwrite is True, the input vector will be overwritten.
        If None and overwrite is False, a temporary file will be created. Default is None.

    overwrite : bool, optional
        If True, allows overwriting existing files. Default is True.

    Returns
    -------
    str
        The path to the vector with the updated projection.

    Raises
    ------
    RuntimeError
        If the driver could not be obtained or the output datasource could not be created.
    ValueError
        If the input vector could not be opened.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(projection,[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")

    in_vector_path = utils_io._get_input_paths(vector, "vector")[0]

    if out_path is None:
        out_path = utils_path._get_temp_filepath("vector_change_crs.gpkg", add_uuid=True, add_timestamp=True) # Changed default to gpkg
    else:
        out_path = utils_io._get_output_paths(in_vector_path, out_path)[0]

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    srs = utils_projection.parse_projection(projection)
    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    if driver is None:
        raise RuntimeError(f"Could not get OGR driver for: {driver_name}")

    src_ds = _open_vector(in_vector_path)

    if src_ds is None:
        raise ValueError(f"Unable to open vector dataset: {in_vector_path}")

    dst_ds = driver.CreateDataSource(out_path)

    if dst_ds is None:
        raise RuntimeError(f"Could not create output datasource: {out_path}")

    src_layers = _vector_get_layer(src_ds, layer_name_or_id=None)

    for src_layer in src_layers:
        geom_type = src_layer.GetGeomType()
        layer_name = src_layer.GetName()
        dst_layer = dst_ds.CreateLayer(layer_name, srs, geom_type=geom_type)

        if dst_layer is None:
            raise RuntimeError(f"Could not create layer: {layer_name}")

        src_layer_defn = src_layer.GetLayerDefn()

        for i in range(src_layer_defn.GetFieldCount()):
            field_defn = src_layer_defn.GetFieldDefn(i)
            dst_layer.CreateField(field_defn)

        src_layer.ResetReading()

        for src_feat in src_layer:
            dst_feat = ogr.Feature(dst_layer.GetLayerDefn())
            dst_feat.SetFrom(src_feat)
            if dst_layer.CreateFeature(dst_feat) != ogr.OGRERR_NONE:
                 raise RuntimeError("Could not create feature.")
            dst_feat = None

    src_ds = None
    dst_ds = None

    return out_path
