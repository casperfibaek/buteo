"""### Basic IO functions for working with Vectprs ###

The basic module for interacting with vector data
"""

# TODO: More attribute functions
# TODO: Repair vector functions
# TODO: Sanity checks
# TODO: Joins (by attribute, location, summary, etc..)
# TODO: Union, Erase, ..
# TODO: Sampling functions
# TODO: Vector intersects, etc..

# Standard library
import sys; sys.path.append("../../")
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


def _vector_open(
    vector: Union[str, ogr.DataSource, gdal.Dataset],
    writeable: bool = True,
    allow_raster: bool = True,
) -> ogr.DataSource:
    """Internal."""
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a path or a DataSource"

    opened = None
    if utils_gdal._check_is_vector(vector):
        if isinstance(vector, str):
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            opened = ogr.Open(vector, gdal.GF_Write) if writeable else ogr.Open(vector, gdal.GF_Read)
            gdal.PopErrorHandler()
        elif isinstance(vector, ogr.DataSource):
            opened = vector
        else:
            raise RuntimeError(f"Could not read input vector: {vector}")

    elif allow_raster and utils_gdal._check_is_raster(vector):
        if isinstance(vector, str):
            gdal.PushErrorHandler("CPLQuietErrorHandler")
            opened = gdal.Open(vector, gdal.GF_Write) if writeable else gdal.Open(vector, gdal.GF_Read)
            gdal.PopErrorHandler()
        elif isinstance(vector, gdal.Dataset):
            opened = vector
        else:
            raise RuntimeError(f"Could not read input vector: {vector}")

        bbox = utils_bbox._get_bbox_from_geotransform(
            opened.GetGeoTransform(),
            opened.RasterXSize,
            opened.RasterYSize,
        )

        projection_wkt = opened.GetProjection()
        projection_osr = osr.SpatialReference()
        projection_osr.ImportFromWkt(projection_wkt)

        vector_bbox = utils_bbox._get_vector_from_bbox(bbox, projection_osr)
        opened = ogr.Open(vector_bbox, gdal.GF_Write) if writeable else ogr.Open(vector_bbox, gdal.GF_Read)

    else:
        raise RuntimeError(f"Could not read input vector: {vector}")

    if opened is None:
        raise RuntimeError(f"Could not read input vector: {vector}")

    return opened


def vector_open(
    vector: Union[str, ogr.DataSource, gdal.Dataset, List[Union[str, ogr.DataSource, gdal.Dataset]]],
    writeable: bool = True,
    allow_raster: bool = True,
) -> Union[ogr.DataSource, List[ogr.DataSource]]:
    """Opens a vector to an ogr.Datasource class.

    Parameters
    ----------
    vector : str, ogr.DataSource, gdal.Dataset, list[str, ogr.DataSource, gdal.Dataset]
        The vector to open. If a raster is supplied the bounding box is opened as a vector.

    writeable : bool, optional
        If True, the vector is opened in write mode. Default: True

    allow_raster : bool, optional
        If True, a raster will be opened as a vector bounding box. Default: True

    allow_lists : bool, optional
        If True, the input can be a list of vectors. Default: True

    Returns
    -------
    ogr.DataSource, List[ogr.DataSource]
        The opened vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, gdal.Dataset, [str, ogr.DataSource, gdal.Dataset]], "vector")
    utils_base._type_check(writeable, [bool], "writeable")
    utils_base._type_check(allow_raster, [bool], "allow_raster")

    vectors = utils_base._get_variable_as_list(vector)

    output = []
    for element in vectors:
        output.append(_vector_open(element, writeable=writeable, allow_raster=allow_raster))

    if isinstance(vector, list):
        return output

    return output[0]



def _get_basic_metadata_vector(
    vector: Union[str, ogr.DataSource],
) -> Dict[str, Any]:
    """Get basic metadata from a vector.

    Parameters
    ----------
    raster : str or ogr.DataSource
        The raster to get the metadata from.

    Returns
    -------
    Dict[str]
        A dictionary with the metadata.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")

    datasource = _vector_open(vector)

    # Paths
    path = utils_path._get_unix_path(datasource.GetDescription())
    in_memory = utils_path._check_is_valid_mem_filepath(path)

    layers = []
    vector_bbox = None
    layer_count = datasource.GetLayerCount()
    total_features = 0
    for layer_index in range(layer_count):
        layer = datasource.GetLayerByIndex(layer_index)

        x_min, x_max, y_min, y_max = layer.GetExtent()
        layer_bbox = [x_min, x_max, y_min, y_max]
        vector_bbox = layer_bbox

        layer_name = layer.GetName()

        column_fid = layer.GetFIDColumn()
        column_geom = layer.GetGeometryColumn()

        if column_geom == "":
            column_geom = "geom"

        feature_count = layer.GetFeatureCount()

        projection_osr = layer.GetSpatialRef()
        projection_wkt = layer.GetSpatialRef().ExportToWkt()

        if layer_index > 0:
            v_x_min, v_x_max, v_y_min, v_y_max = vector_bbox
            if x_min < v_x_min:
                vector_bbox[0] = x_min
            if x_max > v_x_max:
                vector_bbox[1] = x_max
            if y_min < v_y_min:
                vector_bbox[2] = y_min
            if y_max > v_y_max:
                vector_bbox[3] = y_max

        layer_defn = layer.GetLayerDefn()

        geom_type_ogr = layer_defn.GetGeomType()
        geom_type = ogr.GeometryTypeToName(layer_defn.GetGeomType())

        field_count = layer_defn.GetFieldCount()
        field_names = []
        field_types = []
        field_types_ogr = []

        for field_index in range(field_count):
            field_defn = layer_defn.GetFieldDefn(field_index)
            field_names.append(field_defn.GetName())
            field_type = field_defn.GetType()
            field_types_ogr.append(field_type)
            field_types.append(field_defn.GetFieldTypeName(field_type))

        layer_dict = {
            "layer_name": layer_name,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "column_fid": column_fid,
            "column_geom": column_geom,
            "feature_count": feature_count,
            "projection_wkt": projection_wkt,
            "projection_osr": projection_osr,
            "geom_type": geom_type,
            "geom_type_ogr": geom_type_ogr,
            "field_count": field_count,
            "field_names": field_names,
            "field_types": field_types,
            "field_types_ogr": field_types_ogr,
            "bbox": layer_bbox,
        }

        layers.append(layer_dict)
        total_features += feature_count

    x_min, x_max, y_min, y_max = vector_bbox
    area = (x_max - x_min) * (y_max - y_min)

    bbox_latlng = utils_projection.reproject_bbox(vector_bbox, projection_osr, utils_projection._get_default_projection_osr())
    bounds_latlng = utils_bbox._get_bounds_from_bbox(vector_bbox, projection_osr, wkt=False)
    bounds_area = bounds_latlng.GetArea()
    x_min, x_max, y_min, y_max = vector_bbox
    area = (x_max - x_min) * (y_max - y_min)
    bounds_wkt = bounds_latlng.ExportToWkt()
    bounds_vector_raw = utils_bbox._get_geom_from_bbox(vector_bbox)
    bounds_vector = bounds_vector_raw.ExportToWkt()

    _centroid = bounds_vector_raw.Centroid()
    centroid = (_centroid.GetX(), _centroid.GetY())
    if projection_osr.IsGeographic():
        centroid_latlng = (centroid[1], centroid[0])
    else:
        _centroid_latlng = bounds_latlng.Centroid()
        centroid_latlng = (_centroid_latlng.GetY(), _centroid_latlng.GetX())


    metadata = {
        "path": path,
        "basename": os.path.basename(path),
        "name": os.path.splitext(os.path.basename(path))[0],
        "folder": os.path.dirname(path),
        "ext": os.path.splitext(path)[1],
        "in_memory": in_memory,
        "driver": datasource.GetDriver().GetName(),
        "projection_osr": projection_osr,
        "projection_wkt": projection_wkt,
        "feature_count": total_features,
        "bbox": vector_bbox,
        "bbox_gdal": utils_bbox._get_gdal_bbox_from_ogr_bbox(vector_bbox),
        "bbox_latlng": bbox_latlng,
        "bounds_latlng": bounds_wkt,
        "bounds_vector": bounds_vector,
        "centroid": centroid,
        "centroid_latlng": centroid_latlng,
        "x_min": vector_bbox[0],
        "x_max": vector_bbox[1],
        "y_min": vector_bbox[2],
        "y_max": vector_bbox[3],
        "area_bounds": bounds_area,
        "area": area,
        "layer_count": layer_count,
        "layers": layers,
    }

    layer = None
    datasource = None
    return metadata


def _vector_filter(
    vector: Union[ogr.DataSource, str],
    filter_function: Callable,
    out_path: Optional[str] = None,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = True,
) -> str:
    """Internal."""
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(filter_function, (type(lambda: True))), "filter_function must be a function."

    metadata = _get_basic_metadata_vector(vector)

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, prefix=prefix, suffix=suffix)

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), f"out_path is not a valid output path. {out_path}"

    projection = metadata["projection_osr"]

    driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
    driver = ogr.GetDriverByName(driver_name)

    datasource_destination = driver.CreateDataSource(out_path)
    datasource_original = _vector_open(vector)

    for i in range(metadata["layer_count"]):
        if process_layer != -1 and i != process_layer:
            continue

        features = metadata["layers"][i]["feature_count"]
        field_names = metadata["layers"][i]["field_names"]

        geom_type = metadata["layers"][i]["geom_type_ogr"]

        layer_original = datasource_original.GetLayer(i)
        layer_destination = datasource_destination.CreateLayer(layer_original.GetName(), projection, geom_type)

        for feature in range(features):
            feature = layer_original.GetNextFeature()

            field_values = []
            for field in field_names:
                field_values.append(feature.GetField(field))

            field_dict = {}
            for j, value in enumerate(field_values):
                field_dict[field_names[j]] = value

            if filter_function(field_dict):
                layer_destination.CreateFeature(feature.Clone())

        layer_destination.SyncToDisk()
        layer_destination.ResetReading()
        layer_destination = None

    return out_path


def vector_filter(
    vector: Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]],
    filter_function: Callable,
    out_path: Optional[Union[str, List[str]]] = None,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """Filters a vector using its attribute table and a function.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]]
        A vector layer(s) or path(s) to a vector file.

    filter_function : Callable
        A function that takes a dictionary of attributes and returns a boolean.

    out_path : Optional[str], optional
        Path to the output vector file. If None, a memory vector will be created. default: None

    process_layer : int, optional
        The index of the layer to process. If -1, all layers will be processed. default: -1

    prefix : str, optional
        A prefix to add to the output vector file. default: ""

    suffix : str, optional
        A suffix to add to the output vector file. default: ""

    add_uuid : bool, optional
        If True, a uuid will be added to the output path. default: False

    overwrite : bool, optional
        If True, the output file will be overwritten if it already exists. default: True

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector file(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(filter_function, [type(lambda: True)], "filter_function")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(vector, list)
    input_list = utils_io._get_input_paths(vector, "vector")
    output_list = utils_io._get_output_paths(
        input_list,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
        change_ext="gpkg",
    )

    utils_path._delete_if_required_list(output_list, overwrite)

    output = []
    for idx, in_vector in enumerate(input_list):
        output.append(_vector_filter(
            in_vector,
            filter_function,
            out_path=output_list[idx],
            process_layer=process_layer,
            prefix=prefix,
            suffix=suffix,
            overwrite=overwrite,
        ))

    if input_is_list:
        return output

    return output[0]


def vector_add_index(
    vector: Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]],
) -> Union[str, List[str]]:
    """Adds a spatial index to the vector in place, if it doesn't have one.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]]
        A vector layer(s) or path(s) to a vector file.

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the input vector file(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")

    input_is_list = isinstance(vector, list)
    input_list = utils_io._get_input_paths(vector, "vector")

    try:
        for in_vector in input_list:
            metadata = _get_basic_metadata_vector(in_vector)
            ref = _vector_open(in_vector)

            for layer in metadata["layers"]:
                name = layer["layer_name"]
                geom = layer["column_geom"]

                sql = f"SELECT CreateSpatialIndex('{name}', '{geom}') WHERE NOT EXISTS (SELECT HasSpatialIndex('{name}', '{geom}'));"
                ref.ExecuteSQL(sql, dialect="SQLITE")
    except:
        raise RuntimeError(f"Error while creating indices for {vector}") from None

    if input_is_list:
        return input_list

    return input_list[0]


def _vector_get_attribute_table(
    vector: Union[str, ogr.DataSource],
    process_layer: int = 0,
    include_fids: bool = False,
    include_geometry: bool = False,
    include_attributes: bool = True,
    return_header: bool = True,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any]]]:
    """Get the attribute table(s) of a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    process_layer : int, optional
        The layer to process. Default: 0 (first layer).

    include_fids : bool, optional
        If True, will include the FID column. Default: False.

    include_geometry : bool, optional
        If True, will include the geometry column. Default: False.

    include_attributes : bool, optional
        If True, will include the attribute columns. Default: True.

    return_header : bool, optional
        If True, will return the header. Default: True.

    Returns
    -------
    attribute_table : Dict[str, Any]
        The attribute table(s) of the vector(s).
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(process_layer, int), "process_layer must be an integer."
    assert isinstance(include_fids, bool), "include_fids must be a boolean."
    assert isinstance(include_geometry, bool), "include_geometry must be a boolean."
    assert isinstance(include_attributes, bool), "include_attributes must be a boolean."
    assert isinstance(return_header, bool), "return_header must be a boolean."

    ref = _vector_open(vector)
    metadata = _get_basic_metadata_vector(ref)

    attribute_table_header = metadata["layers"][process_layer]["field_names"]
    attribute_table = []

    layer = ref.GetLayer(process_layer)
    layer.ResetReading()
    while True:
        feature = layer.GetNextFeature()

        if feature is None:
            break

        attributes = [feature.GetFID()]

        for field_name in attribute_table_header:
            attributes.append(feature.GetField(field_name))

        if include_geometry:
            geom_defn = feature.GetGeometryRef()
            attributes.append(geom_defn.ExportToIsoWkt())

        attribute_table.append(attributes)

    attribute_table_header.insert(0, "fid")

    if include_geometry:
        attribute_table_header.append("geom")

    ref = None
    layer = None

    if return_header:
        return attribute_table_header, attribute_table

    return attribute_table


def vector_get_attribute_table(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    process_layer: int = 0,
    include_fids: bool = False,
    include_geometry: bool = False,
    include_attributes: bool = True,
    return_header: bool = True,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the attribute table(s) of a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    process_layer : int, optional
        The layer to process. Default: 0 (first layer).

    include_fids : bool, optional
        If True, will include the FID column. Default: False.

    include_geometry : bool, optional
        If True, will include the geometry column. Default: False.

    include_attributes : bool, optional
        If True, will include the attribute columns. Default: True.

    return_header : bool, optional
        If True, will return the header. Default: True.

    Returns
    -------
    attribute_table : Dict[str, Any]
        The attribute table(s) of the vector(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(include_fids, [bool], "include_fids")
    utils_base._type_check(include_geometry, [bool], "include_geometry")
    utils_base._type_check(include_attributes, [bool], "include_attributes")
    utils_base._type_check(return_header, [bool], "return_header")

    input_is_list = isinstance(vector, list)
    input_list = utils_io._get_input_paths(vector, "vector")

    output_attributes = []
    output_headers = []
    for in_vector in input_list:
        if return_header:
            header, table = _vector_get_attribute_table(
                in_vector,
                process_layer=process_layer,
                include_fids=include_fids,
                include_geometry=include_geometry,
                include_attributes=include_attributes,
                return_header=True,
            )
            output_headers.append(header)
            output_attributes.append(table)

        else:
            output_attributes.append(_vector_get_attribute_table(
                in_vector,
                process_layer=process_layer,
                include_fids=include_fids,
                include_geometry=include_geometry,
                include_attributes=include_attributes,
                return_header=False,
            ))

    if return_header:
        if input_is_list:
            return output_headers, output_attributes

        return output_headers[0], output_attributes[0]
    
    if input_is_list:
        return output_attributes

    return output_attributes[0]


def vector_filter_layer(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    layer_name_or_idx: Union[str, int],
    out_path: Optional[Union[str, List[str]]] = None,
    prefix: str = "",
    suffix: str = "_layer",
    add_uuid: bool = False,
    overwrite: bool = True,
):
    """Filters a multi-layer vector source to a single layer.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    layer_name_or_idx : Union[str, int]
        The name or index of the layer to filter.

    out_path : Optional[str], optional
        The path to the output vector. If None, will create a new file in the same directory as the input vector. Default: None.

    prefix : str, optional
        Prefix to add to the output vector. Default: "".

    suffix : str, optional
        Suffix to add to the output vector. Default: "_layer".

    add_uuid : bool, optional
        If True, will add a UUID to the output vector. Default: False.

    overwrite : bool, optional
        If True, will overwrite the output vector if it already exists. Default: True.

    Returns
    -------
    out_path : str
        Path to the output vector.
    """
    input_is_list = isinstance(vector, list)

    input_list = utils_io._get_input_paths(vector, "vector")
    output_list = utils_io._get_output_paths(
        input_list,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
        change_ext="gpkg",
    )

    utils_path._delete_if_required_list(output_list, overwrite)

    output = []
    for idx, in_vector in enumerate(input_list):
        ref = _vector_open(in_vector)
        out_path = output_list[idx]

        if isinstance(layer_name_or_idx, int):
            layer = ref.GetLayerByIndex(layer_name_or_idx)
        elif isinstance(layer_name_or_idx, str):
            layer = ref.GetLayer(layer_name_or_idx)
        else:
            raise RuntimeError("Wrong datatype for layer selection")

        driver_name = utils_gdal._get_vector_driver_name_from_path(out_path)
        driver = ogr.GetDriverByName(driver_name)

        destination = driver.CreateDataSource(out_path)
        destination.CopyLayer(layer, layer.GetName(), ["OVERWRITE=YES"])
        destination.FlushCache()

        destination = None
        ref = None

        output.append(out_path)

    if input_is_list:
        return output

    return output[0]


def vector_to_extent(
    vector: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    *,
    latlng: bool = False,
    overwrite: bool = True,
) -> str:
    """Converts a vector to a vector file with the extent as a polygon.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to convert.

    out_path : str, optional
        The path to save the extent to. If None, the extent is saved in memory. Default: None.

    latlng : bool, optional
        If True, the extent is returned in latlng coordinates. If false,
        the projection of the vector is used. Default: False.

    overwrite : bool, optional
        If True, the output file is overwritten if it exists. Default: True.

    Returns
    -------
    str
        The path to the extent.
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(out_path, [str, type(None)], "out_path")
    utils_base._type_check(latlng, [bool], "latlng")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if out_path is None:
        out_path = utils_path._get_temp_filepath("temp_extent.gpkg", add_timestamp=True, add_uuid=True)

    if not utils_path._check_is_valid_output_filepath(out_path):
        raise ValueError(f"Invalid output path: {out_path}")

    metadata = _get_basic_metadata_vector(vector)

    if latlng:
        extent = metadata["bounds_latlng"]
    else:
        extent = metadata["bounds_vector"]

    extent = ogr.CreateGeometryFromWkt(extent, metadata["projection_osr"])

    driver_name = utils_gdal._get_driver_name_from_path(out_path)

    driver = ogr.GetDriverByName(driver_name)
    extent_ds = driver.CreateDataSource(out_path)
    extent_layer = extent_ds.CreateLayer("extent", metadata["projection_osr"], ogr.wkbPolygon)
    extent_feature = ogr.Feature(extent_layer.GetLayerDefn())
    extent_feature.SetGeometry(extent)
    extent_layer.CreateFeature(extent_feature)

    extent_ds = None
    extent_layer = None
    extent_feature = None

    return out_path


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

    input_list = utils_io._get_input_paths(vector, "vector")
    output_list = utils_io._get_output_paths(input_list, out_path)

    utils_path._delete_if_required_list(output_list, overwrite=True)

    output = []
    for idx, in_vector in enumerate(input_list):
        ref = _vector_open(in_vector)

        driver_name = utils_gdal._get_vector_driver_name_from_path(output_list[idx])
        driver = ogr.GetDriverByName(driver_name)

        destination = driver.CreateDataSource(output_list[idx])
        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()
            destination.CopyLayer(layer, layer.GetName(), ["OVERWRITE=YES"])

        destination.FlushCache()
        destination = None
        ref = None

        output.append(output_list[idx])

    if input_is_list:
        return output

    return output[0]


def vector_reset_fids(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
) -> Union[str, List[str]]:
    """Resets the FID column of a vector to 0, 1, 2, 3, ...

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

    input_is_list = isinstance(vector, list)

    input_list = utils_io._get_input_paths(vector, "vector")

    output = []
    for idx, in_vector in enumerate(input_list):
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()
            fids = []

            for feature in layer:
                fids.append(feature.GetFID())

            layer.ResetReading()
            fids = sorted(fids)

            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            for feature in layer:
                current_fid = feature.GetFID()
                target_fid = fids.index(current_fid)

                # If there is a fid field, update it too.
                if "fid" in field_names:
                    feature.SetField("fid", str(target_fid))

                feature.SetFID(target_fid)
                layer.SetFeature(feature)

            layer.SyncToDisk()

        ref = None

        output.append(input_list[idx])

    if input_is_list:
        return output

    return output[0]


def vector_create_fid_field(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
):
    """Creates a FID field in a vector if it doesn't exist.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    Returns
    -------
        str: original vector path
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."

    input_is_list = isinstance(vector, list)

    input_list = utils_io._get_input_paths(vector, "vector")

    output = []
    for idx, in_vector in enumerate(input_list):
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()
            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            if "fid" not in field_names:
                field = ogr.FieldDefn("fid", ogr.OFTInteger)
                layer.CreateField(field)

            layer.ResetReading()

            for idx, feature in enumerate(layer):
                feature.SetField("fid", idx)
                layer.SetFeature(feature)

            layer.SyncToDisk()

        ref = None

        output.append(input_list[idx])

    if input_is_list:
        return output
    
    return output[0]


def vector_create_attribute_from_fid(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    attribute_name: str = "id",
):
    """Creates an attribute from the FID field in a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).

    attribute_name : str, optional
        The name of the attribute to create. Default: "id"

    Returns
    -------
        str: original vector path
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(attribute_name, str), "attribute_name must be a string."

    input_is_list = isinstance(vector, list)

    input_list = utils_io._get_input_paths(vector, "vector")

    output = []
    for idx, in_vector in enumerate(input_list):
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()
            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            if "fid" in field_names and attribute_name not in field_names:
                field = ogr.FieldDefn(attribute_name, ogr.OFTInteger)
                layer.CreateField(field)

            layer.ResetReading()

            for feature in layer:
                feature.SetField(attribute_name, feature.GetFID())
                layer.SetFeature(feature)

            layer.SyncToDisk()

        ref = None

        output.append(input_list[idx])

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


def vector_add_field(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    field_name: str,
    field_type: str,
) -> Union[str, List[str]]:
    """Adds a field to a vector in place.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector layer(s) or path(s) to vector layer(s).

    field : str
        The name of the field to add.

    field_type : str
        The type of the field to add.
        `['int', 'integer', 'float', 'double', 'string', 'date', 'datetime', 'time', 'binary', 'intlist', 'integerlist', 'floatlist', 'doublelist', 'stringlist', 'datelist', 'datetimelist', 'timelist', 'binarylist']`

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector file(s).
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(field_name, str), "field must be a string."
    assert isinstance(field_type, str), "field_type must be a valid vector type."

    if field_type in ['int', 'integer']:
        field_type = ogr.OFTInteger
    elif field_type in ['float', 'double']:
        field_type = ogr.OFTReal
    elif field_type in ['string']:
        field_type = ogr.OFTString
    elif field_type in ['date', 'datetime', 'time']:
        field_type = ogr.OFTDateTime
    elif field_type in ['binary']:
        field_type = ogr.OFTBinary
    elif field_type in ['intlist', 'integerlist']:
        field_type = ogr.OFTIntegerList
    elif field_type in ['floatlist', 'doublelist']:
        field_type = ogr.OFTRealList
    elif field_type in ['stringlist']:
        field_type = ogr.OFTStringList
    else:
        raise ValueError(f"Invalid field_type: {field_type}")

    input_is_list = isinstance(vector, list)

    input_list = utils_io._get_input_paths(vector, "vector")

    for idx, in_vector in enumerate(input_list):
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()

            layer.CreateField(ogr.FieldDefn(field_name, field_type))

            layer.SyncToDisk()

        ref = None

    if input_is_list:
        return input_list
    
    return input_list[0]


def vector_set_attribute_table(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    header: List[str],
    attribute_table: List[List[Any]],
    match: Optional[str] = 'fid',
) -> Union[str, List[str]]:
    """Sets the attribute table of a vector in place.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector layer(s) or path(s) to vector layer(s).

    header : List[str]
        The header of the attributes to update in the table.

    attribute_table : List[List[Any]]
        The attributes to update in the table.

    match_fids : bool, optional
        If True, will match the FIDs of the input vector and the attribute table. Default: False.

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector file(s).
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(header, list), "header must be a list."
    assert isinstance(attribute_table, list), "attribute_table must be a list."
    assert len(header) == len(attribute_table[0]), "header and attribute_table must have the same number of columns."
    assert isinstance(match, (str, type(None))), "match must be a string or None."

    if match is not None:
        assert match in header, "match must be in header."

    match_idx = header.index(match)

    input_is_list = isinstance(vector, list)
    input_list = utils_io._get_input_paths(vector, "vector")

    for in_vector in input_list:
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()

            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            for field in header:
                if field not in field_names and field != 'fid':
                    layer.CreateField(ogr.FieldDefn(field, ogr.OFTString))

            if match is not None:
                for attr in attribute_table:
                    attr_id = attr[match_idx]
                    feature = layer.GetFeature(attr_id)

                    for field_idx, field in enumerate(header):
                        if field == match:
                            continue

                        feature.SetField(field, attr[field_idx])

                    layer.SetFeature(feature)

            else:
                feature_idx = 0
                for feature in layer:
                    for field_idx, field in enumerate(header):
                        feature.SetField(field, attribute_table[feature_idx][field_idx])
                        feature_idx += 1

                    layer.SetFeature(feature)

            layer.SyncToDisk()

        ref = None

    if input_is_list:
        return input_list
    
    return input_list[0]


def vector_delete_fields(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    fields: List[str],
) -> Union[str, List[str]]:
    """Deletes fields from a vector in place.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        Vector layer(s) or path(s) to vector layer(s).

    fields : List[str]
        The fields to delete.

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the output vector file(s).
    """
    assert isinstance(vector, (str, ogr.DataSource)), "vector must be a string or an ogr.DataSource object."
    assert isinstance(fields, list), "fields must be a list."

    input_is_list = isinstance(vector, list)
    input_list = utils_io._get_input_paths(vector, "vector")

    for in_vector in input_list:
        ref = _vector_open(in_vector)

        layers = ref.GetLayerCount()

        for layer_index in range(layers):
            layer = ref.GetLayer(layer_index)
            layer.ResetReading()

            layer_defn = layer.GetLayerDefn()
            field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

            for field in fields:
                if field in field_names:
                    layer.DeleteField(field)

            layer.SyncToDisk()

        ref = None

    if input_is_list:
        return input_list
    
    return input_list[0]