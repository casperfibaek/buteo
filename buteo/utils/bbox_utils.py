"""
### Bounding box utility functions ###

Various utility functions to work with bounding boxes and gdal.

There are two different formats for bounding boxes used by GDAL:</br>
OGR:  `[x_min, x_max, y_min, y_max]`</br>
WARP: `[x_min, y_min, x_max, y_max]`</br>

_If nothing else is stated, the OGR format is used._

The GDAL geotransform is a list of six parameters:</br>
`x_min, pixel_width, row_skew, y_max, column_skew, pixel_height (negative for north-up)`

TODO:
    * create tests
"""

# Standard library
import sys; sys.path.append("../../")
from uuid import uuid4

# External
import numpy as np
from osgeo import ogr, osr, gdal

# Internal
from buteo.utils import core_utils



def is_valid_bbox(bbox_ogr):
    """
    Checks if a bbox is valid.

    A valid ogr formatted bbox has the form: </br>
    `[x_min, x_max, y_min, y_max]`

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_bool_): **True** if the bbox is valid, **False** otherwise.
    """
    if not isinstance(bbox_ogr, list):
        return False

    if len(bbox_ogr) != 4:
        return False

    for val in bbox_ogr:
        if not core_utils.is_number(val):
            return False

    x_min, x_max, y_min, y_max = bbox_ogr

    if x_min > x_max or y_min > y_max:
        return False

    return True


def is_valid_bbox_latlng(bbox_ogr_latlng):
    """
    Checks if a bbox is valid and latlng.

    A valid ogr formatted bbox has the form: </br>
    `[x_min, x_max, y_min, y_max]`

    ## Args:
    `bbox_ogr_latlng` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_bool_): **True** if the bbox is valid, **False** otherwise.
    """
    if not is_valid_bbox(bbox_ogr_latlng):
        return False

    x_min, x_max, y_min, y_max = bbox_ogr_latlng
    if x_min < -180 or x_max > 180 or y_min < -90 or y_max > 90:
        return False

    return True


def is_valid_geotransform(geotransform):
    """
    Checks if a geotransform is valid.

    A valid geotransform has the form: </br>
    `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`

    ## Args:
    `geotransform` (_list_/_tuple_): A GDAL formatted geotransform.

    ## Returns:
    (_bool_): **True** if the geotransform is valid, **False** otherwise.
    """
    if not isinstance(geotransform, (list, tuple)):
        return False

    if len(geotransform) != 6:
        return False

    for val in geotransform:
        if not core_utils.is_number(val):
            return False

    return True


def ensure_negative(number):
    """
    Ensures that a valid is negative. If the number is positive, it is made negative.

    ## Args:
    `number` (_int_/_float_): A float or int number. </br>

    ## Returns:
    (_int_/_float_): The same number made **negative** if necesary.
    """
    assert core_utils.is_number(number), f"number must be a number. Received: {number}"

    if number <= 0:
        return -number

    return number


def get_pixel_offsets(geotransform, bbox_ogr):
    """
    Get the pixels offsets for a bbox and a geotransform.

    ## Args:
    `geotransform` (_list_/_tuple_): A GDAL GeoTransform. </br>
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_list_): A list of pixel offsets. `[x_start, y_start, x_size, y_size]`
    """
    assert is_valid_geotransform(
        geotransform
    ), f"geotransform must be a list of length six. Received: {geotransform}."
    assert is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr must be a valid OGR formatted bbox. Received: {bbox_ogr}."

    x_min, x_max, y_min, y_max = bbox_ogr

    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = abs(geotransform[1])
    pixel_height = abs(geotransform[5])

    x_1 = int((x_min - origin_x) / pixel_width)
    x_2 = int((x_max - origin_x) / pixel_width) + 1

    y_1 = int((y_max - origin_y) / pixel_height)
    y_2 = int((y_min - origin_y) / pixel_height) + 1

    x_size = x_2 - x_1
    y_size = y_2 - y_1

    x_start = x_1
    y_start = y_1

    return [x_start, y_start, x_size, y_size]


def get_bbox_from_geotransform(geotransform, raster_x_size, raster_y_size):
    """
    Get an OGR bounding box from a geotransform and raster sizes.

    ## Args:
    `geotransform` (_list_/_tuple_): A GDAL GeoTransform. </br>
    `raster_x_size` (_int_): The number of pixels in the x direction. </br>
    `raster_y_size` (_int_): The number of pixels in the y direction. </br>

    ## Returns:
    (_list_): An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert is_valid_geotransform(
        geotransform
    ), f"geotransform was not a valid geotransform. Received: {geotransform}"

    x_min, pixel_width, _row_skew, y_max, _column_skew, pixel_height = geotransform

    x_max = x_min + (raster_x_size * pixel_width)
    y_min = y_max + (raster_y_size * pixel_height)

    return [x_min, x_max, y_min, y_max]


def get_bbox_from_raster(raster_dataframe):
    """
    Gets an OGR bounding box from a GDAL raster dataframe.

    ## Args:
    `raster_dataframe` (_gdal.DataFrame_): A GDAL raster dataframe. </br>

    ## Returns:
    (_list_): An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert isinstance(
        raster_dataframe, gdal.Dataset
    ), f"raster_dataframe was not a gdal.Datasource. Received: {raster_dataframe}"

    bbox = get_bbox_from_geotransform(
        raster_dataframe.GetGeoTransform(),
        raster_dataframe.RasterXSize,
        raster_dataframe.RasterYSize,
    )

    return bbox


def get_bbox_from_vector(vector_dataframe):
    """
    Gets an OGR bounding box from an OGR dataframe.

    ## Args:
    `vector_dataframe` (_ogr.DataSource_): An OGR vector dataframe. </br>

    ## Returns:
    (_list_): An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert isinstance(
        vector_dataframe, ogr.DataSource
    ), f"vector_dataframe was not a valid ogr.DataSource. Received: {vector_dataframe}"

    layer_count = vector_dataframe.GetLayerCount()

    assert (
        layer_count > 0
    ), f"vector_dataframe did not contain any layers. Received: {vector_dataframe}"

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    for layer_index in range(layer_count):
        layer = vector_dataframe.GetLayerByIndex(layer_index)
        layer_x_min, layer_x_max, layer_y_min, layer_y_max = layer.GetExtent()

        if layer_index == 0:
            x_min = layer_x_min
            x_max = layer_x_max
            y_min = layer_y_min
            y_max = layer_y_max
        else:
            if layer_x_min < x_min:
                x_min = layer_x_min
            if layer_x_max > x_max:
                x_max = layer_x_max
            if layer_y_min < y_min:
                y_min = layer_y_min
            if layer_y_max > y_max:
                y_max = layer_y_max

    return [x_min, x_max, y_min, y_max]


def get_bbox_from_vector_layer(vector_layer):
    """
    Gets an OGR bounding box from an OGR dataframe layer.

    ## Args:
    `vector_layer` (_ogr.Layer_): An OGR vector dataframe layer. </br>

    ## Returns:
    (_list_): An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert isinstance(
        vector_layer, ogr.Layer
    ), f"vector_layer was not a valid ogr.Layer. Received: {vector_layer}"

    x_min, x_max, y_min, y_max = vector_layer.GetExtent()

    return [x_min, x_max, y_min, y_max]


def get_bbox_from_dataset(dataset):
    """
    Get the bbox from a dataset.

    ## Args:
    `dataset` (_str_/_gdal.Dataset_/_ogr.DataSource): A dataset or dataset path. </br>

    ## Returns:
    (_list_): The bounding box in ogr format: [x_min, x_max, y_min, y_max].
    """
    assert isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)), "DataSet must be a string, ogr.DataSource, or gdal.Dataset."

    opened = dataset if isinstance(dataset, (gdal.Dataset, ogr.DataSource)) else None

    if opened is None:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = gdal.Open(dataset, gdal.GA_ReadOnly)

        if opened is None:
            opened = ogr.Open(dataset, gdal.GA_ReadOnly)

        gdal.PopErrorHandler()
        if opened is None:
            raise Exception(f"Could not open dataset. {dataset}")

    if isinstance(opened, gdal.Dataset):
        return get_bbox_from_raster(opened)

    if isinstance(opened, ogr.DataSource):
        return get_bbox_from_vector(opened)

    raise Exception(f"Could not get bbox from dataset. {dataset}")


def get_sub_geotransform(geotransform, bbox_ogr):
    """
    Create a GeoTransform and the raster sizes for an OGR formatted bbox.

    ## Args:
    `geotransform` (_list_): A GDAL geotransform. </br>
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_dict_): { "Transform": _list_, "RasterXSize": _int_, "RasterYSize": _int_ }
    """
    assert is_valid_geotransform(
        geotransform
    ), f"geotransform must be a valid geotransform. Received: {geotransform}."
    assert is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    raster_x_size = round((x_max - x_min) / pixel_width)
    raster_y_size = round((y_max - y_min) / pixel_height)

    return {
        "Transform": [x_min, pixel_width, 0, y_max, 0, ensure_negative(pixel_height)],
        "RasterXSize": abs(raster_x_size),
        "RasterYSize": abs(raster_y_size),
    }


def convert_bbox_to_geom(bbox_ogr):
    """
    Convert an OGR bounding box to ogr.Geometry.</br>
    `[x_min, x_max, y_min, y_max] -> ogr.Geometry`

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_ogr.Geometry_): An OGR geometry.
    """
    assert is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_min, y_min)
    ring.AddPoint(x_max, y_min)
    ring.AddPoint(x_max, y_max)
    ring.AddPoint(x_min, y_max)
    ring.AddPoint(x_min, y_min)

    geom = ogr.Geometry(ogr.wkbPolygon)
    geom.AddGeometry(ring)

    return geom


def convert_geom_to_bbox(geom):
    """
    Convert an ogr.Geometry to an OGR bounding box.</br>
    `ogr.Geometry -> [x_min, x_max, y_min, y_max]`

    ## Args:
    `geom` (_ogr.Geometry_): An OGR geometry. </br>

    ## Returns:
    (_list_): An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert isinstance(
        geom, ogr.Geometry
    ), f"geom was not a valid ogr.Geometry. Received: {geom}"

    bbox_ogr = list(geom.GetEnvelope()) # [x_min, x_max, y_min, y_max]

    return bbox_ogr


def convert_bbox_to_geotransform(bbox_ogr, raster_x_size, raster_y_size):
    """
    Convert an OGR formatted bounding box to a GDAL GeoTransform.</br>
    `[x_min, x_max, y_min, y_max] -> [x_min, pixel_width, x_skew, y_max, y_skew, pixel_height]`

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>
    `raster_x_size` (_int_): The number of pixels in the x direction. </br>
    `raster_y_size` (_int_): The number of pixels in the y direction. </br>

    ## Returns:
    (_list_): A GDAL GeoTransform. `[x_min, pixel_width, x_skew, y_max, y_skew, pixel_height]`
    """
    assert is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    origin_x = x_min
    origin_y = y_max
    pixel_width = (x_max - x_min) / raster_x_size
    pixel_height = (y_max - y_min) / raster_y_size

    return [origin_x, pixel_width, 0, origin_y, 0, ensure_negative(pixel_height)]


def convert_ogr_bbox_to_gdal_bbox(bbox_ogr):
    """
    Converts an OGR formatted bbox to a GDAL formatted one.</br>
    `[x_min, x_max, y_min, y_max] -> [x_min, y_min, x_max, y_max]`

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_list_): A GDAL formatted bbox. `[x_min, y_min, x_max, y_max]`
    """
    assert is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    return [x_min, y_min, x_max, y_max]


def convert_gdal_bbox_to_ogr_bbox(bbox_gdal):
    """
    Converts a GDAL formatted bbox to an OGR formatted one.</br>
    `[x_min, y_min, x_max, y_max] -> [x_min, x_max, y_min, y_max]`

    ## Args:
    `bbox_gdal` (_list_): A GDAL formatted bbox. </br>

    ## Returns:
    (_list_): An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert (
        isinstance(bbox_gdal, list) and len(bbox_gdal) == 4
    ), f"bbox_gdal must be a list of length four. Received: {bbox_gdal}."

    x_min, y_min, x_max, y_max = bbox_gdal

    return [x_min, x_max, y_min, y_max]


def convert_bbox_to_wkt(bbox_ogr):
    """
    Converts an OGR formatted bbox to a WKT string.</br>
    `[x_min, x_max, y_min, y_max] -> WKT`

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_str_): A WKT Polygon string. `POLYGON ((...))`
    """
    assert is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    wkt = f"POLYGON (({x_min} {y_min}, {x_max} {y_min}, {x_max} {y_max}, {x_min} {y_max}, {x_min} {y_min}))"

    return wkt


def convert_bbox_to_geojson(bbox_ogr):
    """
    Converts an OGR formatted bbox to a GeoJson dictionary.</br>
    `[x_min, x_max, y_min, y_max] -> GeoJson`

    ## Args:
    `bbox_ogr` (_list_): an OGR formatted bbox. </br>

    ## Returns:
    (_dict_): A GeoJson Dictionary. `{ "type": "Polygon", "coordinates": [ ... ] }`
    """
    assert is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    geojson = {
        "type": "Polygon",
        "coordinates": [
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
        ],
    }

    return geojson


def convert_bbox_to_vector(bbox_ogr, projection_osr):
    """
    Converts an OGR formatted bbox to an in-memory vector.</br>
    _Vectors are stored in /vsimem/ as .gpkg files._</br>
    **OBS**: Layers should be manually cleared when no longer used.

    ## Args:
    `bbox_ogr` (_list_): an OGR formatted bbox. </br>
    `projection_osr` (_osr.SpatialReference_): The projection of the vector. </br>

    ## Returns:
    (_ogr.DataSource_): The bounding box as a vector.
    """
    assert is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"
    assert isinstance(
        projection_osr, osr.SpatialReference
    ), f"projection_osr not a valid spatial reference. Recieved: {projection_osr}"

    geom = convert_bbox_to_geom(bbox_ogr)

    driver = ogr.GetDriverByName("GPKG")
    extent_name = f"/vsimem/{core_utils.get_unix_seconds_as_str()}_{uuid4().int}_extent.gpkg"
    extent_ds = driver.CreateDataSource(extent_name)

    layer = extent_ds.CreateLayer("extent_ogr", projection_osr, ogr.wkbPolygon)

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)

    feature = None
    layer.SyncToDisk()

    return extent_name


def bboxes_intersect(bbox1_ogr, bbox2_ogr):
    """
    Checks if two OGR formatted bboxes intersect.

    ## Args:
    `bbox1_ogr` (_list_): An OGR formatted bbox. </br>
    `bbox2_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_bool_): **True** if the bboxes intersect, **False** otherwise.
    """
    assert is_valid_bbox(
        bbox1_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox1_ogr}"
    assert is_valid_bbox(
        bbox2_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox2_ogr}"

    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    if bbox2_x_min > bbox1_x_max:
        return False

    if bbox2_y_min > bbox1_y_max:
        return False

    if bbox2_x_max > bbox1_x_min:
        return False

    if bbox2_y_max > bbox1_y_min:
        return False

    return True


def bboxes_within(bbox1_ogr, bbox2_ogr):
    """
    Checks if one OGR formatted bbox is within another.

    ## Args:
    `bbox1_ogr` (_list_): An OGR formatted bbox. </br>
    `bbox2_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_bool_): **True** if the bbox is within the other, **False** otherwise.
    """
    assert is_valid_bbox(
        bbox1_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox1_ogr}"
    assert is_valid_bbox(
        bbox2_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox2_ogr}"

    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    return (
        (bbox1_x_min >= bbox2_x_min)
        and (bbox1_x_max <= bbox2_x_max)
        and (bbox1_y_min >= bbox2_y_min)
        and (bbox1_y_max <= bbox2_y_max)
    )


def get_intersection_bboxes(bbox1_ogr, bbox2_ogr):
    """
    Get the intersection of two OGR formatted bboxes.

    ## Args:
    `bbox1_ogr` (_list_): An OGR formatted bbox. </br>
    `bbox2_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_list_): An OGR formatted bbox of the intersection. `[x_min, x_max, y_min, y_max]`
    """
    assert bboxes_intersect(
        bbox1_ogr, bbox2_ogr
    ), "The two bounding boxes do not intersect."

    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    return [
        max(bbox1_x_min, bbox2_x_min),
        min(bbox1_x_max, bbox2_x_max),
        max(bbox1_y_min, bbox2_y_min),
        min(bbox1_y_max, bbox2_y_max),
    ]


def get_union_bboxes(bbox1_ogr, bbox2_ogr):
    """
    Get the union of two OGR formatted bboxes.

    ## Args:
    `bbox1_ogr` (_list_): An OGR formatted bbox. </br>
    `bbox2_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_list_): An OGR formatted bbox of the union. `[x_min, x_max, y_min, y_max]`
    """
    assert is_valid_bbox(
        bbox1_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox1_ogr}"
    assert is_valid_bbox(
        bbox2_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox2_ogr}"

    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    return [
        min(bbox1_x_min, bbox2_x_min),
        max(bbox1_x_max, bbox2_x_max),
        min(bbox1_y_min, bbox2_y_min),
        max(bbox1_y_max, bbox2_y_max),
    ]


def get_projection_from_raster(raster):
    """
    Get the projection from a raster.

    ## Args:
    `raster` (_str_/_gdal.Dataset_): A raster or raster path. </br>

    ## Returns:
    (_osr.SpatialReference_): The projection in OSR format.
    """
    opened = None
    if isinstance(raster, gdal.Dataset):
        opened = raster
    else:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = gdal.Open(raster, gdal.GA_ReadOnly)
        gdal.PopErrorHandler()

    if opened is None:
        raise Exception(f"Could not open raster. {raster}")

    projection = osr.SpatialReference()
    projection.ImportFromWkt(opened.GetProjection())
    opened = None

    return projection


def get_projection_from_vector(vector):
    """
    Get the projection from a vector.

    ## Args:
    `vector` (_str_/_gdal.Dataset_): A vector or vector path. </br>

    ## Returns:
    (_osr.SpatialReference_): The projection in OSR format.
    """
    opened = None
    if isinstance(vector, ogr.DataSource):
        opened = vector
    else:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = ogr.Open(vector, gdal.GA_ReadOnly)
        gdal.PopErrorHandler()

    if opened is None:
        raise Exception(f"Could not open vector. {vector}")

    layer = opened.GetLayer()
    projection = layer.GetSpatialRef()
    opened = None

    return projection


def get_projection_from_dataset(dataset):
    """
    Get the projection from a dataset.

    ## Args:
    `dataset` (_str_/_gdal.Dataset_/_ogr.DataSource): A dataset or dataset path. </br>

    ## Returns:
    (_osr.SpatialReference_): The projection in OSR format.
    """
    assert isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)), "DataSet must be a string, ogr.DataSource, or gdal.Dataset."

    opened = dataset if isinstance(dataset, (gdal.Dataset, ogr.DataSource)) else None

    if opened is None:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = gdal.Open(dataset, gdal.GA_ReadOnly)

        if opened is None:
            opened = ogr.Open(dataset, gdal.GA_ReadOnly)

        gdal.PopErrorHandler()
        if opened is None:
            raise Exception(f"Could not open dataset. {dataset}")

    if isinstance(opened, gdal.Dataset):
        return get_projection_from_raster(opened)

    if isinstance(opened, ogr.DataSource):
        return get_projection_from_vector(opened)

    raise Exception(f"Could not get projection from dataset. {dataset}")


def align_bboxes_to_pixel_size(bbox1_ogr, bbox2_ogr, pixel_width, pixel_height):
    """
    Aligns two OGR formatted bboxes to a pixel size.

    ## Args:
    `bbox1_ogr` (_list_): An OGR formatted bbox. </br>
    `bbox2_ogr` (_list_): An OGR formatted bbox. </br>
    `pixel_width` (_float_/_int_): The width of the pixel. </br>
    `pixel_height` (_float_/_int_): The height of the pixel. </br>

    ## Returns:
    (_list_): An OGR formatted bbox of the alignment. `[x_min, x_max, y_min, y_max]`
    """
    assert (
        isinstance(bbox1_ogr, list) and len(bbox1_ogr) == 4
    ), f"bbox1_ogr must be a list of length four. Received: {bbox1_ogr}."
    assert (
        isinstance(bbox2_ogr, list) and len(bbox2_ogr) == 4
    ), f"bbox2_ogr must be a list of length four. Received: {bbox2_ogr}."

    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    x_min = bbox2_x_min - ((bbox2_x_min - bbox1_x_min) % pixel_width)
    x_max = bbox2_x_max + ((bbox1_x_max - bbox2_x_max) % pixel_width)

    y_min = bbox2_y_min - ((bbox2_y_min - bbox1_y_min) % abs(pixel_height))
    y_max = bbox2_y_max + ((bbox1_y_max - bbox2_y_max) % abs(pixel_height))

    return x_min, x_max, y_min, y_max


def reproject_bbox(
    bbox_ogr,
    source_projection_osr,
    target_projection_osr,
):
    """
    Reprojects an OGR formatted bbox.

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>
    `source_projection_osr` (_osr.SpatialReference_): The source projection. </br>
    `target_projection_osr` (_osr.SpatialReference_): The target projection. </br>

    ## Returns:
    (_list_): An OGR formatted reprojected bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert is_valid_bbox(bbox_ogr), f"Invalid bbox. Received: {bbox_ogr}."
    assert isinstance(
        source_projection_osr, osr.SpatialReference
    ), f"source_projection not a valid spatial reference. Recieved: {source_projection_osr}"
    assert isinstance(
        target_projection_osr, osr.SpatialReference
    ), f"target_projection not a valid spatial reference. Recieved: {target_projection_osr}"

    if source_projection_osr.IsSame(target_projection_osr):
        return bbox_ogr

    transformer = osr.CoordinateTransformation(
        source_projection_osr, target_projection_osr
    )

    x_min, x_max, y_min, y_max = bbox_ogr

    transformed_x_min, transformed_y_min, _z = transformer.TransformPoint(float(x_min), float(y_min))
    transformed_x_max, transformed_y_max, _z = transformer.TransformPoint(float(x_max), float(y_max))

    return [
        transformed_x_min,
        transformed_x_max,
        transformed_y_min,
        transformed_y_max,
    ]


def get_utm_zone_from_latlng(latlng, return_name=False):
    """ Get the UTM ZONE from a latlng list.

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_osr.SpatialReference_): The UTM zone projection.
    """
    assert isinstance(latlng, (list, np.ndarray)), "latlng must be in the form of a list."

    zone = round(((latlng[1] + 180) / 6) + 1)
    n_or_s = "S" if latlng[0] < 0 else "N"

    if return_name:
        return f"UTM_{zone}_{n_or_s}"

    false_northing = "10000000" if n_or_s == "S" else "0"
    central_meridian = str(round(((zone * 6) - 180) - 3))
    epsg = f"32{'7' if n_or_s == 'S' else '6'}{str(zone)}"

    wkt = f"""
        PROJCS["WGS 84 / UTM zone {str(zone)}{n_or_s}",
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]],
        PROJECTION["Transverse_Mercator"],
        PARAMETER["latitude_of_origin",0],
        PARAMETER["central_meridian",{central_meridian}],
        PARAMETER["scale_factor",0.9996],
        PARAMETER["false_easting",500000],
        PARAMETER["false_northing",{false_northing}],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["EPSG","{epsg}"]]
    """
    projection = osr.SpatialReference()
    projection.ImportFromWkt(wkt)

    return projection


def get_utm_zone_from_bbox(bbox_ogr_latlng):
    """
    Get the UTM zone from an OGR formatted bbox.

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>

    ## Returns:
    (_osr.SpatialReference_): The UTM zone projection.
    """
    assert is_valid_bbox(
        bbox_ogr_latlng
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr_latlng}"
    assert is_valid_bbox_latlng(bbox_ogr_latlng), "Bbox is not in latlng format."

    bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = bbox_ogr_latlng

    mid_lng = (bbox_x_min + bbox_x_max) / 2
    mid_lat = (bbox_y_min + bbox_y_max) / 2

    return get_utm_zone_from_latlng([mid_lat, mid_lng])


def get_utm_zone_from_dataset(dataset):
    """
    Get the UTM zone from a GDAL dataset.

    ## Args:
    `dataset` (_obj_): A GDAL dataset. </br>

    ## Returns:
    (_str_): The UTM zone.
    """
    assert isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)), "dataset was not a valid dataset."

    bbox = get_bbox_from_dataset(dataset)
    source_projection = get_projection_from_dataset(dataset)
    target_projection = osr.SpatialReference(); target_projection.ImportFromEPSG(4326)

    bbox = reproject_bbox(bbox, source_projection, target_projection)
    utm_zone = get_utm_zone_from_bbox(bbox)

    return utm_zone


def get_utm_zone_from_dataset_list(datasets):
    """
    Get the UTM zone from a list of GDAL datasets.

    ## Args:
    `datasets` (_list_): A list of GDAL datasets. </br>

    ## Returns:
    (_str_): The UTM zone.
    """
    assert isinstance(datasets, (list, np.ndarray)), "datasets was not a valid list."

    latlng_bboxes = []
    latlng_proj = osr.SpatialReference(); latlng_proj.ImportFromEPSG(4326)

    for dataset in datasets:
        bbox = get_bbox_from_dataset(dataset)
        projection = get_projection_from_dataset(dataset)
        latlng_bboxes.append(reproject_bbox(bbox, projection, latlng_proj))

    union_bbox = latlng_bboxes[0]
    for bbox in latlng_bboxes[1:]:
        union_bbox = get_union_bboxes(union_bbox, bbox)

    utm_zone = get_utm_zone_from_bbox(union_bbox)

    return utm_zone


def reproject_latlng_point_to_utm(latlng):
    """ Converts a latlng point into an UTM point.

        Takes point in [lat, lng], returns [utm_x, utm_y].
    """
    source_projection = osr.SpatialReference(); source_projection.ImportFromEPSG(4326)
    target_projection = get_utm_zone_from_latlng(latlng)

    transformer = osr.CoordinateTransformation(
        source_projection, target_projection
    )

    utm_x, utm_y, _utm_z = transformer.TransformPoint(float(latlng[0]), float(latlng[1]))

    return [utm_x, utm_y]


def additional_bboxes(bbox_ogr, projection_osr):
    """
    This is an internal utility function for metadata generation. It takes a standard
    OGR bounding box and returns a list of variations of bounding boxes.

    ## Args:
    `bbox_ogr` (_list_): An OGR formatted bbox. </br>
    `projection_osr` (_osr.SpatialReference_): The projection. </br>

    ## Returns:
    (_dict_): A dictionary of the added bboxes. Contains the following keys: </br>
    `bbox_latlng`: The bbox in latlng coordinates. </br>
    `bbox_wkt`: The bbox in WKT format. </br>
    `bbox_wkt_latlng`: The bbox in WKT format in latlng coordinates. </br>
    `bbox_geom`: The bbox in ogr.Geometry format. </br>
    `bbox_geom_latlng`: The bbox in ogr.Geometry format in latlng coordinates. </br>
    `bbox_gdal`: The bbox in GDAL format. </br>
    `bbox_gdal_latlng`: The bbox in GDAL format in latlng coordinates. </br>
    `bbox_dict`: The bbox in a dictionary format. { "x_min": x_min, ... } </br>
    `bbox_dict_latlng`: The bbox in a dictionary format in latlng coordinates. </br>
    """
    assert is_valid_bbox(bbox_ogr), f"Invalid bbox. Received: {bbox_ogr}."
    assert isinstance(
        projection_osr, osr.SpatialReference
    ), f"source_projection not a valid spatial reference. Recieved: {projection_osr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    original_projection = osr.SpatialReference()
    original_projection.ImportFromWkt(projection_osr.ExportToWkt())

    latlng_projection = osr.SpatialReference()
    latlng_projection.ImportFromEPSG(4326)

    bbox_ogr_latlng = reproject_bbox(bbox_ogr, original_projection, latlng_projection)
    latlng_x_min, latlng_x_max, latlng_y_min, latlng_y_max = bbox_ogr_latlng

    bbox_geom = convert_bbox_to_geom(bbox_ogr)
    bbox_geom_latlng = convert_bbox_to_geom(bbox_ogr_latlng)

    bbox_wkt = convert_bbox_to_wkt(bbox_ogr)
    bbox_wkt_latlng = convert_bbox_to_wkt(bbox_ogr_latlng)

    return {
        "bbox_latlng": bbox_ogr_latlng,
        "bbox_wkt": bbox_wkt,
        "bbox_wkt_latlng": bbox_wkt_latlng,
        "bbox_geom": bbox_geom,
        "bbox_geom_latlng": bbox_geom_latlng,
        "bbox_gdal": convert_ogr_bbox_to_gdal_bbox(bbox_ogr),
        "bbox_gdal_latlng": convert_ogr_bbox_to_gdal_bbox(bbox_ogr_latlng),
        "bbox_dict": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "bbox_dict_latlng": {
            "x_min": latlng_x_min,
            "x_max": latlng_x_max,
            "y_min": latlng_y_min,
            "y_max": latlng_y_max,
        },
        "bbox_geojson": convert_bbox_to_geojson(bbox_ogr_latlng),
    }
