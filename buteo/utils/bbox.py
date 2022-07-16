"""
Various utility functions to work with bounding boxes and gdal.

GDAL uses two different formats.
    WARP: x_min, y_min, x_max, y_max
    OGR: x_min, x_max, y_min, y_max - x, y

If nothing else is stated, the OGR format is used.

The GDAL geotransform is a list of six parameters:
    x_min, pixel_width, row_skew, y_max, column_skew, pixel_height (negative for north-up)

TODO:
    - create tests

"""

# Standard library
from uuid import uuid4

# External
from osgeo import ogr, osr, gdal

# Internal
from .core import is_number


def is_valid_bbox(bbox_ogr):
    """
    ### Description:
    Checks if a bbox is valid.
    A valid bbox has the form: `[x_min, x_max, y_min, y_max]`

    ### Args:
    `bbox_ogx`: An OGR formatted bbox.

    ### Returns:
    **True** if the bbox is valid, **False** otherwise.
    """
    if not isinstance(bbox_ogr, list):
        return False

    if len(bbox_ogr) != 4:
        return False

    for val in bbox_ogr:
        if not is_number(val):
            return False

    x_min, x_max, y_min, y_max = bbox_ogr

    if x_min > x_max or y_min > y_max:
        return False

    return True


def is_valid_geotransform(geotransform):
    """
    ### Description:
    Checks if a geotransform is valid.
    A valid geotransform has the form: `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`

    ### Args:
    `geotransform`: A GDAL formatted geotransform.

    ### Returns:
    **True** if the geotransform is valid, **False** otherwise.
    """
    if not isinstance(geotransform, (list, tuple)):
        return False

    if len(geotransform) != 6:
        return False

    for val in geotransform:
        if not is_number(val):
            return False

    return True


def ensure_negative(number):
    """
    ### Description:
    Ensures that a valid is negative. If the number is positive, it is made negative.

    ### Args:
    `number`: A float or int number.

    ### Returns:
    The same number made **negative** if necesary.
    """
    assert is_number(number), \
        f"number must be a number. Received: {number}"

    if number <= 0:
        return -number

    return number


def get_pixel_offsets(geotransform, bbox_ogr):
    """
    The pixels offsets for a bbox and a geotransform.

    Returns [x1, y1, xsize, ysize]

    """
    assert is_valid_geotransform(geotransform), \
        f"geotransform must be a list of length six. Received: {geotransform}."
    assert is_valid_bbox(bbox_ogr), \
        f"bbox_ogr must be a valid ogr style bbox. Received: {bbox_ogr}."

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

    return [x_1, y_1, x_size, y_size]


def get_bbox_from_geotransform(geotransform, raster_x_size, raster_y_size):
    """
    Gets an ogr bounding box from a gdal raster dataframe.
    OGR format: x_min, x_max, y_min, y_max

    Returns [x_min, x_max, y_min, y_max]
    """
    assert is_valid_geotransform(geotransform), \
        f"geotransform was not a valid geotransform. Received: {geotransform}"

    x_min, pixel_width, _row_skew, y_max, _column_skew, pixel_height = geotransform

    x_max = x_min + (raster_x_size * pixel_width)
    y_min = y_max + (raster_y_size * pixel_height)

    return [x_min, x_max, y_min, y_max]


def get_bbox_from_raster(raster_dataframe):
    """
    Gets an ogr bounding box from a gdal raster dataframe.
    OGR format: x_min, x_max, y_min, y_max

    Returns [x_min, y_min, x_max, y_max]
    """
    assert isinstance(raster_dataframe, gdal.Dataset), \
        f"raster_dataframe was not a gdal.Datasource. Received: {raster_dataframe}"

    bbox = get_bbox_from_geotransform(
        raster_dataframe.GetGeoTransform(),
        raster_dataframe.RasterXSize,
        raster_dataframe.RasterYSize,
    )

    return bbox


def get_bbox_from_vector(vector_dataframe):
    """
    Gets an ogr bounding box from a gdal vector dataframe.
    OGR format: x_min, x_max, y_min, y_max

    Returns [x_min, x_max, y_min, y_max]
    """
    assert isinstance(vector_dataframe, ogr.DataSource), \
        f"vector_dataframe was not a valid ogr.DataSource. Received: {vector_dataframe}"

    layer_count = vector_dataframe.GetLayerCount()

    assert layer_count > 0, \
        f"vector_dataframe did not contain any layers. Received: {vector_dataframe}"

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
    Gets an ogr bounding box from a gdal vector layer.
    OGR format: x_min, x_max, y_min, y_max

    Returns [x_min, x_max, y_min, y_max]
    """
    assert isinstance(vector_layer, ogr.Layer), \
        f"vector_layer was not a valid ogr.Layer. Received: {vector_layer}"

    x_min, x_max, y_min, y_max = vector_layer.GetExtent()

    return [x_min, x_max, y_min, y_max]


def convert_bbox_to_geom(bbox_ogr):
    """
    Converts an OGR formatted bbox to a GDAL geom.

    [x_min, x_max, y_min, y_max] -> ogr.Geometry.Polygon
    """
    assert is_valid_bbox(bbox_ogr), \
        f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

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


def convert_bbox_to_geotransform(bbox_ogr, raster_x_size, raster_y_size):
    """
    Converts an OGR formatted bbox to a GDAL geotransform.

    [x_min, x_max, y_min, y_max] -> [x_min, pixel_width, 0, y_max, 0, pixel_height]
    """
    assert is_valid_bbox(bbox_ogr), \
        f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    origin_x = x_min
    origin_y = y_max
    pixel_width = (x_max - x_min) / raster_x_size
    pixel_height = (y_max - y_min) / raster_y_size

    return [origin_x, pixel_width, 0, origin_y, 0, ensure_negative(pixel_height)]


def convert_geom_to_bbox(geom):
    """
    Converts a GDAL geom to an OGR formatted bbox.

    ogr.Geometry.Polygon -> [x_min, x_max, y_min, y_max]
    """
    assert isinstance(geom, ogr.Geometry), \
        f"geom was not a valid ogr.Geometry. Received: {geom}"

    bbox_ogr = geom.GetEnvelope()

    return bbox_ogr # [x_min, x_max, y_min, y_max]


def convert_ogr_bbox_to_gdal_bbox(bbox_ogr):
    """
    Converts an OGR formatted bbox to a GDAL formatted one.

    [x_min, x_max, y_min, y_max] -> [x_min, y_min, x_max, y_max]
    """
    assert is_valid_bbox(bbox_ogr), \
        f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    return [x_min, y_min, x_max, y_max]


def convert_gdal_bbox_to_ogr_bbox(bbox_gdal):
    """
    Converts an GDAL formatted bbox to an OGR formatted one.

    [x_min, y_min, x_max, y_max] -> [x_min, x_max, y_min, y_max]
    """
    assert isinstance(bbox_gdal, list) and len(bbox_gdal) == 4, \
        f"bbox_gdal must be a list of length four. Received: {bbox_gdal}."

    x_min, y_min, x_max, y_max = bbox_gdal

    return [x_min, x_max, y_min, y_max]


def convert_bbox_to_wkt(bbox_ogr):
    """
    Converts an OGR formatted bbox to a WKT string.

    [x_min, x_max, y_min, y_max] -> WKT
    """
    assert is_valid_bbox(bbox_ogr), \
        f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    wkt = f"POLYGON (({x_min} {y_min}, {x_max} {y_min}, {x_max} {y_max}, {x_min} {y_max}, {x_min} {y_min}))"

    return wkt


def convert_bbox_to_geojson(bbox_ogr):
    """
    Converts an OGR formatted bbox to a GeoJSON string.

    [x_min, x_max, y_min, y_max] -> GeoJSON
    """
    assert is_valid_bbox(bbox_ogr), \
        f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    geojson = {
        "type": "Polygon",
        "coordinates": [
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min]
            ]
        ]
    }

    return geojson


def convert_bbox_to_vector(bbox_ogr, projection_osr):
    """
    Converts an OGR formatted bbox to an ogr.datasource path in memory.

    OBS: Remember to clear layer when no longer in use.
    """
    assert is_valid_bbox(bbox_ogr), \
        f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    assert isinstance(projection_osr, osr.SpatialReference), \
        f"projection_osr not a valid spatial reference. Recieved: {projection_osr}"

    geom = convert_bbox_to_geom(bbox_ogr)

    driver = ogr.GetDriverByName("FlatGeobuf")
    extent_name = f"/vsimem/{uuid4().int}_extent.fgb"
    extent_ds = driver.CreateDataSource(extent_name)

    layer = extent_ds.CreateLayer(
        "extent_ogr", projection_osr, ogr.wkbPolygon
    )

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)

    feature = None
    layer.SyncToDisk()

    return extent_name


def get_sub_geotransform(geotransform, bbox_ogr):
    """
    Creates a geotransform for the bounding box.

    returns a dictionary with a geotransform and raster sizes.
    """
    assert is_valid_geotransform(geotransform), \
        f"geotransform must be a valid geotransform. Received: {geotransform}."
    assert is_valid_bbox(bbox_ogr), \
        f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

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


def bboxes_intersect(bbox1_ogr, bbox2_ogr):
    """
    Checks if two bounding boxes intersects.

    returns a dictionary with a geotransform and raster sizes.
    """
    assert is_valid_bbox(bbox1_ogr), \
        f"bbox1_ogr was not a valid bbox. Received: {bbox1_ogr}"
    assert is_valid_bbox(bbox2_ogr), \
        f"bbox1_ogr was not a valid bbox. Received: {bbox2_ogr}"

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
    Checks if two bounding boxes intersects.

    returns a dictionary with a geotransform and raster sizes.
    """
    assert isinstance(bbox1_ogr, list) and len(bbox1_ogr) == 4, \
        f"bbox1_ogr must be a list of length four. Received: {bbox1_ogr}."
    assert isinstance(bbox2_ogr, list) and len(bbox2_ogr) == 4, \
        f"bbox2_ogr must be a list of length four. Received: {bbox2_ogr}."

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
    Get the intersection of two bounding boxes.
    Throws an error if no intersection is found

    Returns an ogr bounding box of the intersection.
    """
    assert bboxes_intersect(bbox1_ogr, bbox2_ogr), \
        "The two bounding boxes do not intersect."

    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    return (
        max(bbox1_x_min, bbox2_x_min),
        min(bbox1_x_max, bbox2_x_max),
        max(bbox1_y_min, bbox2_y_min),
        min(bbox1_y_max, bbox2_y_max),
    )


def align_bboxes_to_pixel_size(bbox1_ogr, bbox2_ogr, pixel_width, pixel_height):
    """
    Aligns two bounding boxes, so the pixel inside are aligned.

    Returns an ogr bounding box.
    """
    assert isinstance(bbox1_ogr, list) and len(bbox1_ogr) == 4, \
        f"bbox1_ogr must be a list of length four. Received: {bbox1_ogr}."
    assert isinstance(bbox2_ogr, list) and len(bbox2_ogr) == 4, \
        f"bbox2_ogr must be a list of length four. Received: {bbox2_ogr}."

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
    Reprojects a bounding box from one projection to another.

    Returns an ogr bounding box.
    """
    assert is_valid_bbox(bbox_ogr), \
        f"Invalid bbox. Received: {bbox_ogr}."

    assert isinstance(source_projection_osr, osr.SpatialReference), \
        f"source_projection not a valid spatial reference. Recieved: {source_projection_osr}"

    assert isinstance(target_projection_osr, osr.SpatialReference), \
        f"target_projection not a valid spatial reference. Recieved: {target_projection_osr}"

    if source_projection_osr.IsSame(target_projection_osr):
        return bbox_ogr

    transformer = osr.CoordinateTransformation(source_projection_osr, target_projection_osr)

    x_min, x_max, y_min, y_max = bbox_ogr

    transformed_x_min, transformed_y_min = transformer.TransformPoint(x_min, y_min)
    transformed_x_max, transformed_y_max = transformer.TransformPoint(x_max, y_max)

    return transformed_x_min, transformed_x_max, transformed_y_min, transformed_y_max



def additional_bboxes(bbox_ogr, projection_osr):
    """
    This is an internal utility functions for metadata generation. Takes a standard
    OGR bounding box and returns a list of variations of bounding boxes.
    """
    assert is_valid_bbox(bbox_ogr), \
        f"Invalid bbox. Received: {bbox_ogr}."
    assert isinstance(projection_osr, osr.SpatialReference), \
        f"source_projection not a valid spatial reference. Recieved: {projection_osr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    original_projection = osr.SpatialReference()
    original_projection.ImportFromWkt(projection_osr.ExportToWkt())

    latlng_projection = osr.SpatialReference()
    latlng_projection.ImportFromEPSG(4326)

    bbox_ogr_latlng = reproject_bbox(bbox_ogr, original_projection, latlng_projection)
    latlng_x_min, latlng_x_max, latlng_y_min, latlng_y_max = bbox_ogr_latlng

    bbox_wkt = convert_bbox_to_wkt(bbox_ogr)
    bbox_wkt_latlng = convert_bbox_to_wkt(bbox_ogr_latlng)

    return {
        "bbox_latlng": bbox_ogr_latlng,
        "bbox_wkt": bbox_wkt,
        "bbox_wkt_latlng": bbox_wkt_latlng,
        "bbox_warp": convert_ogr_bbox_to_gdal_bbox(bbox_ogr),
        "bbox_warp_latlng": convert_ogr_bbox_to_gdal_bbox(bbox_ogr_latlng),
        "bbox_dict": { "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max },
        "bbox_dict_latlng": { "x_min": latlng_x_min, "x_max": latlng_x_max, "y_min": latlng_y_min, "y_max": latlng_y_max },
        "bbox_geojson": convert_bbox_to_geojson(bbox_ogr_latlng),
    }

print("bob")