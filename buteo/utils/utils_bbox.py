"""
### Bounding box utility functions ###

Various utility functions to work with bounding boxes and gdal.

There are two different formats for bounding boxes used by GDAL:</br>
OGR:  `[x_min, x_max, y_min, y_max]`</br>
WARP: `[x_min, y_min, x_max, y_max]`</br>

_If nothing else is stated, the OGR format is used._

The GDAL geotransform is a list of six parameters:</br>
`x_min, pixel_width, row_skew, y_max, column_skew, pixel_height (negative for north-up)`
"""

# Standard library
import sys; sys.path.append("../../")
from typing import List, Union, Dict, Any
from uuid import uuid4

# External
import numpy as np
from osgeo import ogr, osr, gdal

# Internal
from buteo.utils import utils_base, utils_gdal_projection



def _check_is_valid_bbox(bbox_ogr: List[Union[int, float]]) -> bool:
    """
    Checks if a bbox is valid.

    A valid ogr formatted bbox has the form:
        `[x_min, x_max, y_min, y_max]`

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    bool
        True if the bbox is valid, False otherwise.
    """
    if not isinstance(bbox_ogr, list):
        return False

    if len(bbox_ogr) != 4:
        return False

    for val in bbox_ogr:
        if not utils_base._check_variable_is_number_type(val):
            return False

    x_min, x_max, y_min, y_max = bbox_ogr

    if True in [np.isinf(val) for val in bbox_ogr]:
        return True

    if x_min > x_max or y_min > y_max:
        return False

    return True


def _check_is_valid_bbox_latlng(bbox_ogr_latlng: List[Union[int, float]]) -> bool:
    """
    Checks if a bbox is valid and latlng.

    A valid ogr formatted bbox has the form:
        `[x_min, x_max, y_min, y_max]`

    Parameters
    ----------
    bbox_ogr_latlng : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
        
    Returns
    -------
    bool
        True if the bbox is valid, False otherwise.
    """
    if not _check_is_valid_bbox(bbox_ogr_latlng):
        return False

    x_min, x_max, y_min, y_max = bbox_ogr_latlng
    if x_min < -180 or x_max > 180 or y_min < -90 or y_max > 90:
        return False

    return True


def _check_is_valid_geotransform(geotransform: List[Union[int, float]]) -> bool:
    """
    Checks if a geotransform is valid.

    A valid geotransform has the form:
    `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`

    Parameters
    ----------
    geotransform : list
        A GDAL formatted geotransform.
        `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`
        pixel_height is negative for north up.

    Returns
    -------
    bool
        True if the geotransform is valid, False otherwise.
    """
    if not isinstance(geotransform, (list, tuple)):
        return False

    if len(geotransform) != 6:
        return False

    for val in geotransform:
        if not utils_base._check_variable_is_number_type(val):
            return False

    return True


def _get_pixel_offsets(
    geotransform: List[Union[int, float]],
    bbox_ogr: List[Union[int, float]],
):
    """
    Get the pixels offsets for a bbox and a geotransform.

    Parameters
    ----------
    geotransform : list
        A GDAL formatted geotransform.
        `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`
        pixel_height is negative for north up.

    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    list
        A list of pixel offsets. `[x_start, y_start, x_size, y_size]`
    """
    assert _check_is_valid_geotransform(
        geotransform
    ), f"geotransform must be a list of length six. Received: {geotransform}."
    assert _check_is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr must be a valid OGR formatted bbox. Received: {bbox_ogr}."

    x_min, x_max, y_min, y_max = bbox_ogr

    origin_x = geotransform[0]
    origin_y = geotransform[3]
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    x_start = int(np.rint((x_min - origin_x) / pixel_width))
    y_start = int(np.rint((y_max - origin_y) / pixel_height))
    x_size = int(np.rint((x_max - x_min) / pixel_width))
    y_size = int(np.rint((y_min - y_max) / pixel_height))

    return [x_start, y_start, x_size, y_size]


def _get_bbox_from_geotransform(
    geotransform: List[Union[int, float]],
    raster_x_size: int,
    raster_y_size: int,
) -> List[Union[int, float]]:
    """
    Get an OGR bounding box from a geotransform and raster sizes.

    Parameters
    ----------
    geotransform : list
        A GDAL formatted geotransform.
        `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`

    raster_x_size : int
        The number of pixels in the x direction.

    raster_y_size : int
        The number of pixels in the y direction.

    Returns
    -------
    list
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert _check_is_valid_geotransform(
        geotransform
    ), f"geotransform was not a valid geotransform. Received: {geotransform}"

    x_min, pixel_width, _row_skew, y_max, _column_skew, pixel_height = geotransform

    x_max = x_min + (raster_x_size * pixel_width)
    y_min = y_max + (raster_y_size * pixel_height)

    return [x_min, x_max, y_min, y_max]


def _get_bbox_from_raster(
    raster_dataframe: gdal.Dataset,
) -> List[Union[int, float]]:
    """
    Gets an OGR bounding box from a GDAL raster dataframe.

    Parameters
    ----------
    raster_dataframe : gdal.Dataset
        A GDAL raster dataframe.

    Returns
    -------
    list
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert isinstance(
        raster_dataframe, gdal.Dataset
    ), f"raster_dataframe was not a gdal.Datasource. Received: {raster_dataframe}"

    bbox = _get_bbox_from_geotransform(
        raster_dataframe.GetGeoTransform(),
        raster_dataframe.RasterXSize,
        raster_dataframe.RasterYSize,
    )

    return bbox


def _get_bbox_from_vector(
    vector_dataframe: ogr.DataSource,
) -> List[Union[int, float]]:
    """
    Gets an OGR bounding box from an OGR dataframe.

    Parameters
    ----------
    vector_dataframe : ogr.DataSource
        An OGR vector dataframe.

    Returns
    -------
    list
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
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


def _get_bbox_from_vector_layer(
    vector_layer: ogr.Layer,
) -> List[Union[int, float]]:
    """
    Gets an OGR bounding box from an OGR dataframe layer.

    Parameters
    ----------
    vector_layer : ogr.Layer
        An OGR vector dataframe layer.

    Returns
    -------
    list
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert isinstance(
        vector_layer, ogr.Layer
    ), f"vector_layer was not a valid ogr.Layer. Received: {vector_layer}"

    x_min, x_max, y_min, y_max = vector_layer.GetExtent()

    return [x_min, x_max, y_min, y_max]


def _get_bbox_from_dataset(
    dataset: Union[str, gdal.Dataset, ogr.DataSource],
) -> List[Union[int, float]]:
    """
    Get the bbox from a dataset.

    Parameters
    ----------
    dataset : str, gdal.Dataset, ogr.DataSource
        A dataset or dataset path.

    Returns
    -------
    list
        The bounding box in ogr format: `[x_min, x_max, y_min, y_max]`.
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
            raise RuntimeError(f"Could not open dataset. {dataset}")

    if isinstance(opened, gdal.Dataset):
        return _get_bbox_from_raster(opened)

    if isinstance(opened, ogr.DataSource):
        return _get_bbox_from_vector(opened)

    raise RuntimeError(f"Could not get bbox from dataset. {dataset}")


def _get_sub_geotransform(
    geotransform: List[Union[int, float]],
    bbox_ogr: List[Union[int, float]],
) -> Dict:
    """
    Create a GeoTransform and the raster sizes for an OGR formatted bbox.

    Parameters
    ----------
    geotransform : list
        A GDAL geotransform.
        `[top_left_x, pixel_width, rotation_x, top_left_y, rotation_y, pixel_height]`
    
    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    dict
        { "Transform": _list_, "RasterXSize": _int_, "RasterYSize": _int_ }
    """
    assert _check_is_valid_geotransform(
        geotransform
    ), f"geotransform must be a valid geotransform. Received: {geotransform}."
    assert _check_is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr
    pixel_width = geotransform[1]
    pixel_height = geotransform[5]

    raster_x_size = round((x_max - x_min) / pixel_width)
    raster_y_size = round((y_max - y_min) / pixel_height)

    return {
        "Transform": [x_min, pixel_width, 0, y_max, 0, utils_base._ensure_negative(pixel_height)],
        "RasterXSize": abs(raster_x_size),
        "RasterYSize": abs(raster_y_size),
    }


def _get_geom_from_bbox(
    bbox_ogr: List[Union[int, float]],
) -> ogr.Geometry:
    """
    Convert an OGR bounding box to ogr.Geometry.
    `[x_min, x_max, y_min, y_max] -> ogr.Geometry`

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    ogr.Geometry
        An OGR geometry.
    """
    assert _check_is_valid_bbox(
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


def _get_bbox_from_geom(geom: ogr.Geometry) -> List[Union[int, float]]:
    """
    Convert an ogr.Geometry to an OGR bounding box.
    `ogr.Geometry -> [x_min, x_max, y_min, y_max]`

    Parameters
    ----------
    geom : ogr.Geometry
        An OGR geometry.

    Returns
    -------
    list
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    assert isinstance(
        geom, ogr.Geometry
    ), f"geom was not a valid ogr.Geometry. Received: {geom}"

    bbox_ogr = list(geom.GetEnvelope()) # [x_min, x_max, y_min, y_max]

    return bbox_ogr


def _get_geotransform_from_bbox(
    bbox_ogr: List[Union[int, float]],
    raster_x_size: int,
    raster_y_size: int,
) -> List[Union[int, float]]:
    """
    Convert an OGR formatted bounding box to a GDAL GeoTransform.
    `[x_min, x_max, y_min, y_max] -> [x_min, pixel_width, x_skew, y_max, y_skew, pixel_height]`

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    
    raster_x_size : int
        The number of pixels in the x direction.

    raster_y_size : int
        The number of pixels in the y direction.

    Returns
    -------
    list
        A GDAL GeoTransform. `[x_min, pixel_width, x_skew, y_max, y_skew, pixel_height]`
    """
    assert _check_is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    origin_x = x_min
    origin_y = y_max
    pixel_width = (x_max - x_min) / raster_x_size
    pixel_height = (y_max - y_min) / raster_y_size

    return [origin_x, pixel_width, 0, origin_y, 0, utils_base._ensure_negative(pixel_height)]


def _get_gdal_bbox_from_ogr_bbox(
    bbox_ogr: List[Union[int, float]],
) -> List[Union[int, float]]:
    """
    Converts an OGR formatted bbox to a GDAL formatted one.
    `[x_min, x_max, y_min, y_max] -> [x_min, y_min, x_max, y_max]`

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.

    Returns
    -------
    list
        A GDAL formatted bbox. `[x_min, y_min, x_max, y_max]`
    """
    if isinstance(bbox_ogr, tuple):
        bbox_ogr = [val for val in bbox_ogr]

    assert _check_is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    return [x_min, y_min, x_max, y_max]


def _get_ogr_bbox_from_gdal_bbox(
    bbox_gdal: List[Union[int, float]],
) -> List[Union[int, float]]:
    """
    Converts a GDAL formatted bbox to an OGR formatted one.
    `[x_min, y_min, x_max, y_max] -> [x_min, x_max, y_min, y_max]`

    Parameters
    ----------
    bbox_gdal : list
        A GDAL formatted bbox.

    Returns
    -------
    list
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`
    """
    if isinstance(bbox_ogr, tuple):
        bbox_ogr = [val for val in bbox_ogr]

    assert (
        isinstance(bbox_gdal, list) and len(bbox_gdal) == 4
    ), f"bbox_gdal must be a list of length four. Received: {bbox_gdal}."

    x_min, y_min, x_max, y_max = bbox_gdal

    return [x_min, x_max, y_min, y_max]


def _get_wkt_from_bbox(
    bbox_ogr: List[Union[int, float]],
) -> str:
    """
    Converts an OGR formatted bbox to a WKT string.
    `[x_min, x_max, y_min, y_max] -> WKT`

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.

    Returns
    -------
    str
        A WKT string. `POLYGON ((...))`
    """
    assert _check_is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    wkt = f"POLYGON (({x_min} {y_min}, {x_max} {y_min}, {x_max} {y_max}, {x_min} {y_max}, {x_min} {y_min}))"

    return wkt


def _get_geojson_from_bbox(
    bbox_ogr: List[Union[int, float]],
) -> Dict[str, Union[str, List]]:
    """
    Converts an OGR formatted bbox to a GeoJson dictionary.
    `[x_min, x_max, y_min, y_max] -> GeoJson`

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    
    Returns
    -------
    dict
        A GeoJson dictionary. `{ "type": "Polygon", "coordinates": [ ... ] }`
    """
    assert _check_is_valid_bbox(
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


def _get_vector_from_bbox(
    bbox_ogr: List[Union[int, float]],
    projection_osr: osr.SpatialReference,
) -> ogr.DataSource:
    """
    Converts an OGR formatted bbox to an in-memory vector.
    _Vectors are stored in /vsimem/ as .gpkg files.
    **OBS**: Layers should be manually cleared when no longer used.

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    
    projection_osr : osr.SpatialReference
        The projection of the vector.

    Returns
    -------
    ogr.DataSource
        The bounding box as a vector.
    """
    assert _check_is_valid_bbox(
        bbox_ogr
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr}"
    assert isinstance(
        projection_osr, osr.SpatialReference
    ), f"projection_osr not a valid spatial reference. Recieved: {projection_osr}"

    geom = _get_geom_from_bbox(bbox_ogr)

    driver = ogr.GetDriverByName("GPKG")
    extent_name = f"/vsimem/{str(uuid4().int)}_extent.gpkg"

    extent_ds = driver.CreateDataSource(extent_name)

    layer = extent_ds.CreateLayer("extent_ogr", projection_osr, ogr.wkbPolygon)

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)

    feature = None
    layer.SyncToDisk()

    return extent_name


def _check_bboxes_intersect(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]],
) -> bool:
    """
    Checks if two OGR formatted bboxes intersect.

    Parameters
    ----------
    bbox1_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    bbox2_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    bool
        **True** if the bboxes intersect, **False** otherwise.
    """
    assert _check_is_valid_bbox(
        bbox1_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox1_ogr}"
    assert _check_is_valid_bbox(
        bbox2_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox2_ogr}"

    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    if bbox2_x_min > bbox1_x_max:
        return False

    if bbox2_y_min > bbox1_y_max:
        return False

    if bbox2_x_max < bbox1_x_min:
        return False

    if bbox2_y_max < bbox1_y_min:
        return False

    return True


def _check_bboxes_within(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]],
) -> bool:
    """
    Checks if one OGR formatted bbox is within another.

    Parameters
    ----------
    bbox1_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    bbox2_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    bool
        **True** if the bbox is within the other, **False** otherwise.
    """
    assert _check_is_valid_bbox(
        bbox1_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox1_ogr}"
    assert _check_is_valid_bbox(
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


def _get_intersection_bboxes(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]],
) -> List[Union[int, float]]:
    """
    Get the intersection of two OGR formatted bboxes.

    Parameters
    ----------
    bbox1_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    
    bbox2_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    
    Returns
    -------
    list
        An OGR formatted bbox of the intersection.
    """
    assert _check_bboxes_intersect(
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


def _get_union_bboxes(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]],
) -> List[Union[int, float]]:
    """
    Get the union of two OGR formatted bboxes.

    Parameters
    ----------
    bbox1_ogr : list
        An OGR formatted bbox.

    bbox2_ogr : list
        An OGR formatted bbox.

    Returns
    -------
    list
        An OGR formatted bbox of the union.
    """
    assert _check_is_valid_bbox(
        bbox1_ogr
    ), f"bbox1_ogr was not a valid bbox. Received: {bbox1_ogr}"
    assert _check_is_valid_bbox(
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


def _get_aligned_bbox_to_pixel_size(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]],
    pixel_width: Union[int, float],
    pixel_height: Union[int, float],
) -> List[Union[int, float]]:
    """
    Aligns two OGR formatted bboxes to a pixel size. Output is an augmented version
    of bbox2.

    Parameters
    ----------
    bbox1_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    
    bbox2_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    pixel_width : int or float
        The width of the pixel.
    
    pixel_height : int or float
        The height of the pixel.

    Returns
    -------
    list
        An OGR formatted bbox of the alignment.
        `[x_min, x_max, y_min, y_max]`
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


def _get_geometry_latlng_from_bbox(
    bbox_ogr: List[Union[int, float]],
    projection_osr: osr.SpatialReference,
    latlng_projection_osr: osr.SpatialReference,
) -> ogr.Geometry:
    """ 
    This is an internal utility function for metadata generation. It takes a standard
    OGR bounding box and the geometry of the source dataset in latlng and returns a
    geometry in the source dataset's projection. This is important as, as when 
    reprojecting to latlng, you might get a skewed bounding box. This function returns
    the skewed bounds in latlng, which is what you want for overlap checks across
    projections.

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    projection_osr : osr.SpatialReference
        The projection.

    latlng_projection_osr : osr.SpatialReference
        The latlng projection.

    Returns
    -------
    ogr.Geometry
        The geometry in the source dataset's projection.
    """
    transformer = osr.CoordinateTransformation(projection_osr, latlng_projection_osr)

    x_min, x_max, y_min, y_max = bbox_ogr

    p1 = [x_min, y_min]
    p2 = [x_max, y_min]
    p3 = [x_max, y_max]
    p4 = [x_min, y_max]

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        p1t = transformer.TransformPoint(p1[0], p1[1])
        p2t = transformer.TransformPoint(p2[0], p2[1])
        p3t = transformer.TransformPoint(p3[0], p3[1])
        p4t = transformer.TransformPoint(p4[0], p4[1])
    except RuntimeError:
        p1t = transformer.TransformPoint(float(p1[0]), float(p1[1]))
        p2t = transformer.TransformPoint(float(p2[0]), float(p2[1]))
        p3t = transformer.TransformPoint(float(p3[0]), float(p3[1]))
        p4t = transformer.TransformPoint(float(p4[0]), float(p4[1]))
    gdal.PopErrorHandler()

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(p1t[0], p1t[1])
    ring.AddPoint(p2t[0], p2t[1])
    ring.AddPoint(p3t[0], p3t[1])
    ring.AddPoint(p4t[0], p4t[1])
    ring.AddPoint(p1t[0], p1t[1])
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)

    return polygon


def _get_utm_zone_from_bbox(
    bbox_ogr_latlng: List[Union[int, float]],
) -> osr.SpatialReference:
    """
    Get the UTM zone from an OGR formatted bbox.

    Parameters
    ----------
    bbox_ogr_latlng : list
        An OGR formatted bbox.

    Returns
    -------
    osr.SpatialReference
        The UTM zone projection.
    """
    assert _check_is_valid_bbox(
        bbox_ogr_latlng
    ), f"bbox_ogr was not a valid bbox. Received: {bbox_ogr_latlng}"
    assert _check_is_valid_bbox_latlng(bbox_ogr_latlng), "Bbox is not in latlng format."

    bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max = bbox_ogr_latlng

    mid_lng = (bbox_x_min + bbox_x_max) / 2
    mid_lat = (bbox_y_min + bbox_y_max) / 2

    return utils_gdal_projection._get_utm_zone_from_latlng([mid_lat, mid_lng])


def _get_utm_zone_from_dataset(
    dataset: Union[str, gdal.Dataset, ogr.DataSource],
) -> str:
    """
    Get the UTM zone from a GDAL dataset.

    Parameters
    ----------
    dataset : str or gdal.Dataset or ogr.DataSource
        A GDAL dataset.

    Returns
    -------
    str
        The UTM zone.
    """
    assert isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)), "dataset was not a valid dataset."

    bbox = _get_bbox_from_dataset(dataset)
    source_projection = utils_gdal_projection._get_projection_from_dataset(dataset)
    target_projection = osr.SpatialReference()
    target_projection_wkt = utils_gdal_projection._get_default_projection()
    target_projection.ImportFromWkt(target_projection_wkt)

    bbox = utils_gdal_projection.reproject_bbox(bbox, source_projection, target_projection)
    utm_zone = _get_utm_zone_from_bbox(bbox)

    return utm_zone


def _get_utm_zone_from_dataset_list(
    datasets: List[Union[str, gdal.Dataset, ogr.DataSource]],
) -> str:
    """
    Get the UTM zone from a list of GDAL datasets.

    Parameters
    ----------
    datasets : list
        A list of GDAL datasets.

    Returns
    -------
    str
        The UTM zone.
    """
    assert isinstance(datasets, (list, np.ndarray)), "datasets was not a valid list."

    latlng_bboxes = []
    latlng_proj = osr.SpatialReference()
    # latlng_proj.ImportFromEPSG(4326)
    latlng_proj.ImportFromWkt('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]')

    for dataset in datasets:
        bbox = _get_bbox_from_dataset(dataset)
        projection = utils_gdal_projection._get_projection_from_dataset(dataset)
        latlng_bboxes.append(utils_gdal_projection.reproject_bbox(bbox, projection, latlng_proj))

    union_bbox = latlng_bboxes[0]
    for bbox in latlng_bboxes[1:]:
        union_bbox = _get_union_bboxes(union_bbox, bbox)

    utm_zone = _get_utm_zone_from_bbox(union_bbox)

    return utm_zone


def _additional_bboxes(
    bbox_ogr: List[Union[int, float]],
    projection_osr: osr.SpatialReference,
) -> Dict[str, Any]:
    """
    This is an internal utility function for metadata generation. It takes a standard
    OGR bounding box and returns a list of variations of bounding boxes.

    Parameters
    ----------
    bbox_ogr : list
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    projection_osr : osr.SpatialReference
        The projection.

    Returns
    -------
    dict
        A dictionary of the added bboxes. Contains the following keys:
        `bbox_latlng`: The bbox in latlng coordinates.
        `bbox_wkt`: The bbox in WKT format.
        `bbox_wkt_latlng`: The bbox in WKT format in latlng coordinates.
        `bbox_geom`: The bbox in ogr.Geometry format.
        `bbox_geom_latlng`: The bbox in ogr.Geometry format in latlng coordinates.
        `bbox_gdal`: The bbox in GDAL format.
        `bbox_gdal_latlng`: The bbox in GDAL format in latlng coordinates.
        `bbox_dict`: The bbox in a dictionary format.
        `bbox_dict_latlng`: The bbox in a dictionary format in latlng coordinates.
    """
    assert _check_is_valid_bbox(bbox_ogr), f"Invalid bbox. Received: {bbox_ogr}."
    assert isinstance(
        projection_osr, osr.SpatialReference
    ), f"source_projection not a valid spatial reference. Recieved: {projection_osr}"

    x_min, x_max, y_min, y_max = bbox_ogr

    original_projection = osr.SpatialReference()
    original_projection.ImportFromWkt(projection_osr.ExportToWkt())

    latlng_projection = osr.SpatialReference()
    wgs84_wkt = utils_gdal_projection._get_default_projection()
    latlng_projection.ImportFromWkt(wgs84_wkt)

    bbox_ogr_latlng = utils_gdal_projection.reproject_bbox(bbox_ogr, original_projection, latlng_projection)

    world = False
    if np.isinf(bbox_ogr_latlng).any():
        world = True
    if np.isinf(bbox_ogr_latlng[0]):
        bbox_ogr_latlng[0] = -179.999999
    if np.isinf(bbox_ogr_latlng[1]):
        bbox_ogr_latlng[1] = 180.0
    if np.isinf(bbox_ogr_latlng[2]):
        bbox_ogr_latlng[2] = -89.999999
    if np.isinf(bbox_ogr_latlng[3]):
        bbox_ogr_latlng[3] = 90.0

    latlng_x_min, latlng_x_max, latlng_y_min, latlng_y_max = bbox_ogr_latlng

    bbox_geom = _get_geom_from_bbox(bbox_ogr)
    bbox_geom_latlng = _get_geom_from_bbox(bbox_ogr_latlng)
    geom = bbox_geom

    if world:
        geom_latlng = bbox_geom_latlng
    else:
        geom_latlng = _get_geometry_latlng_from_bbox(bbox_ogr, original_projection, latlng_projection)

    bbox_wkt = _get_wkt_from_bbox(bbox_ogr)
    bbox_wkt_latlng = _get_wkt_from_bbox(bbox_ogr_latlng)

    return {
        "bbox_latlng": bbox_ogr_latlng,
        "bbox_wkt": bbox_wkt,
        "bbox_wkt_latlng": bbox_wkt_latlng,
        "bbox_geom": bbox_geom,
        "bbox_geom_latlng": bbox_geom_latlng,
        "bbox_gdal": _get_gdal_bbox_from_ogr_bbox(bbox_ogr),
        "bbox_gdal_latlng": _get_gdal_bbox_from_ogr_bbox(bbox_ogr_latlng),
        "bbox_dict": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
        "bbox_dict_latlng": {
            "x_min": latlng_x_min,
            "x_max": latlng_x_max,
            "y_min": latlng_y_min,
            "y_max": latlng_y_max,
        },
        "bbox_geojson": _get_geojson_from_bbox(bbox_ogr_latlng),
        "area_latlng": geom_latlng.GetArea(),
        "area": geom.GetArea(),
        "geom": geom,
        "geom_latlng": geom_latlng,
    }
