import osgeo
from osgeo import gdal, ogr, osr
from typing import Sequence, Union, Any, Tuple, List
from uuid import uuid4
import numpy as np
import os
import json

from buteo.project_types import (
    Expanded_extents,
    Number,
)
from buteo.utils import path_to_ext, is_number


def raster_to_reference(
    raster: Union[str, gdal.Dataset], writeable: bool = False
) -> gdal.Dataset:
    """ Takes a file path or a gdal.Dataset and opens it with
        GDAL. Raises exception if the raster cannot be read.

    Args:
        raster (path | Dataset): A path to a raster or a GDAL dataframe.

    **kwargs:
        writeable (bool): Indicates if the opened raster be writeable.

    Returns:
        A GDAL.dataset of the input raster.
    """
    try:
        if isinstance(raster, gdal.Dataset):  # Dataset already GDAL dataframe.
            return raster
        else:
            opened = gdal.Open(raster, 1) if writeable else gdal.Open(raster, 0)

            if opened is None:
                raise Exception("Could not read input raster")

            return opened
    except:
        raise Exception("Could not read input raster")


def vector_to_reference(
    vector: Union[str, ogr.DataSource], writeable: bool = False
) -> ogr.DataSource:
    """ Takes a file path or an ogr.DataSOurce and opens it with
        OGR. Raises exception if the raster cannot be read.

    Args:
        raster (path | DataSource): A path to a vector or an ogr.DataSource.

    **kwargs:
        writeable (bool): Indicates if the opened vector be writeable.

    Returns:
        An OGR.DataSource of the input vector.
    """

    try:
        if isinstance(vector, ogr.DataSource):  # Dataset already OGR dataframe.
            return vector
        else:
            opened = ogr.Open(vector, 1) if writeable else ogr.Open(vector, 0)

            if opened is None:
                raise Exception("Could not read input vector")

            return opened
    except:
        raise Exception("Could not read input vector")


def default_options(options: list) -> list:
    """ Takes a list of GDAL options and adds the following
        defaults to it:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW
        If any of the options are already specified, they are not
        added.

    Args:
        options (List): A list of options (str). Can be empty.

    Returns:
        A list of strings with the default options for a GDAL
        raster.
    """
    internal_options = list(options)

    opt_str = " ".join(internal_options)
    if "TILED" not in opt_str:
        internal_options.append("TILED=YES")

    if "NUM_THREADS" not in opt_str:
        internal_options.append("NUM_THREADS=ALL_CPUS")

    if "BIGTIFF" not in opt_str:
        internal_options.append("BIGTIFF=YES")

    if "COMPRESS" not in opt_str:
        internal_options.append("COMPRESS=LZW")

    return internal_options


def path_to_driver(file_path: str) -> str:
    """ Takes a file path and returns the GDAL driver matching
    the extension.

    Args:
        file_path (path): A file path string

    Returns:
        A string representing the GDAL Driver matching the
        extension. If none is found, None is returned.
    """
    ext = path_to_ext(file_path)

    # Raster formats
    if ext == ".tif" or ext == ".tiff":
        return "GTiff"
    elif ext == ".img":
        return "HFA"
    elif ext == ".jp2":
        return "JP2ECW"
    elif ext == ".ecw":
        return "ECW"

    # Vector formats
    elif ext == ".shp":
        return "ESRI Shapefile"
    elif ext == ".gpkg":
        return "GPKG"
    elif ext == ".fgb":
        return "FlatGeobuf"
    elif ext == ".json" or ext == ".geojson":
        return "GeoJSON"

    else:
        raise ValueError(f"Unable to parse GDAL driver from path: {file_path}")


def is_raster(raster: Union[str, gdal.Dataset]) -> bool:
    """ Takes a string or a gdal.Dataset and returns a boolean
    indicating if it is a valid raster.

    Args:
        file_path (path | Dataset): A path to a raster or a GDAL dataframe.

    Returns:
        A boolean. True if input is a valid raster, false otherwise.
    """
    if isinstance(raster, gdal.Dataset):
        return True

    if isinstance(raster, str):

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        ref = gdal.Open(raster, 0)
        gdal.PopErrorHandler()

        if isinstance(ref, gdal.Dataset):
            ref = None
            return True

    return False


def is_vector(vector: Union[str, ogr.DataSource]) -> bool:
    """ Takes a string or an ogr.DataSource and returns a boolean
    indicating if it is a valid vector.

    Args:
        file_path (path | DataSource): A path to a vector or an ogr DataSource.

    Returns:
        A boolean. True if input is a valid vector, false otherwise.
    """
    if isinstance(vector, ogr.DataSource):
        return True

    if isinstance(vector, ogr.Layer):
        return True

    if isinstance(vector, str):

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        ref = ogr.Open(vector, 0)
        gdal.PopErrorHandler()

        if isinstance(ref, ogr.DataSource):
            ref = None
            return True

    return False


def parse_projection(
    target: Union[str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int],
    return_wkt: bool = False,
) -> Union[osr.SpatialReference, str]:
    """ Parses a gdal, ogr og osr data source and extraction the projection. If
        a string is passed, it attempts to open it and return the projection as
        an osr.SpatialReference.
    Args:
        target (str | gdal.datasource): A gdal data source or the path to one.

    **kwargs:
        return_wkt (bool): Indicates if the function should return a wkt string
        instead of an osr.SpatialReference.

    Returns:
        An osr.SpatialReference matching the input. If return_wkt is true, WKT
        string representing the projection is returned.
    """
    err_msg = f"Unable to parse target projection: {target}"
    target_proj = osr.SpatialReference()

    # Suppress gdal errors and handle them ourselves.
    # This ensures that the console is not flooded.
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    if isinstance(target, ogr.DataSource):
        layer = target.GetLayer()
        target_proj = layer.GetSpatialRef()
    elif isinstance(target, gdal.Dataset):
        target_proj.ImportFromWkt(target.GetProjection())
    elif isinstance(target, osr.SpatialReference):
        target_proj = target
    elif isinstance(target, str):
        ref = gdal.Open(target, 0)

        if ref != None:
            target_proj.ImportFromWkt(ref.GetProjection())
        else:
            ref = ogr.Open(target, 0)

            if ref != None:
                layer = ref.GetLayer()
                target_proj = layer.GetSpatialRef()
            else:
                code = target_proj.ImportFromWkt(target)
                if code != 0:
                    code = target_proj.ImportFromProj4(target)
                    if code != 0:
                        raise ValueError(err_msg)
    elif isinstance(target, int):
        code = target_proj.ImportFromEPSG(target)
        if code != 0:
            raise ValueError(err_msg)
    else:
        raise ValueError(err_msg)

    gdal.PopErrorHandler()

    if isinstance(target_proj, osr.SpatialReference):
        if target_proj.GetName() == None:
            raise ValueError(err_msg)

        if return_wkt:
            return target_proj.ExportToWkt()

        return target_proj
    else:
        raise ValueError(err_msg)


def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)


def translate_resample_method(method):
    methods = {
        "nearest": 0,
        "bilinear": 1,
        "cubic": 2,
        "cubicSpline": 3,
        "lanczos": 4,
        "average": 5,
        "mode": 6,
        "max": 8,
        "min": 9,
        "median": 10,
        "q1": 11,
        "q3": 12,
        "sum": 13,
        "rms": 14,
    }

    if method in methods:
        return methods[method]
    else:
        return 0


def numpy_fill_values(datatype):
    datatypes = {
        "int8": -127,
        "int16": -32767,
        "int32": -2147483647,
        "int64": -9223372036854775807,
        "uint8": 255,
        "uint16": 65535,
        "uint32": 4294967295,
        "uint64": 18446744073709551615,
        "float16": -999999.9,
        "float32": -999999.9,
        "float64": -999999.9,
    }

    if datatype in datatypes:
        return datatypes[datatype]
    else:
        return 0


def gdal_nodata_value_from_type(gdal_datatype_raw):
    if gdal_datatype_raw == 0:
        return 0
    elif gdal_datatype_raw == 1:
        return 255
    elif gdal_datatype_raw == 2:
        return 65535
    elif gdal_datatype_raw == 3:
        return -32767
    elif gdal_datatype_raw == 4:
        return 4294967295
    elif gdal_datatype_raw == 5:
        return -2147483647
    elif gdal_datatype_raw == 6:
        return -9999.0
    elif gdal_datatype_raw == 7:
        return -9999.0
    else:
        return 0


def translate_datatypes(datatype):
    datatypes = {
        "byte": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "cint16": 8,
        "cint32": 9,
        "cfloat32": 10,
        "cfloat64": 11,
    }

    if datatype in datatypes.keys():
        return datatypes[datatype]
    else:
        return 6


def gdal_to_numpy_datatype(gdal_int):
    datatypes = {
        "1": "uint8",
        "2": "uint16",
        "3": "int16",
        "4": "uint32",
        "5": "int32",
        "6": "float32",
        "7": "float64",
        "8": "cint16",
        "9": "cint32",
        "10": "cfloat32",
        "11": "cfloat64",
    }
    return datatypes[str(gdal_int)]


def numpy_to_gdal_datatype(datatype):
    datatypes = {
        "int8": gdal.GDT_Int16,
        "int16": gdal.GDT_Int16,
        "int32": gdal.GDT_Int32,
        "int64": gdal.GDT_Int32,
        "uint8": gdal.GDT_Byte,
        "uint16": gdal.GDT_UInt16,
        "uint32": gdal.GDT_UInt32,
        "uint64": gdal.GDT_UInt32,
        "float16": gdal.GDT_Float32,
        "float32": gdal.GDT_Float32,
        "float64": gdal.GDT_Float64,
    }

    if datatype.name in datatypes.keys():
        return datatypes[datatype.name]
    else:
        return gdal.GDT_Float32


def gdal_datatype_is_float(datatype):
    floats = [6, 7, 10, 11]
    if datatype in floats:
        return True
    else:
        return False


def get_extent(dataframe):
    transform = dataframe.GetGeoTransform()

    bottomRightX = transform[0] + (dataframe.RasterXSize * transform[1])
    bottomRightY = transform[3] + (dataframe.RasterYSize * transform[5])

    #      (   minX,         minY,         maxX,         maxY     )
    return (transform[0], bottomRightY, bottomRightX, transform[3])


def get_intersection(extent1, extent2):
    one_bottomLeftX = extent1[0]
    one_topRightX = extent1[2]
    one_bottomLeftY = extent1[1]
    one_topRightY = extent1[3]

    two_bottomLeftX = extent2[0]
    two_topRightX = extent2[2]
    two_bottomLeftY = extent2[1]
    two_topRightY = extent2[3]

    if two_bottomLeftX > one_topRightX:  # Too far east
        return False
    elif two_bottomLeftY > one_topRightY:  # Too far north
        return False
    elif two_topRightX < one_bottomLeftX:  # Too far west
        return False
    elif two_topRightY < one_bottomLeftY:  # Too far south
        return False
    else:
        return (
            max(one_bottomLeftX, two_bottomLeftX),  # minX of intersection
            max(one_bottomLeftY, two_bottomLeftY),  # minY of intersection
            min(one_topRightX, two_topRightX),  # maxX of intersection
            min(one_topRightY, two_topRightY),  # maxY of intersection
        )


def create_geotransform(geo_transform, extent):
    RasterXSize = round(
        (extent[2] - extent[0]) / geo_transform[1]
    )  # (maxX - minX) / pixelWidth
    RasterYSize = round(
        (extent[3] - extent[1]) / geo_transform[5]
    )  # (maxY - minY) / pixelHeight

    return {
        "Transform": [extent[0], geo_transform[1], 0, extent[3], 0, geo_transform[5]],
        "RasterXSize": abs(RasterXSize),
        "RasterYSize": abs(RasterYSize),
    }


def raster_size_from_list(target_size, target_in_pixels=False):
    x_res = None
    y_res = None

    x_pixels = None
    y_pixels = None

    if target_size is None:
        return x_res, y_res, x_pixels, y_pixels

    if isinstance(target_size, gdal.Dataset) or isinstance(target_size, str):
        reference = (
            target_size
            if isinstance(target_size, gdal.Dataset)
            else gdal.Open(target_size, 0)
        )

        transform = reference.GetGeoTransform()

        x_res = transform[1]
        y_res = abs(transform[5])

    elif target_in_pixels:
        if isinstance(target_size, tuple) or isinstance(target_size, list):
            if len(target_size) == 1:
                if is_number(target_size[0]):
                    x_pixels = int(target_size[0])
                    y_pixels = int(target_size[0])
                else:
                    raise ValueError(
                        "target_size_pixels is not a number or a list/tuple of numbers."
                    )
            elif len(target_size) == 2:
                if is_number(target_size[0]) and is_number(target_size[1]):
                    x_pixels = int(target_size[0])
                    y_pixels = int(target_size[1])
            else:
                raise ValueError("target_size_pixels is either empty or larger than 2.")
        elif is_number(target_size):
            x_pixels = int(target_size)
            y_pixels = int(target_size)
        else:
            raise ValueError("target_size_pixels is invalid.")

        x_res = None
        y_res = None
    else:
        if isinstance(target_size, tuple) or isinstance(target_size, list):
            if len(target_size) == 1:
                if is_number(target_size[0]):
                    x_res = float(target_size[0])
                    y_res = float(target_size[0])
                else:
                    raise ValueError(
                        "target_size is not a number or a list/tuple of numbers."
                    )
            elif len(target_size) == 2:
                if is_number(target_size[0]) and is_number(target_size[1]):
                    x_res = float(target_size[0])
                    y_res = float(target_size[1])
            else:
                raise ValueError("target_size is either empty or larger than 2.")
        elif is_number(target_size):
            x_res = float(target_size)
            y_res = float(target_size)
        else:
            raise ValueError("target_size is invalid.")

        x_pixels = None
        y_pixels = None

    return x_res, y_res, x_pixels, y_pixels


def align_bbox(extent_og, extent_ta, pixel_width, pixel_height, warp_format=True):
    x_min_og, y_max_og, x_max_og, y_min_og = extent_og
    x_min_ta, y_max_ta, x_max_ta, y_min_ta = extent_ta

    x_min = x_min_ta - ((x_min_ta - x_min_og) % pixel_width)
    x_max = x_max_ta + ((x_max_og - x_max_ta) % pixel_width)

    y_min = y_min_ta - ((y_min_ta - y_min_og) % abs(pixel_height))
    y_max = y_max_ta + ((y_max_og - y_max_ta) % abs(pixel_height))

    # gdal_warp format
    if warp_format:
        return (x_min, y_min, x_max, y_max)

    return (x_min, y_max, x_max, y_min)


def advanced_extents(
    extent_ogr: List[Number], projection: osr.SpatialReference
) -> Expanded_extents:
    original_projection = projection
    target_projection = osr.SpatialReference()
    target_projection.ImportFromEPSG(4326)

    if int(osgeo.__version__[0]) >= 3:
        original_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    x_min, x_max, y_min, y_max = extent_ogr

    bottom_left = [x_min, y_min]
    top_left = [x_min, y_max]
    top_right = [x_max, y_max]
    bottom_right = [x_max, y_min]

    coord_array = [
        [bottom_left[1], bottom_left[0]],
        [top_left[1], top_left[0]],
        [top_right[1], top_right[0]],
        [bottom_right[1], bottom_right[0]],
        [bottom_left[1], bottom_left[0]],
    ]

    wkt_coords = ""
    for coord in coord_array:
        wkt_coords += f"{coord[1]} {coord[0]}, "
    wkt_coords = wkt_coords[:-2]  # Remove the last ", "

    extent_wkt = f"POLYGON (({wkt_coords}))"

    extent_name = f"/vsimem/{uuid4().int}_extent.shp"

    driver = ogr.GetDriverByName("ESRI Shapefile")
    extent_ds = driver.CreateDataSource(extent_name)
    layer = extent_ds.CreateLayer(
        extent_name + "_layer", original_projection, ogr.wkbPolygon
    )

    feature = ogr.Feature(layer.GetLayerDefn())
    extent_geom = ogr.CreateGeometryFromWkt(extent_wkt, original_projection)
    feature.SetGeometry(extent_geom)
    layer.CreateFeature(feature)
    feature = None

    if not original_projection.IsSame(target_projection):
        tx = osr.CoordinateTransformation(original_projection, target_projection)

        bottom_left = tx.TransformPoint(x_min, y_min)
        top_left = tx.TransformPoint(x_min, y_max)
        top_right = tx.TransformPoint(x_max, y_max)
        bottom_right = tx.TransformPoint(x_max, y_min)

    extent_latlng = [
        top_left[0],
        top_left[1],
        bottom_right[0],
        bottom_right[1],
    ]

    extent_gdal_warp_latlng = [
        top_left[0],
        bottom_right[1],
        bottom_right[0],
        top_left[1],
    ]

    extent_ogr_latlng = [
        top_left[0],
        bottom_right[0],
        bottom_right[1],
        top_left[1],
    ]

    extent_dict_latlng = {
        "left": top_left[0],
        "top": top_left[1],
        "right": bottom_right[0],
        "bottom": bottom_right[1],
    }

    # WKT has latitude first, geojson has longitude first
    coord_array = [
        [bottom_left[1], bottom_left[0]],
        [top_left[1], top_left[0]],
        [top_right[1], top_right[0]],
        [bottom_right[1], bottom_right[0]],
        [bottom_left[1], bottom_left[0]],
    ]

    wkt_coords = ""
    for coord in coord_array:
        wkt_coords += f"{coord[1]} {coord[0]}, "
    wkt_coords = wkt_coords[:-2]  # Remove the last ", "

    extent_wkt_latlng = f"POLYGON (({wkt_coords}))"

    # Create an OGR Datasource in memory with the extent
    extent_name = f"/vsimem/{uuid4().int}_extent_latlng.shp"

    driver = ogr.GetDriverByName("ESRI Shapefile")
    extent_ds_latlng = driver.CreateDataSource(extent_name)
    layer = extent_ds_latlng.CreateLayer(
        extent_name + "_layer", target_projection, ogr.wkbPolygon
    )

    feature = ogr.Feature(layer.GetLayerDefn())
    extent_geom_latlng = ogr.CreateGeometryFromWkt(extent_wkt_latlng, target_projection)
    feature.SetGeometry(extent_geom_latlng)
    layer.CreateFeature(feature)
    feature = None

    # We don't define a geojson in the original projection as geojson is usually expected to be latlng.
    extent_geojson_dict = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [coord_array],},
    }
    extent_geojson = json.dumps(extent_geojson_dict)

    expanded_extents: Expanded_extents = {
        "extent_wkt": extent_wkt,
        "extent_datasource": extent_ds,
        "extent_geom": extent_geom,
        "extent_latlng": extent_latlng,
        "extent_gdal_warp_latlng": extent_gdal_warp_latlng,
        "extent_ogr_latlng": extent_ogr_latlng,
        "extent_dict_latlng": extent_dict_latlng,
        "extent_wkt_latlng": extent_wkt_latlng,
        "extent_datasource_latlng": extent_ds_latlng,
        "extent_geom_latlng": extent_geom_latlng,
        "extent_geojson": extent_geojson,
        "extent_geojson_dict": extent_geojson_dict,
    }

    return expanded_extents


# x_min, x_max, y_min, y_max
def ogr_bbox_within(bbox1: Sequence[Number], bbox2: Sequence[Number]) -> bool:
    return (
        (bbox1[0] >= bbox2[0])
        and (bbox1[1] <= bbox2[1])
        and (bbox1[3] >= bbox2[3])
        and (bbox1[4] <= bbox2[4])
    )


# x_min, y_max, x_max, y_min
def gdal_bbox_within(bbox1: Sequence[Number], bbox2: Sequence[Number]) -> bool:
    return (
        (bbox1[0] >= bbox2[0])
        and (bbox1[1] <= bbox2[1])
        and (bbox1[3] <= bbox2[3])
        and (bbox1[4] >= bbox2[4])
    )


# x_min, x_max, y_min, y_max
def ogr_bbox_intersects(bbox1: Sequence[Number], bbox2: Sequence[Number]) -> bool:
    return (
        (bbox1[0] <= bbox2[1])
        and (bbox2[0] <= bbox1[1])
        and (bbox1[2] <= bbox2[3])
        and (bbox2[2] <= bbox1[3])
    )


# x_min, y_max, x_max, y_min
def gdal_bbox_intersects(bbox1: Sequence[Number], bbox2: Sequence[Number]) -> bool:
    return (
        (bbox1[0] <= bbox2[2])
        and (bbox2[0] <= bbox1[2])
        and (bbox1[3] <= bbox2[1])
        and (bbox2[3] <= bbox1[1])
    )


def reproject_extent(
    extent: Sequence[Number],
    source_projection: osr.SpatialReference,
    target_projection: osr.SpatialReference,  # x_min, y_max, x_max, y_min
):
    """ GDAL Format """
    if len(extent) != 4:
        raise ValueError("Invalid shape of extent.")
    if not isinstance(extent[0], float):
        raise ValueError("x_min not float.")
    if not isinstance(extent[1], float):
        raise ValueError("y_max not float.")
    if not isinstance(extent[2], float):
        raise ValueError("x_max not float.")
    if not isinstance(extent[3], float):
        raise ValueError("y_min not float.")

    if not isinstance(source_projection, osr.SpatialReference):
        raise ValueError("source_projection not a valid spatial reference.")

    if not isinstance(target_projection, osr.SpatialReference):
        raise ValueError("target_projection not a valid spatial reference.")

    tx = osr.CoordinateTransformation(source_projection, target_projection)

    top_left = tx.TransformPoint(extent[0], extent[1])
    bottom_right = tx.TransformPoint(extent[2], extent[3])

    return [
        top_left[0],
        top_left[1],
        bottom_right[0],
        bottom_right[1],
    ]


# TODO: Verify folder exists.
def to_path_list(variable: Any) -> List[str]:
    return_list = []
    if isinstance(variable, list):
        return_list = variable
    else:
        return_list.append(variable)

    if len(return_list) == 0:
        raise ValueError("Empty array list.")

    for path in return_list:
        if not isinstance(path, str):
            raise ValueError(f"Invalid string in  path list: {variable}")

    return return_list


def to_array_list(variable: Any) -> List[Union[List, np.ndarray]]:
    return_list = []
    if isinstance(variable, list):
        return_list = variable
    else:
        return_list.append(variable)

    if len(return_list) == 0:
        raise ValueError("Empty array list.")

    for array in return_list:
        if not isinstance(array, np.ndarray):
            if isinstance(array, str) and os.path.exists(array):
                try:
                    _ = np.load(array)
                except:
                    raise ValueError(f"Invalid array in list: {array}")
        else:
            raise ValueError(f"Invalid array in list: {array}")

    return return_list


def to_band_list(variable: Any, band_count: int,) -> List[int]:
    return_list = []
    if not isinstance(variable, (int, float, list)):
        raise TypeError(f"Invalid type for band: {type(variable)}")

    if isinstance(variable, list):
        if len(variable) == 0:
            raise ValueError("Provided list of bands is empty.")
        for val in variable:
            try:
                band_int = int(val)
            except:
                raise ValueError(
                    f"List of bands contained non-valid band number: {val}"
                )

            if band_int > band_count - 1:
                raise ValueError("Requested a higher band that is available in raster.")
            else:
                return_list.append(band_int)
    elif variable == -1:
        for val in range(band_count):
            return_list.append(val)
    else:
        if variable > band_count + 1:
            raise ValueError("Requested a higher band that is available in raster.")
        else:
            return_list.append(int(variable))

    return return_list

