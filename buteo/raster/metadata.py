"""
### Read metadata from rasters ###
"""
# Standard library
import sys; sys.path.append("../../")
import os
from typing import List, Union

# External
from osgeo import gdal, osr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_translate,
    utils_projection,
)
from buteo.raster.core_raster import _raster_open


def _raster_to_metadata(
    raster: Union[str, gdal.Dataset],
) -> dict:
    """ Internal. """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")

    dataset = _raster_open(raster)

    raster_driver = dataset.GetDriver()

    path = dataset.GetDescription()
    basename = os.path.basename(path)
    split_path = os.path.splitext(basename)
    name = split_path[0]
    ext = split_path[1]

    driver = raster_driver.ShortName

    in_memory = utils_gdal._check_is_dataset_in_memory(raster)

    transform = dataset.GetGeoTransform()

    projection_wkt = dataset.GetProjection()
    projection_osr = osr.SpatialReference()
    projection_osr.ImportFromWkt(projection_wkt)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    band_count = dataset.RasterCount

    size = [dataset.RasterXSize, dataset.RasterYSize]
    shape = (height, width, band_count)

    pixel_width = abs(transform[1])
    pixel_height = abs(transform[5])

    x_min = transform[0]
    y_max = transform[3]

    x_max = x_min + width * transform[1] + height * transform[2]  # Handle skew
    y_min = y_max + width * transform[4] + height * transform[5]  # Handle skew

    band0 = dataset.GetRasterBand(1)

    dtype_gdal = band0.DataType
    datatype_gdal = gdal.GetDataTypeName(dtype_gdal)

    datatype = utils_translate._translate_dtype_gdal_to_numpy(dtype_gdal).name

    nodata_value = band0.GetNoDataValue()
    has_nodata = nodata_value is not None

    bbox_ogr = [x_min, x_max, y_min, y_max]

    bboxes = utils_bbox._additional_bboxes(bbox_ogr, projection_osr)

    metadata = {
        "path": path,
        "basename": basename,
        "name": name,
        "ext": ext,
        "transform": transform,
        "in_memory": in_memory,
        "projection_wkt": projection_wkt,
        "projection_osr": projection_osr,
        "width": width,
        "height": height,
        "band_count": band_count,
        "driver": driver,
        "size": size,
        "shape": shape,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "x_min": x_min,
        "y_max": y_max,
        "x_max": x_max,
        "y_min": y_min,
        "dtype": datatype,
        "dtype_gdal": datatype_gdal,
        "dtype_gdal_raw": dtype_gdal,
        "datatype": datatype,
        "datatype_gdal": datatype_gdal,
        "dtype_gdal": dtype_gdal,
        "nodata_value": nodata_value,
        "has_nodata": has_nodata,
        "is_raster": True,
        "is_vector": False,
        "bbox": bbox_ogr,
        "extent": bbox_ogr,
    }

    for key, value in bboxes.items():
        metadata[key] = value

    def get_bbox_as_vector():
        return utils_bbox._get_vector_from_bbox(bbox_ogr, projection_osr)

    def get_bbox_as_vector_latlng():
        latlng_wkt = utils_projection._get_default_projection()
        projection_osr_latlng = osr.SpatialReference()
        projection_osr_latlng.ImportFromWkt(latlng_wkt)

        return utils_bbox._get_vector_from_bbox(metadata["bbox_latlng"], projection_osr_latlng)

    metadata["get_bbox_vector"] = get_bbox_as_vector
    metadata["get_bbox_vector_latlng"] = get_bbox_as_vector_latlng

    return metadata


def raster_to_metadata(
    raster: Union[str, gdal.Dataset, List[str], List[gdal.Dataset]],
    *,
    allow_lists: bool = True,
) -> Union[dict, List[dict]]:
    """
    Reads metadata from a raster dataset or a list of raster datasets, and returns a dictionary or a list of dictionaries
    containing metadata information for each raster.

    Parameters
    ----------
    raster : str or gdal.Dataset or list
        A path to a raster or a gdal.Dataset, or a list of paths to rasters.

    allow_lists : bool, optional
        If True, allows the input to be a list of rasters. Otherwise, only a single raster is allowed. Default: True.

    Returns
    -------
    dict or list of dict
        A dictionary or a list of dictionaries containing metadata information for each raster.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    if not allow_lists and isinstance(raster, list):
        raise ValueError("Input raster must be a single raster.")

    if not allow_lists:
        return _raster_to_metadata(raster)

    list_input = utils_base._get_variable_as_list(raster)
    list_return = []

    for in_raster in list_input:
        list_return.append(_raster_to_metadata(in_raster))

    if isinstance(raster, list):
        return list_return

    return list_return[0]
