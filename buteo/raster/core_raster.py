"""
### Basic functionality for working with rasters. ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import List, Optional, Union, Dict, Any
import warnings

# External
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
    utils_translate,
    utils_projection,
    utils_io,
)



def _raster_open(
    raster: Union[str, gdal.Dataset],
    *,
    writeable: bool = True,
) -> gdal.Dataset:
    """
    Opens a raster in read or write mode.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        A path to a raster or a GDAL dataframe.

    writeable : bool, optional
        If True, the raster is opened in write mode. Default: True.
    
    Returns
    -------
    gdal.Dataset
        A gdal.Dataset.
    """
    assert isinstance(raster, (gdal.Dataset, str)), "raster must be a string or a gdal.Dataset"

    if isinstance(raster, gdal.Dataset):
        return raster

    if isinstance(raster, str) and raster.startswith("/vsizip/"):
        writeable = False

    if utils_path._check_file_exists(raster):

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = gdal.Open(raster, gdal.GF_Write) if writeable else gdal.Open(raster, gdal.GF_Read)
        gdal.PopErrorHandler()

        if not isinstance(opened, gdal.Dataset):
            raise ValueError(f"Input raster is not readable. Received: {raster}")

        if opened.GetDescription() == "":
            opened.SetDescription(raster)

        if opened.GetProjectionRef() == "":
            opened.SetProjection(utils_projection._get_default_projection())
            warnings.warn(f"WARNING: Input raster {raster} has no projection. Setting to default: EPSG:4326.", UserWarning)

        return opened

    raise ValueError(f"Input raster does not exists. Received: {raster}")


def _get_first_nodata_value(
    raster: Union[str, gdal.Dataset],
) -> Optional[Union[float, int]]:
    """
    Gets the first nodata value from a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset
        The raster to get the nodata value from.

    Returns
    -------
    float or None
        The nodata value if found, or None if not found.
    """
    utils_base.type_check(raster, [str, gdal.Dataset], "raster")

    nodata = None

    dataset = _raster_open(raster)
    band_count = dataset.RasterCount
    for band in range(1, band_count + 1):
        band_ref = dataset.GetRasterBand(band)
        nodata_value = band_ref.GetNoDataValue()

        if nodata_value is not None:
            nodata = nodata_value
            break

    dataset = None
    return nodata


def _get_basic_metadata_raster(
    raster: Union[str, gdal.Dataset],
) -> Dict[str, Any]:
    """
    Get basic metadata from a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset
        The raster to get the metadata from.

    Returns
    -------
    Dict[str]
        A dictionary with the metadata.
    """
    utils_base.type_check(raster, [str, gdal.Dataset], "raster")

    dataset = _raster_open(raster)
    transform = dataset.GetGeoTransform()
    projection_wkt = dataset.GetProjectionRef()
    projection_osr = osr.SpatialReference()
    projection_osr.ImportFromWkt(projection_wkt)

    bbox = utils_bbox._get_bbox_from_geotransform(transform, dataset.RasterXSize, dataset.RasterYSize)
    bbox_latlng = utils_projection.reproject_bbox(bbox, projection_osr, utils_projection._get_default_projection_osr())
    bounds_latlng = utils_bbox._get_bounds_from_bbox(bbox, projection_osr, wkt=False)
    bounds_area = bounds_latlng.GetArea()
    bounds_wkt = bounds_latlng.ExportToWkt()
    first_band = dataset.GetRasterBand(1)
    dtype = None if first_band is None else first_band.DataType
    dtype_numpy = utils_translate._translate_dtype_gdal_to_numpy(dtype)

    metadata = {
        "path": dataset.GetDescription(),
        "driver": dataset.GetDriver().ShortName,
        "projection_osr": projection_osr,
        "projection_wkt": projection_wkt,
        "geotransform": transform,
        "size": (dataset.RasterXSize, dataset.RasterYSize),
        "shape": [dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount],
        "height": dataset.RasterYSize,
        "width": dataset.RasterXSize,
        "pixel_size": (abs(transform[1]), (transform[5])),
        "pixel_width": abs(transform[1]),
        "pixel_height": abs(transform[5]),
        "origin": (transform[0], transform[3]),
        "origin_x": transform[0],
        "origin_y": transform[3],
        "bbox": bbox,
        "bbox_gdal": utils_bbox._get_gdal_bbox_from_ogr_bbox(bbox),
        "bbox_latlng": bbox_latlng,
        "bounds_latlng": bounds_wkt,
        "x_min": bbox[0],
        "x_max": bbox[1],
        "y_min": bbox[2],
        "y_max": bbox[3],
        "bands": dataset.RasterCount,
        "dtype_gdal": dtype,
        "dtype": dtype_numpy,
        "dtype_name": dtype_numpy.name,
        "area_latlng": bounds_area,
    }

    x_min, x_max, y_min, y_max = bbox
    metadata["area"] = (x_max - x_min) * (y_max - y_min)

    # Add the nodata values
    metadata["nodata"] = False
    metadata["nodata_value"] = None
    for band in range(1, metadata["bands"] + 1):
        band_ref = dataset.GetRasterBand(band)
        nodata_value = band_ref.GetNoDataValue()

        if nodata_value is not None:
            metadata["nodata"] = True
            metadata["nodata_value"] = nodata_value
            break

    dataset = None
    return metadata


def _get_basic_metadata_raster_list(
    rasters: List[Union[str, gdal.Dataset]],
) -> List[Dict[str, Any]]:
    """
    Get basic metadata from a list of rasters.

    Parameters
    ----------
    rasters : List[Union[str, gdal.Dataset]]
        The rasters to get the metadata from.

    Returns
    -------
    List[Dict[str]]
        A list of dictionaries with the metadata.
    """
    utils_base.type_check(rasters, [[str, gdal.Dataset]], "rasters")

    metadata = []
    for raster in rasters:
        metadata.append(_get_basic_metadata_raster(raster))

    return metadata


def _check_raster_has_nodata(
    raster: Union[str, gdal.Dataset],
) -> bool:
    """
    Verifies whether a raster has any nodata values.

    Parameters
    ----------
    raster : str or gdal.Dataset
        A raster, either in gdal.Dataset or a string referring to the dataset.

    Returns
    -------
    bool
        True if raster has nodata values, False otherwise.
    """
    utils_base.type_check(raster, [str, gdal.Dataset], "raster")

    metadata = _get_basic_metadata_raster(raster)
    if metadata["nodata"]:
        return True

    return False


def _check_raster_has_nodata_list(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
) -> bool:
    """
    Verifies whether a list of rasters have any nodata values.

    Parameters
    ----------
    rasters : list
        A list of rasters, either in gdal.Dataset or a string referring to the dataset.

    Returns
    -------
    bool
        True if all rasters have nodata values, False otherwise.
    """
    utils_base.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    rasters = utils_io._get_input_paths(rasters, "raster")

    metadata = _get_basic_metadata_raster_list(rasters)
    for meta in metadata:
        if meta["nodata"]:
            return True

    return False


def _raster_count_bands_list(
    rasters: List[Union[str, gdal.Dataset]]
) -> int:
    """
    Counts the number of bands in a list of rasters.

    Parameters
    ----------
    rasters : list
        A list of rasters, either in gdal.Dataset or a string referring to the dataset.

    Returns
    -------
    int
        The number of bands in the rasters.
    """
    utils_base.type_check(rasters, [[str, gdal.Dataset]], "raster")

    rasters = utils_io._get_input_paths(rasters, "raster")
    metadata = _get_basic_metadata_raster_list(rasters)

    bands = 0
    for meta in metadata:
        bands += meta["bands"]

    return bands


def check_rasters_have_same_nodata(
    rasters: List[Union[str, gdal.Dataset]],
) -> bool:
    """
    Verifies whether a list of rasters have the same nodata values.

    Parameters
    ----------
    rasters : list
        A list of rasters, either in gdal.Dataset or a string referring to the dataset.

    Returns
    -------
    bool
        True if all rasters have the same nodata value, False otherwise.
    """
    utils_base.type_check(rasters, [[str, gdal.Dataset]], "raster")

    rasters = utils_io._get_input_paths(rasters, "raster")

    metadata = _get_basic_metadata_raster_list(rasters)

    nodata_value = metadata[0]["nodata_value"]
    for meta in metadata[1:]:
        if meta["nodata_value"] != nodata_value:
            return False

    return True


def check_rasters_intersect(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
) -> bool:
    """
    Checks if two rasters intersect using their latlong boundaries.

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        The first raster.

    raster2 : str or gdal.Dataset
        The second raster.

    Returns
    -------
    bool
        True if the rasters intersect, False otherwise.
    """
    utils_base.type_check(raster1, [str, gdal.Dataset], "raster1")
    utils_base.type_check(raster2, [str, gdal.Dataset], "raster2")

    meta_1 = _get_basic_metadata_raster(raster1)
    meta_2 = _get_basic_metadata_raster(raster2)

    geom_1 = ogr.CreateGeometryFromWkt(meta_1["bounds_latlng"], meta_1["projection_osr"])
    geom_2 = ogr.CreateGeometryFromWkt(meta_2["bounds_latlng"], meta_2["projection_osr"])

    # Do the layers intersect?
    intersect = geom_1.Intersects(geom_2)

    return intersect


def get_raster_intersection(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
) -> ogr.Geometry:
    """
    Gets the latlng intersection of two rasters.

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        The first raster.

    raster2 : str or gdal.Dataset
        The second raster.

    Returns
    -------
    tuple or ogr.Geometry
        If return_as_vector is False, returns a tuple `(xmin, ymin, xmax, ymax)` representing
        the intersection of the two rasters. If return_as_vector is True, returns an ogr.Geometry
        object representing the intersection.
    """
    utils_base.type_check(raster1, [str, gdal.Dataset], "raster1")
    utils_base.type_check(raster2, [str, gdal.Dataset], "raster2")

    if not check_rasters_intersect(raster1, raster2):
        raise ValueError("Rasters do not intersect.")

    meta_1 = _get_basic_metadata_raster(raster1)
    meta_2 = _get_basic_metadata_raster(raster2)

    geom_1 = ogr.CreateGeometryFromWkt(meta_1["bounds_latlng"], meta_1["projection_osr"])
    geom_2 = ogr.CreateGeometryFromWkt(meta_2["bounds_latlng"], meta_2["projection_osr"])

    if not geom_1.Intersects(geom_2):
        raise ValueError("Rasters do not intersect.")

    # Get the intersection.
    intersection = geom_1.Intersection(geom_2)

    return intersection


def get_raster_overlap_fraction(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
) -> float:
    """
    Get the fraction of the overlap between two rasters.
    (e.g. 0.9 for mostly overlapping rasters)

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        The first raster (master).

    raster2 : str or gdal.Dataset
        The second raster.

    Returns
    -------
    float
        A value between 0.0 and 1.0 representing the degree of overlap between the two rasters.
    """
    utils_base.type_check(raster1, [str, gdal.Dataset, [str, gdal.Dataset]], "raster1")
    utils_base.type_check(raster2, [str, gdal.Dataset, [str, gdal.Dataset]], "raster2")

    if not check_rasters_intersect(raster1, raster2):
        return 0.0

    meta_1 = _get_basic_metadata_raster(raster1)
    meta_2 = _get_basic_metadata_raster(raster2)

    geom_1 = ogr.CreateGeometryFromWkt(meta_1["bounds_latlng"], meta_1["projection_osr"])
    geom_2 = ogr.CreateGeometryFromWkt(meta_2["bounds_latlng"], meta_2["projection_osr"])

    if not geom_1.Intersects(geom_2):
        return 0.0

    try:
        intersection = geom_1.Intersection(geom_2)
        overlap = intersection.GetArea() / geom_1.GetArea()
    except RuntimeError:
        overlap = 0.0

    return overlap


def check_rasters_are_aligned(
    rasters: List[Union[str, gdal.Dataset]],
    *,
    same_dtype: bool = False,
    same_nodata: bool = False,
    same_bands: bool = False,
    threshold: float = 0.0001,
) -> bool:
    """
    Verifies whether a list of rasters are aligned.

    Parameters
    ----------
    rasters : list
        A list of rasters, either in gdal.Dataset or a string referring to the dataset.

    same_dtype : bool, optional
        If True, all the rasters should have the same data type. Default: False.

    same_nodata : bool, optional
        If True, all the rasters should have the same nodata value. Default: False.

    threshold : float, optional
        The threshold for the difference between the rasters. Default: 0.0001.

    Returns
    -------
    bool
        True if rasters are aligned and optional parameters are True, False otherwise.
    """
    utils_base.type_check(rasters, [[str, gdal.Dataset]], "rasters")
    utils_base.type_check(same_dtype, [bool], "same_dtype")
    utils_base.type_check(same_nodata, [bool], "same_nodata")
    utils_base.type_check(threshold, [float], "threshold")

    if len(rasters) == 1:
        return True

    assert utils_gdal._check_is_raster_list(rasters), "Input is not a list of valid rasters."

    # Get the metadata of the first raster
    metadata_base = _get_basic_metadata_raster(rasters[0])

    for raster in rasters[1:]:
        # Get the metadata of the current raster
        metadata_current = _get_basic_metadata_raster(raster)

        # Check if the same amount of pixels
        if not metadata_base["size"][0] == metadata_current["size"][0] and metadata_base["size"][1] == metadata_current["size"][1]:
            return False

        # Check if the projections are the same
        if not metadata_base["projection_osr"].IsSame(metadata_current["projection_osr"]):
            return False

        # Check if origin is the same
        origin_base = metadata_base["origin"]
        origin_current = metadata_current["origin"]

        if not utils_base._check_number_is_within_threshold(origin_base[0], origin_current[0], threshold):
            return False

        if not utils_base._check_number_is_within_threshold(origin_base[1], origin_current[1], threshold):
            return False

        # Check if the pixel size is the same
        pixel_size_base = metadata_base["pixel_size"]
        pixel_size_current = metadata_current["pixel_size"]

        if not utils_base._check_number_is_within_threshold(pixel_size_base[0], pixel_size_current[0], threshold):
            return False

        if not utils_base._check_number_is_within_threshold(pixel_size_base[1], pixel_size_current[1], threshold):
            return False

        if same_dtype:
            if not metadata_base["dtype"] == metadata_current["dtype"]:
                return False

        if same_nodata:
            if not metadata_base["nodata"] == metadata_current["nodata"]:
                return False

            if metadata_base["nodata"]:
                if not metadata_base["nodata_value"] == metadata_current["nodata_value"]:
                    return False

        if same_bands:
            if not metadata_base["bands"] == metadata_current["bands"]:
                return False

    return True


def raster_open(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    *,
    writeable=True,
) -> Union[gdal.Dataset, List[gdal.Dataset]]:
    """
    Opens a raster in read or write mode.
    Supports lists.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
        A path to a raster or a GDAL dataframe.

    writeable : bool, optional
        If True, the raster is opened in write mode. Default: True.

    Returns
    -------
    Union[gdal.Dataset, List[gdal.Dataset]]
        A gdal.Dataset or a list of gdal.Datasets.
    """
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(writeable, [bool], "writeable")

    input_is_list = isinstance(raster, list)
    in_raster = utils_io._get_input_paths(raster, "raster")

    list_return = []
    for r in in_raster:
        try:
            list_return.append(_raster_open(r, writeable=writeable))
        except Exception:
            raise ValueError(f"Could not open raster: {r}") from None

    if input_is_list:
        return list_return

    return list_return[0]
