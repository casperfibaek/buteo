"""
### Utility functions to work with GDAL and projections ###
"""

# Standard Library
import sys; sys.path.append("../../")
from typing import Union, List

# External
from osgeo import gdal, ogr, osr
import numpy as np

# Internal
from buteo.utils import utils_gdal


def _get_default_projection() -> str:
    """
    Get the default projection for a new raster.
    EPSG:4326 in WKT format.

    Returns
    -------
    str:
        The default projection. (EPSG:4326) in WKT format.
    """
    epsg_4326_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'

    return epsg_4326_wkt


def _get_default_projection_osr() -> osr.SpatialReference:
    """
    Get the default projection for a new raster.
    EPSG:4326 in osr.SpatialReference format.

    Returns
    -------
    osr.SpatialReference:
        The default projection. (EPSG:4326) in osr.SpatialReference format.
    """
    epsg_4326_wkt = _get_default_projection()

    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(epsg_4326_wkt)

    return spatial_ref


def _get_esri_projection(esri_code: str) -> str:
    """
    Imports a projection from an ESRI code.

    Parameters
    ----------
    esri_code : str
        The ESRI code to import.

    Returns
    -------
    str:
        The projection in WKT format.

    Notes
    -----
    The following ESRI codes are currently supported:
    - ESRI:54009 (EPSG:3785) ~ World_Mollweide
    """

    # ESRI:54009 is equivalent to EPSG:3785
    if esri_code == "ESRI:54009":
        wkt_code = """
        PROJCRS["World_Mollweide",
            BASEGEOGCRS["WGS 84",
                DATUM["World Geodetic System 1984",
                    ELLIPSOID["WGS 84",6378137,298.257223563,
                        LENGTHUNIT["metre",1]]],
                PRIMEM["Greenwich",0,
                    ANGLEUNIT["Degree",0.0174532925199433]]],
            CONVERSION["World_Mollweide",
                METHOD["Mollweide"],
                PARAMETER["Longitude of natural origin",0,
                    ANGLEUNIT["Degree",0.0174532925199433],
                    ID["EPSG",8802]],
                PARAMETER["False easting",0,
                    LENGTHUNIT["metre",1],
                    ID["EPSG",8806]],
                PARAMETER["False northing",0,
                    LENGTHUNIT["metre",1],
                    ID["EPSG",8807]]],
            CS[Cartesian,2],
                AXIS["(E)",east,
                    ORDER[1],
                    LENGTHUNIT["metre",1]],
                AXIS["(N)",north,
                    ORDER[2],
                    LENGTHUNIT["metre",1]],
            USAGE[
                SCOPE["Not known."],
                AREA["World."],
                BBOX[-90,-180,90,180]],
            ID["ESRI",54009]]
        """
    else:
        raise ValueError("Unknown ESRI code")

    # Create a spatial reference object
    spatial_ref = osr.SpatialReference()

    # Import the EPSG code
    spatial_ref.ImportFromWkt(wkt_code)

    return spatial_ref.ExportToWkt()


def parse_projection(
    projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    return_wkt: bool = False,
) -> Union[osr.SpatialReference, str]:
    """
    Parses a gdal, ogr or osr data source and extracts the projection. If
    a string or int is passed, it attempts to open it and return the projection as
    an osr.SpatialReference.

    Parameters
    ----------
    projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The projection to parse.

    return_wkt : bool, optional
        Whether to return the projection as a WKT string or an osr.SpatialReference

    Returns
    -------
    Union[osr.SpatialReference, str]:
        The projection as an osr.SpatialReference or a WKT string.
    """
    valid_types = (str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference)
    if not isinstance(projection, valid_types):
        raise ValueError("projection must be a string, int, gdal.Dataset, ogr.DataSource, or osr.SpatialReference.")

    err_msg = f"Unable to parse target projection: {projection}"
    target_proj = osr.SpatialReference()

    # Suppress gdal errors and handle them ourselves.
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    try:
        if isinstance(projection, ogr.DataSource):
            layer = projection.GetLayer()
            target_proj = layer.GetSpatialRef()

        elif isinstance(projection, gdal.Dataset):
            target_proj.ImportFromWkt(projection.GetProjection())

        elif isinstance(projection, osr.SpatialReference):
            if projection.ExportToWkt() == "":
                raise ValueError("Spatial reference is empty.")
            target_proj = projection

        elif isinstance(projection, str):
            if utils_gdal._check_is_raster(projection):
                ref = gdal.Open(projection, 0)
                target_proj.ImportFromWkt(ref.GetProjection())
            elif utils_gdal._check_is_vector(projection):
                ref = ogr.Open(projection, 0)
                layer = ref.GetLayer()
                target_proj = layer.GetSpatialRef()
            else:
                if not (target_proj.ImportFromWkt(projection) == 0 or
                        target_proj.ImportFromProj4(projection) == 0 or
                        (projection.lower().startswith("epsg:") and
                         target_proj.ImportFromEPSG(int(projection.split(":")[1])) == 0) or
                        (projection.lower().startswith("esri:") and
                         target_proj.ImportFromWkt(_get_esri_projection(projection)) == 0)):
                    raise ValueError(err_msg)

        elif isinstance(projection, int):
            if target_proj.ImportFromEPSG(projection) != 0:
                raise ValueError(err_msg)

        else:
            raise ValueError(err_msg)

    finally:
        gdal.PopErrorHandler()

    if not isinstance(target_proj, osr.SpatialReference) or target_proj.ExportToWkt() == "":
        raise ValueError(err_msg)

    return target_proj.ExportToWkt() if return_wkt else target_proj


def _check_projections_match(
    source1: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    source2: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
) -> bool:
    """
    Tests if two projection sources have the same projection.

    Parameters
    ----------
    source1 : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The first projection to test.

    source2 : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The second projection to test.

    Returns
    -------
    bool:
        **True** if the projections match, **False** otherwise.

    """
    proj1 = parse_projection(source1)
    proj2 = parse_projection(source2)

    if proj1.IsSame(proj2):
        return True
    elif proj1.ExportToProj4() == proj2.ExportToProj4():
        return True

    return False


def _check_projections_match_list(
    list_of_projection_sources: List[Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]],
) -> bool:
    """
    Tests if a list of projection sources all have the same projection.

    Parameters
    ----------
    list_of_projection_sources : List[Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]]
        The list of projections to test.

    Returns
    -------
    bool:
        **True** if the projections match, **False** otherwise.
    """
    assert isinstance(list_of_projection_sources, list), "list_of_projection_sources must be a list."

    if len(list_of_projection_sources) == 0:
        raise ValueError("list_of_projection_sources must not be empty.")

    if len(list_of_projection_sources) == 1:
        return True

    first = None
    for index, source in enumerate(list_of_projection_sources):
        if index == 0:
            first = parse_projection(source)
        else:
            compare = parse_projection(source)

            if not first.IsSame(compare) and first.ExportToProj4() != compare.ExportToProj4():
                return False

    return True



def _get_projection_from_raster(
    raster: Union[str, gdal.Dataset],
) -> osr.SpatialReference:
    """
    Get the projection from a raster.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        A raster or raster path.
    
    Returns
    -------
    osr.SpatialReference:
        The projection in OSR format.
    """
    opened = None
    if isinstance(raster, gdal.Dataset):
        opened = raster
    else:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = gdal.Open(raster, gdal.GA_ReadOnly)
        gdal.PopErrorHandler()

    if opened is None:
        raise RuntimeError(f"Could not open raster. {raster}")

    projection = osr.SpatialReference()
    projection.ImportFromWkt(opened.GetProjection())
    opened = None

    return projection


def _get_projection_from_vector(
    vector: Union[str, ogr.DataSource],
) -> osr.SpatialReference:
    """
    Get the projection from a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        A vector or vector path.

    Returns
    -------
    osr.SpatialReference:
        The projection in OSR format.
    """
    opened = None
    if isinstance(vector, ogr.DataSource):
        opened = vector
    else:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        opened = ogr.Open(vector, gdal.GA_ReadOnly)
        gdal.PopErrorHandler()

    if opened is None:
        raise RuntimeError(f"Could not open vector. {vector}")

    layer = opened.GetLayer()
    projection = layer.GetSpatialRef()
    opened = None

    return projection


def _get_projection_from_dataset(
    dataset: Union[str, gdal.Dataset, ogr.DataSource],
) -> osr.SpatialReference:
    """
    Get the projection from a dataset.

    Parameters
    ----------
    dataset : Union[str, gdal.Dataset, ogr.DataSource]
        A dataset or dataset path.
    
    Returns
    -------
    osr.SpatialReference:
        The projection in OSR format.
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
        return _get_projection_from_raster(opened)

    if isinstance(opened, ogr.DataSource):
        return _get_projection_from_vector(opened)

    raise RuntimeError(f"Could not get projection from dataset. {dataset}")


def reproject_bbox(
    bbox_ogr: List[Union[int, float]],
    source_projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    target_projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
) -> List[Union[int, float]]:
    """
    Reprojects an OGR formatted bbox.
    OGR formatted bboxes are in the format `[x_min, x_max, y_min, y_max]`.

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        The OGR formatted bbox to reproject.

    source_projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The source projection.

    target_projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The target projection.

    Returns
    -------
    List[Union[int, float]]:
        The reprojected bbox.
    """
    assert isinstance(bbox_ogr, list), "bbox_ogr must be a list."
    assert len(bbox_ogr) == 4, "bbox_ogr must have 4 elements."
    for val in bbox_ogr:
        assert isinstance(val, (int, float)), "bbox_ogr must only contain numbers."
    assert isinstance(source_projection, (str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference)), "source_projection must be a string, int, gdal.Dataset, ogr.DataSource, or osr.SpatialReference."
    assert isinstance(target_projection, (str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference)), "target_projection must be a string, int, gdal.Dataset, ogr.DataSource, or osr.SpatialReference."

    x_min, x_max, y_min, y_max = bbox_ogr
    assert x_min <= x_max, "x_min must be less than or equal to x_max."
    assert y_min <= y_max, "y_min must be less than or equal to y_max."

    src_proj = parse_projection(source_projection)
    dst_proj = parse_projection(target_projection)

    if _check_projections_match(src_proj, dst_proj):
        return bbox_ogr

    transformer = osr.CoordinateTransformation(src_proj, dst_proj)

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

    transformed_x_min = min(p1t[0], p2t[0], p3t[0], p4t[0])
    transformed_x_max = max(p1t[0], p2t[0], p3t[0], p4t[0])
    transformed_y_min = min(p1t[1], p2t[1], p3t[1], p4t[1])
    transformed_y_max = max(p1t[1], p2t[1], p3t[1], p4t[1])

    return [
        transformed_x_min,
        transformed_x_max,
        transformed_y_min,
        transformed_y_max,
    ]


def _reproject_point(
    p: List[Union[int, float]],
    source_projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    target_projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
) -> List[Union[int, float]]:
    """
    Reprojects a point.

    Parameters
    ----------
    p : List[Union[int, float]]
        The point to reproject.

    source_projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The source projection.

    target_projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The target projection.

    Returns
    -------
    List[Union[int, float]]:
        The reprojected point.
    """
    assert isinstance(p, list), "p must be a list."
    assert len(p) == 2, "p must have 2 elements."

    src_proj = parse_projection(source_projection)
    dst_proj = parse_projection(target_projection)

    if _check_projections_match(src_proj, dst_proj):
        return p

    transformer = osr.CoordinateTransformation(src_proj, dst_proj)

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        pt = transformer.TransformPoint(p[0], p[1])
    except RuntimeError:
        pt = transformer.TransformPoint(float(p[0]), float(p[1]))
    gdal.PopErrorHandler()

    return [pt[0], pt[1]]


def _get_utm_zone_from_latlng(
    latlng: List[Union[int, float]],
    return_epsg: bool = False,
) -> str:
    """
    Get the UTM ZONE from a latlng list.

    Parameters
    ----------
    latlng : List[Union[int, float]]
        The latlng list to get the UTM ZONE from.

    return_epsg : bool, optional
        Whether or not to return the EPSG code instead of the WKT, by default False

    Returns
    -------
    str
        The WKT or EPSG code.
    """
    assert isinstance(latlng, (list, np.ndarray)), "latlng must be in the form of a list."

    zone = round(((latlng[1] + 180) / 6) + 1)
    n_or_s = "S" if latlng[0] < 0 else "N"

    false_northing = "10000000" if n_or_s == "S" else "0"
    central_meridian = str(round(((zone * 6) - 180) - 3))
    epsg = f"32{'7' if n_or_s == 'S' else '6'}{str(zone)}"

    if return_epsg:
        return epsg

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


def _reproject_latlng_point_to_utm(
    latlng: List[Union[int, float]],
) -> List[Union[int, float]]:
    """
    Converts a latlng point into an UTM point.
    Takes point in [lat, lng], returns [utm_x, utm_y].

    Parameters
    ----------
    latlng : List[Union[int, float]]
        The latlng point to convert.

    Returns
    -------
    List[Union[int, float]]
        The converted UTM point.
    """
    source_projection = osr.SpatialReference()
    source_projection_wkt = _get_default_projection()
    source_projection.ImportFromWkt(source_projection_wkt)
    target_projection = _get_utm_zone_from_latlng(latlng)

    transformer = osr.CoordinateTransformation(
        source_projection, target_projection
    )

    try:
        utm_x, utm_y, _utm_z = transformer.TransformPoint(latlng[0], latlng[1])
    except: # pylint: disable=bare-except
        utm_x, utm_y, _utm_z = transformer.TransformPoint(float(latlng[0]), float(latlng[1]))

    return [utm_x, utm_y]
