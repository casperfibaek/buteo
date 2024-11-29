"""### Utility functions to work with GDAL and projections. ###"""

# Standard Library
import math
from typing import Union, List, Optional

# External
import numpy as np
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import utils_gdal



def _get_default_projection() -> str:
    """Get the default projection for a new raster.
    EPSG:4326 in WKT format.

    Returns
    -------
    str:
        The default projection. (EPSG:4326) in WKT format.
    """
    epsg_4326_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'

    return epsg_4326_wkt


def _get_default_projection_osr() -> osr.SpatialReference:
    """Get the default projection for a new raster.
    EPSG:4326 in osr.SpatialReference format.

    Returns
    -------
    osr.SpatialReference:
        The default projection. (EPSG:4326) in osr.SpatialReference format.

    Raises
    ------
    RuntimeError:
        If projection could not be created.
    """
    spatial_ref = osr.SpatialReference()
    wkt = _get_default_projection()

    if spatial_ref.ImportFromWkt(wkt) != 0:
        raise RuntimeError("Could not create default projection.")

    return spatial_ref


def _get_pseudo_mercator_projection() -> str:
    """Get the pseudo-mercator projection.
    EPSG:3857 in WKT format.

    Returns
    -------
    str:
        The web-mercator (pseudo-mercator) projection. (EPSG:3857) in WKT format.
    """
    epsg_3857_wkt = 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs"],AUTHORITY["EPSG","3857"]]'

    return epsg_3857_wkt


def _get_pseudo_mercator_projection_osr() -> osr.SpatialReference:
    """Get the pseudo-mercator projection.
    EPSG:3857 in osr.SpatialReference format.

    Returns
    -------
    osr.SpatialReference:
        The web-mercator (pseudo-mercator) projection. (EPSG:3857) in osr.SpatialReference format.

    Raises
    ------
    RuntimeError:
        If projection could not be created.
    """
    spatial_ref = osr.SpatialReference()
    wkt = _get_pseudo_mercator_projection()

    if spatial_ref.ImportFromWkt(wkt) != 0:
        raise RuntimeError("Could not create pseudo-mercator projection.")

    return spatial_ref


def _get_esri_projection(esri_code: str) -> str:
    """Imports a projection from an ESRI code.

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
        wkt = """
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

    if spatial_ref.ImportFromWkt(wkt) != 0:
        raise RuntimeError("Could not create pseudo-mercator projection.")

    return spatial_ref.ExportToWkt()


def parse_projection(
    projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
) -> osr.SpatialReference:
    """Parses a gdal, ogr or osr data source and extracts the projection.

    Parameters
    ----------
    projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The projection to parse.

    Returns
    -------
    osr.SpatialReference
        The projection as an osr.SpatialReference

    Raises
    ------
    ValueError
        If projection is None, invalid type, or cannot be parsed
    """
    if projection is None:
        raise ValueError("Projection cannot be None")

    valid_types = (str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference)
    if not isinstance(projection, valid_types):
        raise ValueError(f"Projection must be one of {valid_types}, got {type(projection)}")

    target_proj = osr.SpatialReference()
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    try:
        if isinstance(projection, ogr.DataSource):
            layer = projection.GetLayer()
            if layer is None:
                raise ValueError("Vector datasource has no layers")
            spatial_ref = layer.GetSpatialRef()
            if spatial_ref is None:
                raise ValueError("Layer has no spatial reference")
            return spatial_ref

        if isinstance(projection, gdal.Dataset):
            wkt = projection.GetProjection()
            if not wkt:
                raise ValueError("Raster has no projection")
            target_proj.ImportFromWkt(wkt)
            return target_proj

        if isinstance(projection, osr.SpatialReference):
            if not projection.ExportToWkt():
                raise ValueError("Spatial reference is empty")
            return projection

        if isinstance(projection, str):
            if utils_gdal._check_is_raster(projection):
                ds = gdal.Open(projection, gdal.GA_ReadOnly)
                if ds is None:
                    raise ValueError("Could not open raster")
                try:
                    wkt = ds.GetProjection()
                    if not wkt:
                        raise ValueError("Raster has no projection")
                    target_proj.ImportFromWkt(wkt)
                    return target_proj
                finally:
                    ds = None

            if utils_gdal._check_is_vector(projection):
                ds = ogr.Open(projection, 0)
                if ds is None:
                    raise ValueError("Could not open vector")
                try:
                    layer = ds.GetLayer()
                    if layer is None:
                        raise ValueError("Vector has no layers")
                    spatial_ref = layer.GetSpatialRef()
                    if spatial_ref is None:
                        raise ValueError("Layer has no spatial reference")
                    return spatial_ref
                finally:
                    ds = None

            # Handle EPSG codes
            if projection.upper().startswith("EPSG:"):
                try:
                    epsg = int(projection.split(":")[1])
                    if target_proj.ImportFromEPSG(epsg) == 0:
                        return target_proj
                except (ValueError, IndexError):
                    pass

            # Handle ESRI codes
            if projection.upper().startswith("ESRI:"):
                try:
                    wkt = _get_esri_projection(projection)
                    if target_proj.ImportFromWkt(wkt) == 0:
                        return target_proj
                except ValueError:
                    pass

            # Try WKT and Proj4 imports
            for import_func in (target_proj.ImportFromWkt, target_proj.ImportFromProj4):
                try:
                    if import_func(projection) == 0:
                        return target_proj
                except Exception:  # pylint: disable=broad-except
                    continue

            raise ValueError(f"Could not parse projection string: {projection}")

        if isinstance(projection, int):
            if target_proj.ImportFromEPSG(projection) != 0:
                raise ValueError(f"Invalid EPSG code: {projection}")
            return target_proj

    except Exception as e:
        raise ValueError(f"Failed to parse projection: {str(e)}") from e
    finally:
        gdal.PopErrorHandler()


def parse_projection_wkt(
    projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
) -> str:
    """Parses a gdal, ogr or osr data source and extracts the projection as WKT.

    Parameters
    ----------
    projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The projection to parse.

    Returns
    -------
    str
        The projection in WKT format

    Raises
    ------
    ValueError
        If projection is None, invalid type, or cannot be parsed
    """
    spatial_ref = parse_projection(projection)
    wkt = spatial_ref.ExportToWkt()

    if not wkt:
        raise ValueError("Failed to export projection to WKT")

    return wkt


def _projection_is_latlng(
    projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
) -> bool:
    """Tests if a projection is in latlng format.

    Parameters
    ----------
    projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The projection to test.

    Returns
    -------
    bool
        True if the projection is in latlng format, False otherwise.

    Raises
    ------
    ValueError
        If projection is None or cannot be parsed
    """
    if projection is None:
        raise ValueError("Projection cannot be None")

    try:
        proj = parse_projection(projection)
        proj = osr.SpatialReference(proj) if isinstance(proj, str) else proj

        return bool(proj.IsGeographic())
    except Exception as e:
        raise ValueError(f"Could not parse projection: {str(e)}") from e


def _check_projections_match(
    source1: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    source2: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
) -> bool:
    """Tests if two projection sources have the same projection.

    Parameters
    ----------
    source1 : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The first projection to test.
    source2 : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The second projection to test.

    Returns
    -------
    bool
        True if the projections match, False otherwise.

    Raises
    ------
    ValueError
        If either source is None or cannot be parsed
    """
    if source1 is None or source2 is None:
        raise ValueError("Source projections cannot be None")

    try:
        proj1 = parse_projection(source1)
        proj2 = parse_projection(source2)

        proj1 = osr.SpatialReference(proj1) if isinstance(proj1, str) else proj1
        proj2 = osr.SpatialReference(proj2) if isinstance(proj2, str) else proj2

        return bool(proj1.IsSame(proj2) or proj1.ExportToProj4() == proj2.ExportToProj4())

    except Exception as e:
        raise ValueError(f"Could not compare projections: {str(e)}") from e


def _check_projections_match_list(
    list_of_projection_sources: List[Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]],
) -> bool:
    """Tests if a list of projection sources all have the same projection.

    Parameters
    ----------
    list_of_projection_sources : List[Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]]
        The list of projections to test.

    Returns
    -------
    bool
        True if the projections match, False otherwise.

    Raises
    ------
    TypeError
        If list_of_projection_sources is not a list
    ValueError
        If list is empty or contains None values
    """
    if not isinstance(list_of_projection_sources, list):
        raise TypeError("list_of_projection_sources must be a list")

    if not list_of_projection_sources:
        raise ValueError("list_of_projection_sources must not be empty")

    if any(src is None for src in list_of_projection_sources):
        raise ValueError("list_of_projection_sources cannot contain None values")

    if len(list_of_projection_sources) == 1:
        return True

    first_proj = parse_projection(list_of_projection_sources[0])
    first_proj = osr.SpatialReference(first_proj) if isinstance(first_proj, str) else first_proj

    for source in list_of_projection_sources[1:]:
        try:
            current_proj = parse_projection(source)
            current_proj = osr.SpatialReference(current_proj) if isinstance(current_proj, str) else current_proj
            if not bool(first_proj.IsSame(current_proj) or
                       first_proj.ExportToProj4() == current_proj.ExportToProj4()):
                return False
        except Exception as e:
            raise ValueError(f"Could not parse projection: {str(e)}") from e

    return True



def _get_projection_from_raster(
    raster: Union[str, gdal.Dataset],
) -> osr.SpatialReference:
    """Get the projection from a raster.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        A raster or raster path.

    Returns
    -------
    osr.SpatialReference
        The projection in OSR format.

    Raises
    ------
    ValueError
        If raster is None or invalid type
    RuntimeError
        If raster cannot be opened or has no projection
    """
    if raster is None:
        raise ValueError("Raster cannot be None")

    if not isinstance(raster, (str, gdal.Dataset)):
        raise ValueError(f"Raster must be str or gdal.Dataset, got {type(raster)}")

    try:
        ds = raster if isinstance(raster, gdal.Dataset) else gdal.Open(raster, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Could not open raster: {raster}")

        wkt = ds.GetProjection()
        if not wkt:
            raise RuntimeError("Raster has no projection")

        projection = osr.SpatialReference()
        if projection.ImportFromWkt(wkt) != 0:
            raise RuntimeError("Failed to parse projection WKT")

        if not projection.ExportToWkt():
            raise RuntimeError("Resulting projection is empty")

        if ds != raster:
            ds = None

        return projection

    except Exception as e:
        if ds != raster:
            ds = None
        raise RuntimeError(f"Failed to get projection from raster: {str(e)}") from e


def _get_projection_from_vector(
    vector: Union[str, ogr.DataSource],
) -> osr.SpatialReference:
    """Get the projection from a vector.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        A vector or vector path.

    Returns
    -------
    osr.SpatialReference
        The projection in OSR format.

    Raises
    ------
    ValueError
        If vector is None or invalid type
    RuntimeError
        If vector cannot be opened, has no layers, or has no projection
    """
    if vector is None:
        raise ValueError("Vector cannot be None")

    if not isinstance(vector, (str, ogr.DataSource)):
        raise ValueError(f"Vector must be str or ogr.DataSource, got {type(vector)}")

    try:
        ds = vector if isinstance(vector, ogr.DataSource) else ogr.Open(vector, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Could not open vector: {vector}")

        layer = ds.GetLayer()
        if layer is None:
            raise RuntimeError("Vector has no layers")

        projection = layer.GetSpatialRef()
        if projection is None:
            raise RuntimeError("Vector has no projection")

        if not projection.ExportToWkt():
            raise RuntimeError("Resulting projection is empty")

        if ds != vector:
            ds = None

        return projection

    except Exception as e:
        if ds != vector:
            ds = None
        raise RuntimeError(f"Failed to get projection from vector: {str(e)}") from e


def _get_projection_from_dataset(
    dataset: Union[str, gdal.Dataset, ogr.DataSource]
) -> osr.SpatialReference:
    """Get the projection from a dataset.

    Parameters
    ----------
    dataset : Union[str, gdal.Dataset, ogr.DataSource]
        A dataset or dataset path.

    Returns
    -------
    osr.SpatialReference
        The projection in OSR format.

    Raises
    ------
    ValueError
        If dataset is None or invalid type
    RuntimeError
        If dataset cannot be opened or has no projection
    """
    if dataset is None:
        raise ValueError("Dataset cannot be None")

    if not isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)):
        raise ValueError(f"Dataset must be str, gdal.Dataset, or ogr.DataSource, got {type(dataset)}")

    opened = None
    try:
        if isinstance(dataset, (gdal.Dataset, ogr.DataSource)):
            opened = dataset
        else:
            gdal.PushErrorHandler('CPLQuietErrorHandler')
            opened = gdal.Open(dataset, gdal.GA_ReadOnly)
            if opened is None:
                opened = ogr.Open(dataset, gdal.GA_ReadOnly)
            gdal.PopErrorHandler()

        if opened is None:
            raise RuntimeError(f"Could not open dataset: {dataset}")

        if isinstance(opened, gdal.Dataset):
            return _get_projection_from_raster(opened)
        elif isinstance(opened, ogr.DataSource):
            return _get_projection_from_vector(opened)
        else:
            raise RuntimeError(f"Unknown dataset type: {type(opened)}")

    finally:
        if opened is not None and opened != dataset:
            opened = None


def reproject_bbox(
    bbox_ogr: List[Union[int, float]],
    source_projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    target_projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
) -> List[Union[int, float]]:
    """Reprojects an OGR formatted bbox.
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
    if not isinstance(bbox_ogr, list):
        raise ValueError("bbox_ogr must be a list.")

    if len(bbox_ogr) != 4:
        raise ValueError("bbox_ogr must have 4 elements.")

    for val in bbox_ogr:
        if not isinstance(val, (int, float)):
            raise ValueError("bbox_ogr must only contain numbers.")

    if not isinstance(source_projection, (str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference)):
        raise ValueError("source_projection must be a string, int, gdal.Dataset, ogr.DataSource, or osr.SpatialReference.")

    if not isinstance(target_projection, (str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference)):
        raise ValueError("target_projection must be a string, int, gdal.Dataset, ogr.DataSource, or osr.SpatialReference.")

    x_min, x_max, y_min, y_max = bbox_ogr
    if x_min > x_max:
        raise ValueError("x_min must be less than or equal to x_max.")
    if y_min > y_max:
        raise ValueError("y_min must be less than or equal to y_max.")

    src_proj = parse_projection(source_projection)
    dst_proj = parse_projection(target_projection)

    if _check_projections_match(src_proj, dst_proj):
        return bbox_ogr

    p1 = [x_min, y_min]
    p2 = [x_max, y_min]
    p3 = [x_max, y_max]
    p4 = [x_min, y_max]

    options = osr.CoordinateTransformationOptions()
    if _projection_is_latlng(src_proj):
        p1.reverse()
        p2.reverse()
        p3.reverse()
        p4.reverse()

        options.SetAreaOfInterest(-180.0, -90.0, 180.0, 90.0)

    transformer = osr.CoordinateTransformation(src_proj, dst_proj, options)

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        p1t = transformer.TransformPoint(p1[0], p1[1])
        p2t = transformer.TransformPoint(p2[0], p2[1])
        p3t = transformer.TransformPoint(p3[0], p3[1])
        p4t = transformer.TransformPoint(p4[0], p4[1])
    except (RuntimeError, TypeError):
        try:
            p1t = transformer.TransformPoint(float(p1[0]), float(p1[1]))
            p2t = transformer.TransformPoint(float(p2[0]), float(p2[1]))
            p3t = transformer.TransformPoint(float(p3[0]), float(p3[1]))
            p4t = transformer.TransformPoint(float(p4[0]), float(p4[1]))
        except (RuntimeError, TypeError, ValueError):
            try:
                p1t = transformer.TransformPoint(p1[1], p1[0])
                p2t = transformer.TransformPoint(p2[1], p2[0])
                p3t = transformer.TransformPoint(p3[1], p3[0])
                p4t = transformer.TransformPoint(p4[1], p4[0])
            except (RuntimeError, TypeError):
                p1t = transformer.TransformPoint(float(p1[1]), float(p1[0]))
                p2t = transformer.TransformPoint(float(p2[1]), float(p2[0]))
                p3t = transformer.TransformPoint(float(p3[1]), float(p3[0]))
                p4t = transformer.TransformPoint(float(p4[1]), float(p4[0]))
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
    """Reprojects a point from source to target projection.

    Parameters
    ----------
    p : List[Union[int, float]]
        The point to reproject as [x, y].
    source_projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The source projection.
    target_projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The target projection.

    Returns
    -------
    List[Unit[int, float]]
        The reprojected point as [x, y].

    Raises
    ------
    ValueError
        If point is None, invalid format, or projection is invalid.
    RuntimeError
        If reprojection fails.
    """
    if p is None:
        raise ValueError("Point cannot be None")

    if not isinstance(p, (list, tuple)):
        raise ValueError("Point must be a list or tuple")

    if len(p) != 2:
        raise ValueError("Point must have exactly 2 coordinates")

    if not all(isinstance(x, (int, float)) for x in p):
        raise ValueError("Point coordinates must be numbers")

    try:
        src_proj = parse_projection(source_projection)
        dst_proj = parse_projection(target_projection)

        # Return original point if projections match
        if _check_projections_match(src_proj, dst_proj):
            return [float(p[0]), float(p[1])]

        options = osr.CoordinateTransformationOptions()
        if _projection_is_latlng(dst_proj):
            options.SetAreaOfInterest(-180.0, -90.0, 180.0, 90.0)

        transformer = osr.CoordinateTransformation(src_proj, dst_proj, options)

        # Attempt transformation
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        try:
            x, y, _ = transformer.TransformPoint(float(p[0]), float(p[1]))
            return [x, y]
        except Exception as e:
            raise RuntimeError(f"Reprojection failed: {str(e)}") from e
        finally:
            gdal.PopErrorHandler()

    except Exception as e:
        raise ValueError(f"Invalid projection: {str(e)}") from e


def _get_utm_epsg_from_latlng(
    latlng: List[Union[int, float]]
) -> str:
    """Get the UTM EPSG code from latitude/longitude coordinates.

    Parameters
    ----------
    latlng : List[Union[int, float]]
        The latitude/longitude coordinates in [lat, lng] format.

    Returns
    -------
    str
        The EPSG code for the UTM zone.

    Raises
    ------
    ValueError
        If latlng is None, not a list/array, wrong length, or invalid coordinates.
    """
    if latlng is None:
        raise ValueError("latlng cannot be None")

    if not isinstance(latlng, (list, np.ndarray)):
        raise ValueError("latlng must be a list or numpy array")

    if len(latlng) != 2:
        raise ValueError("latlng must contain exactly 2 coordinates")

    try:
        lat, lng = float(latlng[0]), float(latlng[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("latlng coordinates must be numeric") from exc

    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180 <= lng <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")

    zone = math.floor(((lng + 180) / 6) + 1)
    hemisphere = "7" if lat < 0 else "6"  # 7 for South, 6 for North

    return f"32{hemisphere}{zone:02d}"


def _get_utm_zone_from_latlng(
    latlng: List[Union[int, float]]
) -> osr.SpatialReference:
    """Get the UTM zone projection from latitude/longitude coordinates.

    Parameters
    ----------
    latlng : List[Union[int, float]]
        The latitude/longitude coordinates in [lat, lng] format.

    Returns
    -------
    osr.SpatialReference
        The UTM projection.

    Raises
    ------
    ValueError
        If latlng is invalid.
    RuntimeError
        If projection creation fails.
    """
    epsg = _get_utm_epsg_from_latlng(latlng)
    lat, lng = float(latlng[0]), float(latlng[1])

    zone = math.floor(((lng + 180) / 6) + 1)
    hemisphere = "S" if lat < 0 else "N"

    wkt = f"""PROJCS["WGS 84 / UTM zone {zone}{hemisphere}",
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
        PARAMETER["central_meridian",{zone * 6 - 183}],
        PARAMETER["scale_factor",0.9996],
        PARAMETER["false_easting",500000],
        PARAMETER["false_northing",{"10000000" if hemisphere == "S" else "0"}],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["EPSG","{epsg}"]]"""

    projection = osr.SpatialReference()
    if projection.ImportFromWkt(wkt) != 0:
        raise RuntimeError("Failed to create UTM projection")

    return projection


def _reproject_latlng_point_to_utm(
    latlng: List[Union[int, float]]
) -> List[Union[int, float]]:
    """Converts a latlng point into UTM coordinates.

    Parameters
    ----------
    latlng : List[Union[int, float]]
        The latlng point to convert in [lat, lng] format.

    Returns
    -------
    List[Union[int, float]]
        The converted UTM point in [utm_x, utm_y] format.

    Raises
    ------
    ValueError
        If latlng is None, not a list, wrong length, or invalid coordinates.
    RuntimeError
        If transformation fails.
    """
    if latlng is None:
        raise ValueError("latlng cannot be None")

    if not isinstance(latlng, (list, np.ndarray)):
        raise ValueError("latlng must be a list or numpy array")

    if len(latlng) != 2:
        raise ValueError("latlng must contain exactly 2 coordinates")

    try:
        lat, lng = float(latlng[0]), float(latlng[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("latlng coordinates must be numeric") from exc

    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180 <= lng <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")

    # Get source (WGS84) and target (UTM) projections
    source_proj = parse_projection("EPSG:4326")
    target_proj = _get_utm_zone_from_latlng([lat, lng])

    # Create transformation
    transformer = osr.CoordinateTransformation(source_proj, target_proj)

    # Transform point
    try:
        utm_x, utm_y, _ = transformer.TransformPoint(lng, lat)
        return [utm_x, utm_y]
    except Exception as e:
        raise RuntimeError(f"UTM transformation failed: {str(e)}") from e


# TODO: Create a function to define the projection of a vector after creation.

def set_projection_raster(
    dataset: Union[str, gdal.Dataset],
    projection: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    pixel_size_x: Optional[float] = None,
    pixel_size_y: Optional[float] = None,
) -> bool:
    """Sets the projection of a raster and optionally adjusts pixel size.

    Parameters
    ----------
    dataset : Union[str, gdal.Dataset]
        The raster dataset to set the projection of.
    projection : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference] 
        The projection to set.
    pixel_size_x : Optional[float], optional
        The x pixel size, by default None
    pixel_size_y : Optional[float], optional
        The y pixel size (negative for north-up), by default None

    Returns
    -------
    bool
        True if successful.

    Raises
    ------
    ValueError
        If dataset or projection is None or invalid type
    RuntimeError
        If dataset cannot be opened or projection cannot be set
    """
    if dataset is None or projection is None:
        raise ValueError("Dataset and projection cannot be None")

    if not isinstance(dataset, (str, gdal.Dataset)):
        raise ValueError("Dataset must be a string or gdal.Dataset")

    if not isinstance(projection, (str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference)):
        raise ValueError("Projection must be a string, int, gdal.Dataset, ogr.DataSource, or osr.SpatialReference")

    # Parse projection first to fail early if invalid
    try:
        proj = parse_projection(projection)
        proj = osr.SpatialReference(proj) if isinstance(proj, str) else proj
    except Exception as e:
        raise ValueError(f"Invalid projection: {str(e)}") from e

    # Open dataset if string path provided
    opened = dataset if isinstance(dataset, gdal.Dataset) else None
    if opened is None:
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        try:
            opened = gdal.Open(dataset, gdal.GA_Update)
        except Exception as e:
            gdal.PopErrorHandler()
            raise RuntimeError(f"Could not open raster dataset: {str(e)}") from e
        gdal.PopErrorHandler()

    if opened is None:
        raise RuntimeError(f"Could not open raster dataset: {dataset}")

    try:
        # Set projection
        opened.SetProjection(proj.ExportToWkt())

        # Update geotransform only if pixel sizes are specified
        if pixel_size_x is not None or pixel_size_y is not None:
            gt = opened.GetGeoTransform()
            if gt:
                new_gt = list(gt)
                if pixel_size_x is not None:
                    new_gt[1] = pixel_size_x  # x pixel size
                if pixel_size_y is not None:
                    # Ensure pixel_size_y is negative for north-up orientation
                    new_gt[5] = -abs(pixel_size_y)  # y pixel size
                opened.SetGeoTransform(new_gt)

        opened.FlushCache()

    except Exception as e:
        raise RuntimeError(f"Failed to set projection: {str(e)}") from e

    finally:
        if opened is not None and opened != dataset:
            opened = None

    return True


def _get_transformer(
    proj_source: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    proj_target: Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
) -> osr.CoordinateTransformation:
    """Get a transformer object for reprojecting coordinates.

    Parameters
    ----------
    proj_source : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The source projection.
    proj_target : Union[str, int, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The target projection.

    Returns
    -------
    osr.CoordinateTransformation
        The transformer object.
        
    Raises
    ------
    ValueError
        If either projection is None or invalid
    RuntimeError
        If transformer cannot be created
    """
    if proj_source is None or proj_target is None:
        raise ValueError("Source and target projections cannot be None")

    try:
        source_sref = parse_projection(proj_source)
        target_sref = parse_projection(proj_target)

        options = osr.CoordinateTransformationOptions()
        if _projection_is_latlng(target_sref):
            options.SetAreaOfInterest(-180.0, -90.0, 180.0, 90.0)

        transformer = osr.CoordinateTransformation(source_sref, target_sref, options)
        if transformer is None:
            raise RuntimeError("Failed to create coordinate transformation")

        return transformer

    except Exception as e:
        raise RuntimeError(f"Failed to create transformer: {str(e)}") from e
