"""###. Basic functionality for working with rasters. ###"""

# Standard library
import os
from typing import List, Optional, Union, Any

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_projection,
    utils_io,
)
from buteo.core_raster.core_raster_read import _open_raster
from buteo.core_raster.core_raster_info import _get_basic_info_raster, get_metadata_raster



def get_raster_overlap_fraction(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
) -> float:
    """Get the fraction of overlap between two rasters relative to the first raster.

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        The reference raster
    raster2 : str or gdal.Dataset
        The second raster

    Returns
    -------
    float
        Overlap fraction between 0.0 and 1.0

    Raises
    ------
    TypeError
        If inputs are not str or gdal.Dataset
    ValueError
        If rasters cannot be opened or processed
    """
    utils_base._type_check(raster1, [str, gdal.Dataset], "raster1")
    utils_base._type_check(raster2, [str, gdal.Dataset], "raster2")

    if not check_rasters_intersect(raster1, raster2):
        return 0.0

    ds1 = _open_raster(raster1)
    ds2 = _open_raster(raster2)

    # Get bounds for first raster
    info1 = _get_basic_info_raster(ds1)
    bbox1 = utils_bbox._get_bbox_from_geotransform(
        info1["transform"], ds1.RasterXSize, ds1.RasterYSize
    )
    geom1 = utils_bbox._get_geom_from_bbox(bbox1)
    geom1.AssignSpatialReference(info1["projection_osr"])

    # Get bounds for second raster
    info2 = _get_basic_info_raster(ds2)
    bbox2 = utils_bbox._get_bbox_from_geotransform(
        info2["transform"], ds2.RasterXSize, ds2.RasterYSize
    )
    geom2 = utils_bbox._get_geom_from_bbox(bbox2)
    geom2.AssignSpatialReference(info2["projection_osr"])

    try:
        intersection = geom1.Intersection(geom2)

        return intersection.GetArea() / geom1.GetArea()
    except RuntimeError as exc:
        raise ValueError("Failed to compute overlap fraction") from exc


def get_raster_intersection(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
) -> ogr.Geometry:
    """Gets the intersection geometry of two rasters in geographic coordinates.

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        The first raster
    raster2 : str or gdal.Dataset
        The second raster

    Returns
    -------
    ogr.Geometry
        Geometry representing the intersection of the two rasters

    Raises
    ------
    TypeError
        If inputs are not str or gdal.Dataset
    ValueError
        If rasters do not intersect or cannot be processed
    """
    utils_base._type_check(raster1, [str, gdal.Dataset], "raster1")
    utils_base._type_check(raster2, [str, gdal.Dataset], "raster2")

    if not check_rasters_intersect(raster1, raster2):
        raise ValueError("Rasters do not intersect")

    ds1 = _open_raster(raster1)
    ds2 = _open_raster(raster2)

    # Get bounds for first raster
    info1 = _get_basic_info_raster(ds1)
    bbox1 = utils_bbox._get_bbox_from_geotransform(
        info1["transform"], ds1.RasterXSize, ds1.RasterYSize
    )
    geom1 = utils_bbox._get_geom_from_bbox(bbox1)
    geom1.AssignSpatialReference(info1["projection_osr"])

    # Get bounds for second raster
    info2 = _get_basic_info_raster(ds2)
    bbox2 = utils_bbox._get_bbox_from_geotransform(
        info2["transform"], ds2.RasterXSize, ds2.RasterYSize
    )
    geom2 = utils_bbox._get_geom_from_bbox(bbox2)
    geom2.AssignSpatialReference(info2["projection_osr"])

    try:
        intersection = geom1.Intersection(geom2)
        return intersection
    except RuntimeError as e:
        raise ValueError(f"Failed to compute intersection: {str(e)}") from e


def check_rasters_intersect(
    raster1: Union[str, gdal.Dataset],
    raster2: Union[str, gdal.Dataset],
) -> bool:
    """Check if two rasters intersect in geographic coordinates.

    Parameters
    ----------
    raster1 : str or gdal.Dataset
        First raster
    raster2 : str or gdal.Dataset
        Second raster

    Returns
    -------
    bool
        True if rasters intersect, False otherwise

    Raises
    ------
    TypeError
        If inputs are not str or gdal.Dataset
    ValueError
        If rasters cannot be opened or projections are invalid
    """
    utils_base._type_check(raster1, [str, gdal.Dataset], "raster1")
    utils_base._type_check(raster2, [str, gdal.Dataset], "raster2")

    ds1 = _open_raster(raster1)
    ds2 = _open_raster(raster2)

    # Get bounds for first raster
    info1 = _get_basic_info_raster(ds1)
    bbox1 = utils_bbox._get_bbox_from_geotransform(
        info1["transform"], ds1.RasterXSize, ds1.RasterYSize
    )
    geom1 = utils_bbox._get_geom_from_bbox(bbox1)
    geom1.AssignSpatialReference(info1["projection_osr"])

    # Get bounds for second raster
    info2 = _get_basic_info_raster(ds2)
    bbox2 = utils_bbox._get_bbox_from_geotransform(
        info2["transform"], ds2.RasterXSize, ds2.RasterYSize
    )
    geom2 = utils_bbox._get_geom_from_bbox(bbox2)
    geom2.AssignSpatialReference(info2["projection_osr"])

    return geom1.Intersects(geom2)


def check_rasters_are_aligned(
    rasters: List[Union[str, gdal.Dataset]],
    *,
    same_dtype: bool = False,
    same_nodata: bool = False,
    same_bands: bool = False,
    threshold: float = 0.0001,
) -> bool:
    """Verifies whether a list of rasters are aligned.

    Parameters
    ----------
    rasters : List[Union[str, gdal.Dataset]]
        List of raster paths or GDAL datasets
    same_dtype : bool, optional
        Check if all rasters have same dtype. Default: False
    same_nodata : bool, optional
        Check if all rasters have same nodata value. Default: False
    same_bands : bool, optional
        Check if all rasters have same number of bands. Default: False
    threshold : float, optional
        Threshold for coordinate comparison. Default: 0.0001

    Returns
    -------
    bool
        True if rasters are aligned and meet criteria

    Raises
    ------
    TypeError
        If inputs have invalid types
    ValueError
        If raster list is empty or rasters are invalid
    """
    utils_base._type_check(rasters, [[str, gdal.Dataset]], "rasters")
    utils_base._type_check(same_dtype, [bool], "same_dtype")
    utils_base._type_check(same_nodata, [bool], "same_nodata")
    utils_base._type_check(same_bands, [bool], "same_bands")
    utils_base._type_check(threshold, [float], "threshold")

    if len(rasters) == 0:
        raise ValueError("Input is an empty list.")
    if len(rasters) == 1:
        return True

    # Get reference dataset
    ref_ds = _open_raster(rasters[0])
    ref_info = _get_basic_info_raster(ref_ds)
    ref_transform = ref_info["transform"]
    ref_size = ref_info["size"]

    for raster in rasters[1:]:
        ds = _open_raster(raster)
        info = _get_basic_info_raster(ds)

        # Check size
        if ref_size != info["size"]:
            return False

        # Check projection
        if not ref_info["projection_osr"].IsSame(info["projection_osr"]):
            return False

        # Check transform
        curr_transform = info["transform"]
        for i in range(6):
            if not utils_base._check_number_is_within_threshold(
                ref_transform[i], curr_transform[i], threshold
            ):
                return False

        # Optional checks
        if same_dtype and ref_info["dtype"] != info["dtype"]:
            return False

        if same_bands and ref_info["bands"] != info["bands"]:
            return False

        if same_nodata:
            ref_nodata = ref_ds.GetRasterBand(1).GetNoDataValue()
            curr_nodata = ds.GetRasterBand(1).GetNoDataValue()
            if ref_nodata != curr_nodata:
                return False

    return True


def _raster_to_vector_extent(
    raster: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    metadata: bool = False,
    latlng: bool = False,
    *,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> str:
    """Converts the extent of a raster to a vector file.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The raster to be converted.
    out_path : Optional[str], optional
        The path to the output vector. If None, the vector is created in memory., default: None
    metadata : bool, optional
        If True, metadata is added to the output vector., default: False
    latlng : bool, optional
        If True, the extent is returned in lat/lon coordinates. If False, the extent is returned in projected coordinates., default: False
    overwrite : bool, optional
        If True, the output vector will be overwritten if it already exists., default: True
    prefix : str, optional
        The prefix to be added to the output vector name., default: ""
    suffix : str, optional
        The suffix to be added to the output vector name., default: ""
    add_uuid : bool, optional
        If True, a unique identifier will be added to the output vector name., default: False
    add_timestamp : bool, optional
        If True, a timestamp will be added to the output vector name., default: False

    Returns
    -------
    str
        The path to the output vector.
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(metadata, [bool], "metadata")
    utils_base._type_check(latlng, [bool], "latlng")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    if not utils_gdal._check_is_raster(raster):
        raise ValueError("Input raster is not a valid raster file.")

    in_path = utils_io._get_input_paths(raster, "raster")[0]
    out_path = utils_io._get_output_paths(
        in_path,
        out_path,
        False,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
    )[0]

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    ds = _open_raster(raster)
    metadata_raster = get_metadata_raster(ds)

    # Projections
    projection_osr = metadata_raster["projection_osr"]
    projection_latlng = utils_projection._get_default_projection_osr()

    # Bounding boxes
    bbox = metadata_raster["bbox"]

    if latlng:
        footprint_geom = utils_bbox._get_bounds_from_bbox_as_geom(
            bbox,
            projection_osr,
        )
        footprint = utils_bbox._get_vector_from_geom(
            footprint_geom,
            projection_osr=projection_latlng,
            out_path=out_path,
            name=os.path.basename(out_path),
        )
    else:
        footprint_geom = utils_bbox._get_geom_from_bbox(
            bbox,
        )
        footprint = utils_bbox._get_vector_from_geom(
            footprint_geom,
            projection_osr=projection_osr,
            out_path=out_path,
            name=os.path.basename(out_path),
        )

    if metadata:
        attributes: List[tuple[str, str, Any]] = [
            ("name", "str", metadata_raster["name"]),
            ("basename", "str", metadata_raster["basename"]),
            ("folder", "str", metadata_raster["folder"]),
            ("ext", "str", metadata_raster["ext"]),
            ("driver", "str", metadata_raster["driver"]),
            ("projection_wkt", "str", metadata_raster["projection_wkt"]),
            ("geotransform", "str", metadata_raster["geotransform"]),
            ("size", "str", metadata_raster["size"]),
            ("width", "int", metadata_raster["width"]),
            ("height", "int", metadata_raster["height"]),
            ("pixel_width", "float", metadata_raster["pixel_width"]),
            ("pixel_height", "float", metadata_raster["pixel_height"]),
            ("x_min", "float", metadata_raster["x_min"]),
            ("x_max", "float", metadata_raster["x_max"]),
            ("y_min", "float", metadata_raster["y_min"]),
            ("y_max", "float", metadata_raster["y_max"]),
            ("shape", "str", metadata_raster["shape"]),
            ("bands", "int", metadata_raster["bands"]),
            ("dtype_gdal", "int", metadata_raster["dtype_gdal"]),
            ("dtype", "str", metadata_raster["dtype"]),
            ("dtype_name", "str", metadata_raster["dtype_name"]),
            ("pixel_size", "str", metadata_raster["pixel_size"]),
            ("origin", "str", metadata_raster["origin"]),
            ("nodata", "bool", metadata_raster["nodata"]),
            ("nodata_value", "str", metadata_raster["nodata_value"]),
            ("bbox", "str", metadata_raster["bbox"]),
            ("bbox_latlng", "str", metadata_raster["bbox_latlng"]),
            ("bbox_gdal", "str", metadata_raster["bbox_gdal"]),
            ("bbox_gdal_latlng", "str", metadata_raster["bbox_gdal_latlng"]),
            ("bounds_latlng", "str", metadata_raster["bounds_latlng"]),
            ("bounds", "str", metadata_raster["bounds"]),
            ("centroid", "str", metadata_raster["centroid"]),
            ("centroid_latlng", "str", metadata_raster["centroid_latlng"]),
            ("area", "float", metadata_raster["area"]),
            ("area_latlng", "float", metadata_raster["area_latlng"]),
        ]

        # Add attributes to the vector
        ds_vector = ogr.Open(footprint, gdal.GF_Write)
        layer = ds_vector.GetLayer()

        # Create all fields first
        for attribute in attributes:
            field_name, field_type, _ = attribute
            if field_type == "str":
                field = ogr.FieldDefn(field_name, ogr.OFTString)
            elif field_type == "int":
                field = ogr.FieldDefn(field_name, ogr.OFTInteger)
            elif field_type == "float":
                field = ogr.FieldDefn(field_name, ogr.OFTReal)
            elif field_type == "bool":
                field = ogr.FieldDefn(field_name, ogr.OFTInteger)
            else:
                raise ValueError(f"Field type {field_type} not supported")
            layer.CreateField(field)

        # Set all field values
        feature = layer.GetNextFeature()
        for attribute in attributes:
            field_name, field_type, field_value = attribute
            if field_type == "bool" or field_type == "int":
                feature.SetField(field_name, int(field_value))
            elif field_type == "float":
                feature.SetField(field_name, float(field_value))
            elif field_type == "str":
                feature.SetField(field_name, str(field_value))
            else:
                raise ValueError(f"Field type {field_type} not supported")

        layer.SetFeature(feature)
        feature = None
        ds_vector.SyncToDisk()
        ds_vector = None

    return out_path


def raster_to_vector_extent(
    raster: Union[str, gdal.Dataset, list[Union[str, gdal.Dataset]]],
    out_path: Optional[Union[str, List[str]]] = None,
    metadata: bool = False,
    latlng: bool = False,
    *,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> Union[str, List[str], gdal.Dataset]:
    """Gets the footprints of a raster or a list of rasters.

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be shifted.
    out_path : Optional[str, Union[str, List[str]]], optional
        The path to the output raster. If None, the raster is created in memory., default: None
    metadata : bool, optional
        If True, metadata is added to the output raster., default: False
    latlng : bool, optional
        If True, the footprints are returned in lat/lon coordinates. If False, the footprints are returned in projected coordinates., default: True
    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists., default: True
    prefix : str, optional
        The prefix to be added to the output raster name., default: ""
    suffix : str, optional
        The suffix to be added to the output raster name., default: ""
    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name., default: False
    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name., default: False

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the shifted raster(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(metadata, [bool], "metadata")
    utils_base._type_check(latlng, [bool], "latlng")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    in_paths = utils_io._get_input_paths(raster, "raster") # type: ignore
    out_paths = utils_io._get_output_paths(
        in_paths, # type: ignore
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    footprints = []
    for idx, in_raster in enumerate(in_paths):
        footprint = _raster_to_vector_extent(
            in_raster,
            out_path=out_paths[idx],
            metadata=metadata,
            latlng=latlng,
            overwrite=overwrite,
            prefix=prefix,
            suffix=suffix,
            add_uuid=add_uuid,
            add_timestamp=add_timestamp,
        )

        footprints.append(footprint)

    if input_is_list:
        return footprints

    return footprints[0]
