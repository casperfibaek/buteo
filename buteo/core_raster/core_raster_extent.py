"""###. Basic functionality for working with rasters. ###"""

# Standard library
import os
from typing import List, Optional, Union

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
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


def raster_to_extent(
    raster: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    *,
    latlng: bool = False,
    overwrite: bool = True,
) -> str:
    """Creates a vector file with the extent polygon of a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset
        Input raster
    out_path : str, optional
        Output vector path. If None, saves to temp file
    latlng : bool, optional
        Convert extent to lat/lng coordinates. Default: False
    overwrite : bool, optional
        Overwrite existing file. Default: True

    Returns
    -------
    str
        Path to output vector file

    Raises
    ------
    TypeError
        If input types are invalid
    ValueError
        If output path is invalid or processing fails
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")
    utils_base._type_check(out_path, [str, type(None)], "out_path")
    utils_base._type_check(latlng, [bool], "latlng")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if out_path is None:
        out_path = utils_path._get_temp_filepath("extent.gpkg", add_timestamp=True)
    elif not utils_path._check_is_valid_output_filepath(out_path):
        raise ValueError(f"Invalid output path: {out_path}")

    # Open and get raster info
    ds = _open_raster(raster)
    info = _get_basic_info_raster(ds)

    # Create extent geometry
    bbox = utils_bbox._get_bbox_from_geotransform(
        info["transform"], ds.RasterXSize, ds.RasterYSize
    )
    extent = utils_bbox._get_geom_from_bbox(bbox)
    extent.AssignSpatialReference(info["projection_osr"])

    # Convert to latlng if requested
    if latlng:
        target_srs = utils_projection._get_default_projection_osr()
        extent.TransformTo(target_srs)
        out_srs = target_srs
    else:
        out_srs = info["projection_osr"]

    # Create vector file
    driver = ogr.GetDriverByName(utils_gdal._get_driver_name_from_path(out_path))
    ds_out = driver.CreateDataSource(out_path)
    layer = ds_out.CreateLayer("extent", out_srs, ogr.wkbPolygon)

    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(extent)
    layer.CreateFeature(feat)

    ds_out = None
    return out_path


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


def raster_get_footprints(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    latlng: bool = True,
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    out_format: str = "gpkg",
) -> Union[str, List[str], gdal.Dataset]:
    """Gets the footprints of a raster or a list of rasters.

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be shifted.

    latlng : bool, optional
        If True, the footprints are returned in lat/lon coordinates. If False, the footprints are returned in projected coordinates., default: True

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., default: None

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

    out_format : str, optional
        The output format of the raster. If None, the format is inferred from the output path., default: "gpkg"

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the shifted raster(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(latlng, [bool], "latlng")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    in_paths = utils_io._get_input_paths(raster, "raster")
    out_paths = utils_io._get_output_paths(
        in_paths,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext=out_format,
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    footprints = []
    for idx, in_raster in enumerate(in_paths):
        metadata = get_metadata_raster(in_raster)
        name = os.path.splitext(os.path.basename(metadata["name"]))[0]

        # Projections
        projection_osr = metadata["projection_osr"]
        projection_latlng = utils_projection._get_default_projection_osr()

        # Bounding boxes
        bbox = metadata["bbox"]

        if latlng:
            footprint_geom = utils_bbox._get_bounds_from_bbox_as_geom(
                bbox,
                projection_osr,
            )
            footprint = utils_bbox._get_vector_from_geom(
                footprint_geom,
                projection_osr=projection_latlng,
                out_path=out_paths[idx],
                name=name,
            )
        else:
            footprint_geom = utils_bbox._get_geom_from_bbox(
                bbox,
            )
            footprint = utils_bbox._get_vector_from_geom(
                footprint_geom,
                projection_osr=projection_osr,
                out_path=out_paths[idx],
                name=name,
            )

        footprints.append(footprint)

    if input_is_list:
        return footprints

    return footprints[0]


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
