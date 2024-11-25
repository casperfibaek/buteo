"""###. Basic functionality for working with rasters. ###"""

# Standard library
import os
import warnings
from typing import List, Optional, Union, Dict, Any, Sequence

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



def _validate_raster_dataset(
    dataset: gdal.Dataset,
    raster_path: str,
    default_projection: Optional[Union[str, int, osr.SpatialReference]] = None,
) -> None:
    """Validates and sets projection for a raster dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        The dataset to validate
    raster_path : str
        Path to the raster file
    default_projection : str, int, or osr.SpatialReference, optional
        The default projection to use if none exists

    Raises
    ------
    ValueError
        If dataset is invalid or projection cannot be set
    """
    utils_base._type_check(dataset, [gdal.Dataset], "dataset")
    utils_base._type_check(raster_path, [str], "raster_path")
    utils_base._type_check(default_projection, [type(None), str, int, osr.SpatialReference], "default_projection")

    if dataset.GetDescription() == "":
        dataset.SetDescription(raster_path)

    if dataset.GetProjectionRef() == "":
        if default_projection is None:
            dataset.SetProjection(utils_projection._get_default_projection())
            dataset.SetGeoTransform([0, 1/dataset.RasterXSize, 0, 0, 0, -1/dataset.RasterYSize])
            warnings.warn(f"Input raster {raster_path} has no projection. Setting to EPSG:4326.", UserWarning)
        else:
            try:
                projection = utils_projection.parse_projection(default_projection)
                dataset.SetProjection(projection.ExportToWkt())
                dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
                warnings.warn(f"Input raster {raster_path} has no projection. Setting to {default_projection}.", UserWarning)
            except Exception as exc:
                raise ValueError(f"Input has no projection and default projection is invalid: {default_projection}") from exc


def _open_raster(
    raster: Union[str, gdal.Dataset],
    *,
    writeable: bool = False,
    default_projection: Optional[Union[str, int, osr.SpatialReference]] = None,
) -> gdal.Dataset:
    """Opens a raster in read or write mode.

    Parameters
    ----------
    raster : str or gdal.Dataset
        A path to a raster or a GDAL dataset
    writeable : bool, optional
        If True, opens in write mode. Default: False
    default_projection : str, int, or osr.SpatialReference, optional
        Default projection if none exists. Default: None

    Returns
    -------
    gdal.Dataset
        The opened raster dataset

    Raises
    ------
    TypeError
        If raster is not str or gdal.Dataset
    ValueError
        If raster path doesn't exist or file cannot be opened
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")
    utils_base._type_check(writeable, [bool], "writeable")
    utils_base._type_check(default_projection, [type(None), str, int, osr.SpatialReference], "default_projection")

    # if already opened
    if isinstance(raster, gdal.Dataset):
        _validate_raster_dataset(raster, raster.GetDescription(), default_projection)
        return raster

    if not utils_path._check_file_exists(raster):
        raise ValueError(f"Input raster does not exist: {raster}")

    if raster.startswith("/vsizip/"):
        writeable = False

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    dataset = gdal.Open(raster, gdal.GF_Write if writeable else gdal.GF_Read)
    gdal.PopErrorHandler()

    _validate_raster_dataset(dataset, raster, default_projection)
    return dataset


def _get_basic_info_raster(dataset: gdal.Dataset) -> Dict[str, Any]:
    """Get basic information from a GDAL dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        The GDAL dataset to extract information from.

    Returns
    -------
    Dict[str, Any]
        Basic raster information including size, projection, and transform.

    Raises
    ------
    ValueError
        If the dataset is invalid or cannot be read.
    """
    utils_base._type_check(dataset, [gdal.Dataset], "dataset")

    transform = dataset.GetGeoTransform()
    projection_wkt = dataset.GetProjectionRef()
    projection_osr = osr.SpatialReference()
    projection_osr.ImportFromWkt(projection_wkt)

    first_band = dataset.GetRasterBand(1)
    dtype = None if first_band is None else first_band.DataType

    return {
        "transform": transform,
        "projection_wkt": projection_wkt,
        "projection_osr": projection_osr,
        "size": (dataset.RasterXSize, dataset.RasterYSize),
        "bands": dataset.RasterCount,
        "dtype": dtype,
    }


def _get_bounds_info_raster(
    dataset: gdal.Dataset,
    projection_osr: osr.SpatialReference,
) -> Dict[str, Any]:
    """Extract bounds and coordinate information from dataset.

    Parameters
    ----------
    dataset : gdal.Dataset
        The GDAL dataset to process
    projection_osr : osr.SpatialReference
        The source projection

    Returns
    -------
    Dict[str, Any]
        Dictionary containing bounds information

    Raises
    ------
    ValueError
        If bounds computation fails
    """
    transform = dataset.GetGeoTransform()
    bbox = utils_bbox._get_bbox_from_geotransform(transform, dataset.RasterXSize, dataset.RasterYSize)
    bounds_raster = utils_bbox._get_geom_from_bbox(bbox)

    try:
        bbox_latlng = utils_projection.reproject_bbox(
            bbox,
            projection_osr,
            utils_projection._get_default_projection_osr()
        )
        bounds_latlng = utils_bbox._get_bounds_from_bbox_as_geom(bbox, projection_osr)
    except RuntimeError as e:
        if "Point outside of projection domain" in str(e):
            bbox_latlng = [0.0, 90.0, 0.0, 180.0]
            bounds_latlng = utils_bbox._get_bounds_from_bbox_as_geom(
                bbox_latlng,
                utils_projection._get_default_projection_osr()
            )
        else:
            raise ValueError("Failed to compute bounds") from e

    centroid = bounds_raster.Centroid()
    centroid_latlng = centroid.Clone()
    centroid_latlng.Transform(
        osr.CoordinateTransformation(
            projection_osr,
            utils_projection._get_default_projection_osr()
        )
    )

    return {
        "bbox": bbox,
        "bbox_latlng": bbox_latlng,
        "bbox_gdal": utils_bbox._get_gdal_bbox_from_ogr_bbox(bbox),
        "bbox_gdal_latlng": utils_bbox._get_gdal_bbox_from_ogr_bbox(bbox_latlng),
        "bounds_latlng": bounds_latlng.ExportToWkt(),
        "bounds_raster": bounds_raster.ExportToWkt(),
        "centroid": (centroid.GetX(), centroid.GetY()),
        "centroid_latlng": (centroid_latlng.GetX(), centroid_latlng.GetY()),
        "area": (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]),
        "area_latlng": bounds_latlng.GetArea(),
    }


def _check_raster_has_nodata(raster: Union[str, gdal.Dataset]) -> bool:
    """Check if raster has nodata values for any band.

    Parameters
    ----------
    raster : str or gdal.Dataset
        Raster dataset or path to raster file

    Returns
    -------
    bool
        True if any band has nodata values, False otherwise

    Raises
    ------
    TypeError
        If raster is not str or gdal.Dataset
    ValueError
        If raster cannot be opened
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")

    dataset = _open_raster(raster, writeable=False)

    for band_idx in range(1, dataset.RasterCount + 1):
        if dataset.GetRasterBand(band_idx).GetNoDataValue() is not None:
            return True

    return False


def check_rasters_have_same_nodata(
    rasters: List[Union[str, gdal.Dataset]],
) -> bool:
    """Verifies whether a list of rasters have the same nodata values.

    Parameters
    ----------
    rasters : List[Union[str, gdal.Dataset]]
        A list of rasters to check.

    Returns
    -------
    bool
        True if all rasters have the same nodata value, False otherwise.

    Raises
    ------
    TypeError
        If rasters is not a list of strings or gdal.Datasets.
    ValueError
        If rasters is empty or if any raster cannot be opened.
    """
    utils_base._type_check(rasters, [[str, gdal.Dataset]], "rasters")

    if len(rasters) == 0:
        raise ValueError("Input rasters list is empty.")

    if len(rasters) == 1:
        return True

    # Open first raster and get nodata value
    first_ds = _open_raster(rasters[0])
    first_nodata = first_ds.GetRasterBand(1).GetNoDataValue()

    # Compare with remaining rasters
    for raster in rasters[1:]:
        ds = _open_raster(raster)
        if ds.GetRasterBand(1).GetNoDataValue() != first_nodata:
            return False

    return True


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


def open_raster(
    raster: Union[str, gdal.Dataset, Sequence[Union[str, gdal.Dataset]]],
    *,
    writeable: bool = False,
    default_projection: int = 4326,
) -> Union[gdal.Dataset, List[gdal.Dataset]]:
    """Opens one or more rasters in read or write mode.

    Parameters
    ----------
    raster : str, gdal.Dataset, or Sequence[Union[str, gdal.Dataset]]
        Path(s) to raster(s) or GDAL dataset(s)
    writeable : bool, optional
        Open in write mode. Default: False
    default_projection : int, optional
        Default projection if none exists. Default: 4326

    Returns
    -------
    Union[gdal.Dataset, List[gdal.Dataset]]
        Single GDAL dataset or list of datasets

    Raises
    ------
    TypeError
        If input types are invalid
    ValueError
        If raster(s) cannot be opened
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(writeable, [bool], "writeable")
    utils_base._type_check(default_projection, [int], "default_projection")

    input_is_sequence = isinstance(raster, Sequence) and not isinstance(raster, str)
    rasters = utils_io._get_input_paths(raster, "raster") # type: ignore

    opened = []
    for r in rasters:
        try:
            opened.append(_open_raster(r, writeable=writeable, default_projection=default_projection))
        except Exception as e:
            raise ValueError(f"Could not open raster: {r}") from e

    return opened if input_is_sequence else opened[0]


def get_metadata_raster(raster: Union[str, gdal.Dataset]) -> Dict[str, Any]:
    """Get metadata from a raster.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The raster to get metadata from

    Returns
    -------
    Dict[str, Any]
        Dictionary containing raster metadata

    Raises
    ------
    TypeError
        If input type is invalid
    ValueError
        If raster is invalid or metadata cannot be extracted
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")

    dataset = _open_raster(raster, writeable=False, default_projection=3857)
    info = _get_basic_info_raster(dataset)
    path = utils_path._get_unix_path(dataset.GetDescription())

    metadata = {
        "path": path,
        "basename": os.path.basename(path),
        "name": os.path.splitext(os.path.basename(path))[0],
        "folder": os.path.dirname(path),
        "ext": os.path.splitext(path)[1],
        "in_memory": utils_path._check_is_valid_mem_filepath(path),
        "driver": dataset.GetDriver().ShortName,
        "projection_osr": info["projection_osr"],
        "projection_wkt": info["projection_wkt"],
        "geotransform": info["transform"],
        "size": info["size"],
        "width": info["size"][0],
        "height": info["size"][1],
        "pixel_width": info["transform"][1],
        "pixel_height": abs(info["transform"][5]),
        "x_min": info["transform"][0],
        "x_max": info["transform"][0] + info["transform"][1] * info["size"][0],
        "y_min": info["transform"][3] + info["transform"][5] * info["size"][1],
        "y_max": info["transform"][3],
        "shape": [dataset.RasterYSize, dataset.RasterXSize, info["bands"]],
        "bands": info["bands"],
        "dtype_gdal": info["dtype"],
        "dtype": utils_translate._translate_dtype_gdal_to_numpy(info["dtype"]),
        "dtype_name": utils_translate._translate_dtype_gdal_to_numpy(info["dtype"]).name,
        "pixel_size": (abs(info["transform"][1]), abs(info["transform"][5])),
        "origin": (info["transform"][0], info["transform"][3]),
        "nodata": False,
        "nodata_value": None,
    }

    bounds_info = _get_bounds_info_raster(dataset, info["projection_osr"])
    metadata.update(bounds_info)

    for band_idx in range(1, info["bands"] + 1):
        nodata = dataset.GetRasterBand(band_idx).GetNoDataValue()
        if nodata is not None:
            metadata["nodata"] = True
            metadata["nodata_value"] = nodata

            break

    return metadata
