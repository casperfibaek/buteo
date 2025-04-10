"""###. Basic functionality for working with rasters. ###"""

# Standard library
import os
from typing import Union, Dict, Any
from warnings import warn

# External
from osgeo import gdal, osr

# Internal
from buteo.utils import (
    utils_base,
    # utils_bbox, # Removed old import
    utils_path,
    utils_gdal,
    utils_translate,
    utils_projection,
)
# Import necessary bbox functions from their new locations
from buteo.bbox.operations import _get_bbox_from_geotransform, _get_gdal_bbox_from_ogr_bbox
from buteo.bbox.conversion import _get_geom_from_bbox, _get_bounds_from_bbox_as_geom

from buteo.core_raster.core_raster_read import _open_raster



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

    if not projection_wkt:
        projection_wkt = utils_projection._get_default_projection()
        warn("Dataset has no projection defined. Using default projection.")

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
    # Use imported _get_bbox_from_geotransform
    bbox = _get_bbox_from_geotransform(transform, dataset.RasterXSize, dataset.RasterYSize)
    # Use imported _get_geom_from_bbox
    bounds_raster = _get_geom_from_bbox(bbox)

    try:
        bbox_latlng = utils_projection.reproject_bbox(
            bbox,
            projection_osr,
            utils_projection._get_default_projection_osr()
        )
        # Use imported _get_bounds_from_bbox_as_geom
        bounds_latlng = _get_bounds_from_bbox_as_geom(bbox, projection_osr)
    except RuntimeError as e:
        if "Point outside of projection domain" in str(e):
            bbox_latlng = [0.0, 90.0, 0.0, 180.0] # Default WGS84 bbox? Seems arbitrary.
            # Use imported _get_bounds_from_bbox_as_geom (Corrected usage)
            bounds_latlng = _get_bounds_from_bbox_as_geom(
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
        # Use imported _get_gdal_bbox_from_ogr_bbox
        "bbox_gdal": _get_gdal_bbox_from_ogr_bbox(bbox),
        "bbox_gdal_latlng": _get_gdal_bbox_from_ogr_bbox(bbox_latlng),
        "bounds_latlng": bounds_latlng.ExportToWkt(),
        "bounds": bounds_raster.ExportToWkt(),
        "centroid": (centroid.GetX(), centroid.GetY()),
        "centroid_latlng": (centroid_latlng.GetX(), centroid_latlng.GetY()),
        "area": (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]),
        "area_latlng": bounds_latlng.GetArea(),
    }


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

    Examples
    --------
    >>> from buteo.core_raster import core_raster_info
    >>> metadata = core_raster_info.get_metadata_raster("path/to/raster.tif")
    >>> print(metadata)
    >>> {
    >>>     "path": "path/to/raster.tif",
    >>>     "basename": "raster.tif",
    >>>     "name": "raster",
    >>>     "folder": "path/to",
    >>>     "ext": ".tif",
    >>>     "in_memory": False,
    >>>     "driver": "GTiff",
    >>>     "projection_osr": osr.SpatialReference,
    >>>     "projection_wkt": "PROJCS[...]",
    >>>     "geotransform": (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    >>>     "size": (512, 512),
    >>>     "width": 512,
    >>>     "height": 512,
    >>>     "pixel_width": 1.0,
    >>>     "pixel_height": 1.0,
    >>>     "x_min": 0.0,
    >>>     "x_max": 512.0,
    >>>     "y_min": -512.0,
    >>>     "y_max": 0.0,
    >>>     "shape": [512, 512, 1],
    >>>     "bands": 1,
    >>>     "dtype_gdal": 1,
    >>>     "dtype": numpy.uint8,
    >>>     "dtype_name": "uint8",
    >>>     "pixel_size": (1.0, 1.0),
    >>>     "origin": (0.0, 0.0),
    >>>     "nodata": False,
    >>>     "nodata_value": None,
    >>>     "bbox": [0.0, 512.0, -512.0, 0.0],
    >>>     "bbox_latlng": [0.0, 90.0, 0.0, 180.0],
    >>>     "bbox_gdal": (0.0, 1.0, -512.0, 0.0),
    >>>     "bbox_gdal_latlng": (0.0, 90.0, 0.0, 180.0),
    >>>     "bounds_latlng": "POLYGON ((0 90, 0 0, 180 0, 180 90, 0 90))",
    >>>     "bounds": "POLYGON ((0 -512, 512 -512, 512 0, 0 0, 0 -512))",
    >>>     "centroid": (256.0, -256.0),
    >>>     "centroid_latlng": (90.0, 0.0),
    >>>     "area": 262144.0,
    >>>     "area_latlng": 32400.0
    >>> }
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")

    dataset = _open_raster(raster, writeable=False)
    info = _get_basic_info_raster(dataset)
    description = dataset.GetDescription()
    path = utils_path._get_unix_path(description) if description != "" else "in_memory.mem"
    dtype_numpy = utils_translate._translate_dtype_gdal_to_numpy(info["dtype"])

    metadata = {
        "path": path,
        "basename": os.path.basename(path),
        "name": os.path.splitext(os.path.basename(path))[0],
        "folder": os.path.dirname(path),
        "ext": os.path.splitext(path)[1],
        "in_memory": utils_gdal._check_is_dataset_in_memory(raster),
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
        "shape": [info["bands"], dataset.RasterYSize, dataset.RasterXSize],
        "bands": info["bands"],
        "dtype_gdal": info["dtype"],
        "dtype": dtype_numpy,
        "dtype_name": dtype_numpy.name,
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
