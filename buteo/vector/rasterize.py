"""
### Rasterize vectors. ###

Functions to rasterize vectors to rasters.

TODO:
    * Add support for projections
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import gdal, ogr, osr
import numpy as np

# Internal
from buteo.utils import (
    utils_gdal,
    utils_path,
    utils_projection,
    utils_translate,
)
from buteo.raster.reproject import raster_reproject
from buteo.vector.reproject import vector_reproject
from buteo.vector import core_vector
from buteo.raster import core_raster



def vector_rasterize(
    vector: Union[str, ogr.DataSource],
    pixel_size: Union[float, int],
    out_path: Optional[str] = None,
    extent: Optional[List[Union[float, int]]] = None,
    projection: Optional[Union[str, int, osr.SpatialReference, gdal.Dataset, ogr.DataSource]] = None,
    all_touch: bool = False,
    dtype: Union[str, np.dtype] = "uint8",
    optim: str = "raster",
    band: int = 1,
    fill_value: Union[int, float] = 0,
    nodata_value: Optional[Union[int, float]] = None,
    check_memory: bool = True,
    burn_value=1,
    attribute=None,
) -> str:
    """
    Rasterize a vector to a raster.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        The vector to rasterize.
    
    pixel_size : Union[float, int]
        The pixel size of the raster.
    
    out_path : Optional[str], optional
        Path to output raster. Default: None

    extent : Optional[List[Union[float, int]]], optional
        Extent of raster. Default: None

    projection : Optional[Union[str, int, osr.SpatialReference, gdal.Dataset, ogr.DataSource]], optional
        Projection of raster. Default: None

    all_touch : bool, optional
        All pixel touch? Default: False

    dtype : Union[str, np.dtype], optional
        Data type of raster. Default: "uint8"

    optim : str, optional
        Optimization for raster or vector? Default: "raster"

    band : int, optional
        Band to rasterize. Default: 1
    
    fill_value : Union[int, float], optional
        Fill value. Default: 0

    nodata_value : Optional[Union[int, float]], optional
        Nodata value. Default: None

    check_memory : bool, optional
        Check memory? Default: True

    burn_value : int, optional
        Burn value (The value to burn into the raster). Default: 1

    attribute : Optional[str], optional
        Attribute to rasterize. Default: None
        This will burn the value of the attribute into the raster.

    Returns
    -------
    str
        Path to output raster.
    """
    vector_fn = vector

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_rasterized")

    if projection is not None:
        projection = utils_projection.parse_projection(projection)

        vector_fn = vector_reproject(vector, projection)

        if isinstance(extent, (gdal.Dataset, ogr.DataSource)):
            if utils_gdal._check_is_raster(extent):
                extent = raster_reproject(extent, projection, add_uuid=True, suffix="_reprojected")
            else:
                extent = vector_reproject(extent, projection, add_uuid=True, suffix="_reprojected")

        if isinstance(pixel_size, (gdal.Dataset, ogr.DataSource)):
            if utils_gdal._check_is_raster(pixel_size):
                pixel_size = raster_reproject(pixel_size, projection, add_uuid=True, suffix="_reprojected")
            else:
                pixel_size = vector_reproject(pixel_size, projection, add_uuid=True, suffix="_reprojected")

    if projection is None and isinstance(pixel_size, (gdal.Dataset, ogr.DataSource)):
        if extent is not None:
            extent = raster_reproject(extent, pixel_size, add_uuid=True, suffix="_reprojected")

        vector_fn = vector_reproject(vector, pixel_size, add_uuid=True, suffix="_reprojected")

    # Open the data source and read in the extent
    source_ds = core_vector._vector_open(vector_fn)
    source_meta = core_vector._vector_to_metadata(vector_fn)
    source_layer = source_ds.GetLayer()
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    if isinstance(pixel_size, (gdal.Dataset, str)):
        pixel_size_x = core_raster._raster_open(pixel_size).GetGeoTransform()[1]
        pixel_size_y = abs(core_raster._raster_open(pixel_size).GetGeoTransform()[5])
    elif isinstance(pixel_size, (int, float)):
        pixel_size_x = pixel_size
        pixel_size_y = pixel_size
    elif isinstance(pixel_size, (list, tuple)):
        pixel_size_x, pixel_size_y = pixel_size
    else:
        raise ValueError("pixel_size must be either a gdal.Dataset or a tuple of (x, y)")

    # Create the destination data source
    x_res = int((x_max - x_min) / pixel_size_x)
    y_res = int((y_max - y_min) / pixel_size_y)

    if extent is not None:
        extent_vector = core_vector._vector_to_metadata(extent)
        extent_dict = extent_vector["bbox_dict"]
        x_res = int((extent_dict["x_max"] - extent_dict["x_min"]) / abs(pixel_size_x))
        y_res = int((extent_dict["y_max"] - extent_dict["y_min"]) / abs(pixel_size_y))
        x_min = extent_dict["x_min"]
        y_max = extent_dict["y_max"]

    if check_memory is False:
        gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "FALSE")

    try:
        target_ds = gdal.GetDriverByName("GTiff").Create(
            out_path,
            x_res,
            y_res,
            1,
            utils_translate._translate_dtype_gdal_to_numpy(dtype),
        )
    finally:
        gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "TRUE")

    if target_ds is None:
        raise RuntimeError("Unable to rasterize.")

    target_ds.SetGeoTransform((x_min, pixel_size_x, 0, y_max, 0, -1 * abs(pixel_size_y)))
    target_ds.SetProjection(source_meta["projection_wkt"])

    band = target_ds.GetRasterBand(1)

    if nodata_value is not None:
        band.SetNoDataValue(nodata_value)
    else:
        band.Fill(fill_value)

    options = []
    if all_touch is True:
        options.append("ALL_TOUCHED=TRUE")
    else:
        options.append("ALL_TOUCHED=FALSE")

    if optim == "raster":
        options.append("OPTIM=RASTER")
    elif optim == "vector":
        options.append("OPTIM=VECTOR")
    else:
        options.append("OPTIM=AUTO")

    if attribute is None:
        gdal.RasterizeLayer(
            target_ds,
            [1],
            source_layer,
            burn_values=[burn_value],
            options=options,
        )
    else:
        options.append(f"ATTRIBUTE={attribute}")
        gdal.RasterizeLayer(
            target_ds, [1], source_layer, options=utils_gdal._get_default_creation_options(options)
        )

    return out_path
