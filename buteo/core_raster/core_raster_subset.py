"""### Basic IO functions for working with Rasters. ###"""

# Standard library
from typing import List, Optional, Union

# External
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_io,
    utils_gdal,
    utils_path,
)
from buteo.core_raster.core_raster_read import _open_raster



def raster_extract_bands(
    raster: Union[str, gdal.Dataset],
    band: Union[int, List[int]],
    out_path: Optional[str] = None,
    overwrite: bool = True,
    creation_options: Union[List[str], None] = None,
) -> str:
    """Extract a band from a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset
        The raster to extract the band from.

    band : int or list
        The band to extract. Can be a single band or a list of bands.

    out_path : str, optional
        The output path. If None, a temporary file will be created.

    overwrite : bool, optional
        If True, the output raster will be overwritten if it exists. Default: True.

    creation_options : list or None, optional
        The creation options for the output raster. Default: None.

    Returns
    -------
    str
        The path to the output raster.
    """
    assert utils_gdal._check_is_raster(raster), "Raster is not valid."

    if isinstance(band, int):
        band = [band]

    assert all([isinstance(b, int) for b in band]), "Band must be an integer or a list of integers."

    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            utils_gdal._get_path_from_dataset(raster),
            ext="tif",
        )
    else:
        assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), (
            f"Output path {out_path} is not valid or already exists. "
        )

    utils_io._delete_if_required(out_path, overwrite)

    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)

    src_ds = _open_raster(raster)
    dst_ds = driver.Create(
        out_path,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        len(band),
        src_ds.GetRasterBand(1).DataType,
        utils_gdal._get_default_creation_options(creation_options),
    )

    dst_ds.SetProjection(src_ds.GetProjection())
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())

    for idx, b in enumerate(band):
        dst_ds.GetRasterBand(idx + 1).WriteArray(src_ds.GetRasterBand(b).ReadAsArray())

    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    return out_path
