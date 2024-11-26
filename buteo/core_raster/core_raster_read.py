"""###. Basic functionality for working with rasters. ###"""

# Standard library
import warnings
from typing import Optional, Union, List, Sequence, Tuple

# External
from osgeo import gdal, osr
import numpy as np

# Internal
from buteo.utils import (
    utils_base,
    utils_path,
    utils_projection,
    utils_io,
)



def _read_raster_band(
    raster: gdal.Dataset,
    band_idx: int,
    pixel_offsets: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Read a single band from a raster dataset. 1-based indexing.

    Parameters
    ----------
    raster : gdal.Dataset
        The raster dataset to read from

    band_idx : int
        The band index to read (1-based, as GDAL uses 1-based indexing)

    pixel_offsets : tuple, optional
        (x_offset, y_offset, x_size, y_size)

    Returns
    -------
    np.ndarray
        The band data as a 2D array

    Raises
    ------
    ValueError
        If band_idx is out of bounds
    """
    utils_base._type_check(raster, [gdal.Dataset], "raster")
    utils_base._type_check(band_idx, [int], "band_idx")
    utils_base._type_check(pixel_offsets, [tuple, type(None)], "pixel_offsets")

    band_count = raster.RasterCount
    if band_idx < 1 or band_idx > band_count:
        raise ValueError(f"Band index {band_idx} is out of bounds. Raster has {band_count} bands (1-{band_count})")

    band = raster.GetRasterBand(band_idx)

    if pixel_offsets:
        x_off, y_off, x_size, y_size = pixel_offsets
        return band.ReadAsArray(x_off, y_off, x_size, y_size)

    return band.ReadAsArray()


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
