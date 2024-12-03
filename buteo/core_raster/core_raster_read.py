"""###. Basic functionality for working with rasters. ###"""

# Standard library
from typing import Optional, Union, List, Sequence, Tuple

# External
from osgeo import gdal
import numpy as np

# Internal
from buteo.utils import (
    utils_base,
    utils_path,
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


def _open_raster(
    raster: Union[str, gdal.Dataset],
    *,
    writeable: bool = False,
) -> gdal.Dataset:
    """Opens a raster in read or write mode.

    Parameters
    ----------
    raster : str or gdal.Dataset
        A path to a raster or a GDAL dataset
    writeable : bool, optional
        If True, opens in write mode. Default: False

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

    # if already opened
    if isinstance(raster, gdal.Dataset):
        return raster

    if not utils_path._check_file_exists(raster):
        raise ValueError(f"Input raster does not exist: {raster}")

    if raster.startswith("/vsizip/"):
        writeable = False

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    dataset = gdal.Open(raster, gdal.GF_Write if writeable else gdal.GF_Read)
    gdal.PopErrorHandler()

    if dataset is None:
        raise ValueError(f"Could not open raster: {raster}")

    return dataset


def open_raster(
    raster: Union[str, gdal.Dataset, Sequence[Union[str, gdal.Dataset]]],
    *,
    writeable: bool = False,
) -> Union[gdal.Dataset, List[gdal.Dataset]]:
    """Opens one or more rasters in read or write mode.

    Parameters
    ----------
    raster : str, gdal.Dataset, or Sequence[Union[str, gdal.Dataset]]
        Path(s) to raster(s) or GDAL dataset(s)
    writeable : bool, optional
        Open in write mode. Default: False

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

    input_is_sequence = isinstance(raster, Sequence) and not isinstance(raster, str)
    rasters = utils_io._get_input_paths(raster, "raster") # type: ignore

    opened = []
    for r in rasters:
        try:
            opened.append(_open_raster(r, writeable=writeable))
        except Exception as e:
            raise ValueError(f"Could not open raster: {r}") from e

    return opened if input_is_sequence else opened[0]


def check_raster_has_crs(raster: Union[str, gdal.Dataset]) -> bool:
    """Check if a raster has a defined coordinate reference system.

    Parameters
    ----------
    raster : str or gdal.Dataset
        Path to raster or GDAL dataset

    Returns
    -------
    bool
        True if raster has a defined CRS, False otherwise
    """
    dataset = _open_raster(raster)
    projection = dataset.GetProjection()

    if isinstance(raster, str):
        dataset = None

    return bool(projection)
