""" ### Coregistering images. ### """ 

# Standard library
from typing import Union, Optional, List

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.raster.core_raster_io import raster_to_array, array_to_raster
from buteo.raster.align import _raster_align_to_reference
from buteo.raster.reproject import _raster_reproject
from buteo.utils.utils_projection import _check_projections_match
from buteo.utils.utils_gdal import delete_dataset_if_in_memory



def coregister_images_efolki(
    master: Union[str, gdal.Dataset],
    slave: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    iteration: int = 4,
    radius: List[int] = [16, 8],
    rank: int = 4,
    levels: int = 5,
    band_to_base_master: int = 1,
    band_to_base_slave: int = 1,
    mask: Optional[np.ndarray] = None,
    fill_value: Optional[Union[int, float]] = None,
    resample_alg="nearest",
):
    """Coregister two images using the EFolki method. This method is based on the paper:

    "GeFolki: A Generic and Efficient Method for Optical Image Registration"

    Parameters
    ----------
    master : str
        Path to the master image.

    slave : str
        Path to the slave image.

    out_path : str
        Path to the output file. If None, the output will be written to a temporary file.

    iteration : int
        Number of iterations to run the algorithm.

    radius : list
        List of two integers specifying the radius of the search window.

    rank : int
        Rank of the algorithm.

    levels : int
        Number of levels to run the algorithm.

    band_to_base_master : int
        Band to use as the base for the coregistration.

    band_to_base_slave : int
        Band to use as the base for the coregistration.

    mask : np.ndarray
        Mask to apply to the coregistration.

    fill_value : int
        Fill value to use for the master and slave images.

    resample_alg : str
        The resampling algorithm, default: "nearest".

    Returns
    -------
    out_path : str
        Path to the output file.
    """
    try:
        from .gefolki import BurtOF, EFolkiIter, wrapData # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Using the GeFolki tools requires skimage and scipy. Please install them using `pip install scikit-image scipy`") from exc

    if not _check_projections_match(master, slave):
        reprojected = True
        reprojected_slave = _raster_reproject(slave, projection=master, resample_alg_gdal=resample_alg, copy_if_same=False)
    else:
        reprojected = False
        reprojected_slave = slave

    aligned_slave = _raster_align_to_reference(reprojected_slave, master, resample_alg=resample_alg)

    master_arr = raster_to_array(master, filled=True, fill_value=fill_value, cast=np.float32) # QB
    slave_arr = raster_to_array(aligned_slave, filled=True, fill_value=fill_value, cast=np.float32) # WV

    if reprojected:
        delete_dataset_if_in_memory(reprojected_slave)
        delete_dataset_if_in_memory(aligned_slave)

    if mask is not None:
        master_arr = master_arr * mask
        slave_arr = slave_arr * mask
    elif fill_value is not None:
        mask = np.logical_and(master_arr != fill_value, slave_arr != fill_value)
        master_arr = master_arr * mask
        slave_arr = slave_arr * mask

    EFolki = BurtOF(EFolkiIter)

    uu, vv = EFolki(
        slave_arr[:, :, band_to_base_slave - 1],
        master_arr[:, :, band_to_base_master - 1],
        iteration=iteration,
        radius=radius,
        rank=rank,
        levels=levels,
    )

    QBrecalee = wrapData(master_arr[:, :, band_to_base_master - 1], uu, vv)
    array_to_raster(QBrecalee[:, :, np.newaxis], reference=master, out_path=out_path)

    return out_path


def coregister_images_gefolki(
    master: Union[str, gdal.Dataset],
    slave: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    iteration: int = 4,
    radius: List[int] = [16, 8],
    rank: int = 4,
    levels: int = 5,
    band_to_base_master: int = 1,
    band_to_base_slave: int = 1,
    mask: Optional[np.ndarray] = None,
    fill_value: Optional[Union[int, float]] = None,
    resample_alg="nearest",
):
    """Coregister two images using the GeFolki method. This method is based on the paper:

    "GeFolki: A Generic and Efficient Method for Optical Image Registration"

    Parameters
    ----------
    master : str
        Path to the master image.

    slave : str
        Path to the slave image.

    out_path : str
        Path to the output file. If None, the output will be written to a temporary file.

    iteration : int
        Number of iterations to run the algorithm.

    radius : list
        List of two integers specifying the radius of the search window.

    rank : int
        Rank of the algorithm.

    levels : int
        Number of levels to run the algorithm.

    band_to_base_master : int
        Band to use as the base for the coregistration.

    band_to_base_slave : int
        Band to use as the base for the coregistration.

    mask : np.ndarray
        Mask to apply to the coregistration.

    fill_value : int
        Fill value to use for the master and slave images.

    Returns
    -------
    out_path : str
        Path to the output file.
    """
    try:
        from .gefolki import BurtOF, GEFolkiIter, wrapData # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Using the GeFolki tools requires skimage and scipy. Please install them using `pip install scikit-image scipy`") from exc

    if not _check_projections_match(master, slave):
        reprojected = True
        reprojected_slave = _raster_reproject(slave, projection=master, resample_alg_gdal=resample_alg, copy_if_same=False)
    else:
        reprojected = False
        reprojected_slave = slave

    aligned_slave = _raster_align_to_reference(reprojected_slave, master, resample_alg=resample_alg)

    master_arr = raster_to_array(master, filled=True, fill_value=fill_value, cast=np.float32) # QB
    slave_arr = raster_to_array(aligned_slave, filled=True, fill_value=fill_value, cast=np.float32) # WV

    if reprojected:
        delete_dataset_if_in_memory(reprojected_slave)
        delete_dataset_if_in_memory(aligned_slave)

    if mask is not None:
        master_arr = master_arr * mask
        slave_arr = slave_arr * mask
    elif fill_value is not None:
        mask = np.logical_and(master_arr != fill_value, slave_arr != fill_value)
        master_arr = master_arr * mask
        slave_arr = slave_arr * mask

    GEFolki = BurtOF(GEFolkiIter)

    uu, vv = GEFolki(
        slave_arr[:, :, band_to_base_slave - 1],
        master_arr[:, :, band_to_base_master - 1],
        iteration=iteration,
        radius=radius,
        rank=rank,
        levels=levels,
    )

    QBrecalee = wrapData(master_arr[:, :, band_to_base_master - 1], uu, vv)

    array_to_raster(QBrecalee[:, :, np.newaxis], reference=master, out_path=out_path)

    return out_path
