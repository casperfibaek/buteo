"""### Calculate zonal statistics between a raster and a vector. ###

"""

# Standard library
from typing import Union, List, Optional

# External
import numpy as np
from osgeo import gdal, ogr
from numba import prange, jit

# Internal
from buteo.utils import (
    utils_path,
    utils_gdal,
    utils_projection,
)
from buteo.raster.core_raster import get_metadata_raster
from buteo.vector.core_vector import _get_basic_metadata_vector, vector_create_attribute_from_fid
from buteo.vector.rasterize import vector_rasterize
from buteo.raster.align import _raster_align_to_reference
from buteo.raster.core_raster_io import raster_to_array, array_to_raster
from buteo.raster.clip import raster_clip


# @jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def _calculate_zonal_statistics(
    raster_arr: np.ndarray,
    vector_arr: np.ndarray,
    zones: np.ndarray,
    statistics: List[str],
) -> np.ndarray:
    """Internal. Calculate the zonal statistics."""
    # statistics
    stats = np.zeros((len(zones), len(statistics)), dtype=np.float32)

    # import pdb; pdb.set_trace()

    # calculate statistics
    for zone_idx in prange(len(zones)):
        zone = zones[zone_idx]
        if zone == 0:
            continue

        mask = vector_arr == zone
        for stat_idx, stat in enumerate(statistics):
            if stat == "mean":
                stats[zone_idx][stat_idx] = np.mean(raster_arr[mask])
            elif stat == "median":
                stats[zone_idx][stat_idx] = np.median(raster_arr[mask])
            elif stat == "std":
                stats[zone_idx][stat_idx] = np.std(raster_arr[mask])
            elif stat == "min":
                stats[zone_idx][stat_idx] = np.min(raster_arr[mask])
            elif stat == "max":
                stats[zone_idx][stat_idx] = np.max(raster_arr[mask])
            elif stat == "sum":
                stats[zone_idx][stat_idx] = np.sum(raster_arr[mask])

    return stats


def _raster_get_zonal_statistics(
    raster: Union[str, gdal.Dataset],
    vector: Union[str, gdal.Dataset],
    statistics: List[str] = ["mean"],
    out_path: Optional[str] = None,
) -> str:
    """TODO: Get zonal statistics between a raster and a vector."""
    assert isinstance(raster, (str, gdal.Dataset)), f"Invalid raster. {raster}"
    assert isinstance(vector, (str, ogr.DataSource)), f"Invalid vector. {vector}"
    assert isinstance(statistics, list), f"Invalid statistics. {statistics}"
    assert all([isinstance(stat, str) for stat in statistics]), f"Invalid statistics. {statistics}"

    if out_path is None:
        out_path = utils_path._get_temp_filepath("zonal_statistics.tif")

    raster_metadata = get_metadata_raster(raster)
    vector_metadata = _get_basic_metadata_vector(vector)

    # assert the projection is the same
    assert utils_projection._check_projections_match(raster_metadata["projection_osr"], vector_metadata["projection_osr"]), "Projections do not match."

    # clip raster to vector
    raster_clipped = raster_clip(raster, vector, add_timestamp=True, add_uuid=True)
    raster_clipped_metadata = get_metadata_raster(raster_clipped)

    vector_updated = vector_create_attribute_from_fid(vector, "_zonal_id")
    import pdb; pdb.set_trace()


    # rasterize vector
    rasterized_vector = vector_rasterize(
        vector,
        pixel_size=raster_clipped_metadata["pixel_size"],
        projection=raster_clipped_metadata["projection_osr"],
        extent=raster_clipped,
        out_path=out_path,
        burn_value=None,
        attribute="_zonal_id",
        dtype="int32",
    )

    rasterized_vector_aligned = _raster_align_to_reference(rasterized_vector, raster_clipped)[0]

    utils_gdal.delete_dataset_if_in_memory(rasterized_vector)

    # calculate zonal statistics
    rasterized_vector_arr = raster_to_array(rasterized_vector_aligned)
    raster_arr = raster_to_array(raster_clipped)

    utils_gdal.delete_dataset_if_in_memory(raster_clipped)
    utils_gdal.delete_dataset_if_in_memory(rasterized_vector_aligned)

    # zones
    zones = np.unique(rasterized_vector_arr)

    stats = _calculate_zonal_statistics(
        raster_arr,
        rasterized_vector_arr,
        zones,
        statistics,
    )
