"""
This module provides varoius indices for Sentinel 2 imagery.

TODO:
    Convert to numpy functions.
    Handle resampling
"""
# Standard library
import sys; sys.path.append("../../")
from typing import List

# External
import numpy as np


def _all_arrays_are_same_size(arrays: List[np.ndarray]) -> bool:
    """ Check if all arrays are the same size. """
    return all([arrays[0].shape == array.shape for array in arrays])

def s2_index_chlre(b_05: np.ndarray, b_08: np.ndarray):
    """ Red Edge Chlorophyll Index
        Calc: B05 / B08
    """
    assert _all_arrays_are_same_size([b_05, b_08]), "All arrays must be the same size."

    result = np.zeros_like(b_08, dtype=np.float32)
    np.divide(b_05, b_08, out=result, where=b_08 != 0)

    return result

def s2_index_rendvi(b_08: np.ndarray, b_06: np.ndarray):
    """ Red Edge NDVI
        Calc: (B08 - B06) / (B08 + B06)
    """
    assert _all_arrays_are_same_size([b_08, b_06]), "All arrays must be the same size."

    result = np.zeros_like(b_08, dtype=np.float32)

    add = b_08 + b_06
    sub = b_08 - b_06
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_s2_rep(b_07: np.ndarray, b_04: np.ndarray, b_05: np.ndarray, b_06: np.ndarray):
    """ Sentinel 2 Red Edge Position
        Calc: 705 + 35 * ((((B07 + B04) / 2) - B05) / (B06 - B05))
    """
    assert _all_arrays_are_same_size([b_07, b_04, b_05, b_06]), "All arrays must be the same size."

    result = np.zeros_like(b_07, dtype=np.float32)

    sub = b_06 - b_05
    np.divide(705 + 35 * ((((b_07 + b_04) / 2) - b_05) / sub), 1, out=result, where=sub != 0)

    return result

def s2_index_ireci(b_07: np.ndarray, b_04: np.ndarray, b_05: np.ndarray, b_06: np.ndarray):
    """ Red Edge Chlorophyll Index
        Calc: (B07 - B04) * B06 / B05
    """
    assert _all_arrays_are_same_size([b_07, b_04, b_05, b_06]), "All arrays must be the same size."

    result = np.zeros_like(b_07, dtype=np.float32)
    np.divide((b_07 - b_04) * b_06, b_05, out=result, where=b_05 != 0)

    return result

def s2_index_mcari(b_05: np.ndarray, b_04: np.ndarray, b_03: np.ndarray):
    """ Modified Chlorophyll Absorption in Reflectance Index
        Calc: (B05 - B04) - 0.2 * (B05 - B03) * (B05 / B04)
    """
    assert _all_arrays_are_same_size([b_05, b_04, b_03]), "All arrays must be the same size."

    return (b_05 - b_04) - 0.2 * (b_05 - b_03) * (b_05 / b_04)

def s2_index_arvi(b_08: np.ndarray, b_04: np.ndarray, b_02: np.ndarray):
    """ Atmospherically Resistant Vegetation Index
        Calc: (B08 - b) / (B08 + b), b = 2 * B04 - B02
    """
    assert _all_arrays_are_same_size([b_08, b_04, b_02]), "All arrays must be the same size."

    b_calc = 2 * b_04 - b_02
    sub = b_08 - b_calc
    add = b_08 + b_calc

    result = np.zeros_like(b_08, dtype=np.float32)
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_savi(b_08: np.ndarray, b_04: np.ndarray):
    """ Soil Adjusted Vegetation Index
        Calc: ((B08 - B04) / ((B08 + B04) + 0.428)) * 1.856
    """
    assert _all_arrays_are_same_size([b_08, b_04]), "All arrays must be the same size."

    sub = b_08 - b_04
    add = b_08 + b_04 + 0.428

    result = np.zeros_like(b_08, dtype=np.float32)
    np.divide(sub, add, out=result, where=add != 0)

    return result * 1.856

def s2_index_msavi2(b_08: np.ndarray, b_04: np.ndarray):
    """ Modified Soil Adjusted Vegetation Index v2
        Calc: (2 * B08 + 1 - sqrt((2 * B08 + 1) ** 2) - 8 * (B08 - B04)) / 2
    """
    assert _all_arrays_are_same_size([b_08, b_04]), "All arrays must be the same size."

    return (2 * b_08 + 1 - np.sqrt((2 * b_08 + 1) ** 2) - 8 * (b_08 - b_04)) / 2

def s2_index_gndvi(b_08: np.ndarray, b_03: np.ndarray):
    """ Green Normalized Difference Vegetation Index
        Calc: (B08 - B03) / (B08 + B03)
    """
    assert _all_arrays_are_same_size([b_08, b_03]), "All arrays must be the same size."

    result = np.zeros_like(b_08, dtype=np.float32)

    add = b_08 + b_03
    sub = b_08 - b_03
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_ndvi(b_08: np.ndarray, b_04: np.ndarray):
    """ Normalized Difference Vegetation Index
        Calc: (B08 - B04) / (B08 + B04)
    """
    assert _all_arrays_are_same_size([b_08, b_04]), "All arrays must be the same size."

    result = np.zeros_like(b_08, dtype=np.float32)

    add = b_08 + b_04
    sub = b_08 - b_04
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_moist(b_8A: np.ndarray, b_11: np.ndarray):
    """ Soil Moisture Index
        Calc: (B8A - B11) / (B8A + B11)
    """
    assert _all_arrays_are_same_size([b_8A, b_11]), "All arrays must be the same size."

    result = np.zeros_like(b_8A, dtype=np.float32)

    add = b_8A + b_11
    sub = b_8A - b_11
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_ndwi(b_08: np.ndarray, b_11: np.ndarray):
    """ Normalized Difference Water Index
        Calc: (B08 - B11) / (B08 + B11)
    """
    assert _all_arrays_are_same_size([b_08, b_11]), "All arrays must be the same size."

    result = np.zeros_like(b_08, dtype=np.float32)

    add = b_08 + b_11
    sub = b_08 - b_11
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_ndwi_v2(b_03: np.ndarray, b_08: np.ndarray):
    """ Normalized Difference Water Index v2
        Calc: (B03 - B08) / (B03 + B08)
    """
    assert _all_arrays_are_same_size([b_03, b_08]), "All arrays must be the same size."

    result = np.zeros_like(b_03, dtype=np.float32)

    add = b_03 + b_08
    sub = b_03 - b_08
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_nbr(b_08: np.ndarray, b_12: np.ndarray):
    """ Normalized Burn Ratio
        Calc: (B08 - B12) / (B08 + B12)
    """
    assert _all_arrays_are_same_size([b_08, b_12]), "All arrays must be the same size."

    result = np.zeros_like(b_08, dtype=np.float32)

    add = b_08 + b_12
    sub = b_08 - b_12
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_nvei(b_02: np.ndarray, b_04: np.ndarray):
    """ Non-elimination vegetation index
        Calc: (B02 - B04) / (B08 + B04)
    """
    assert _all_arrays_are_same_size([b_02, b_04]), "All arrays must be the same size."

    result = np.zeros_like(b_02, dtype=np.float32)

    add = b_02 + b_04
    sub = b_02 - b_04
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_nbai(b_02: np.ndarray, b_08: np.ndarray, b_12: np.ndarray): # Built-up area index
    """ Built-up area index
        Calc: (B12 - d) / (B12 + d), d = B08 / B02
    """
    assert _all_arrays_are_same_size([b_02, b_08, b_12]), "All arrays must be the same size."

    intermediate = np.zeros_like(b_02, dtype=np.float32)
    result = np.zeros_like(b_02, dtype=np.float32)

    np.divide(b_08, b_02, out=intermediate, where=b_02 != 0)

    add = b_12 + intermediate
    sub = b_12 - intermediate
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_brba(b_03: np.ndarray, b_08: np.ndarray):
    """ Band ratio for built-up areas
        Calc: (B03 / B08)
    """
    assert _all_arrays_are_same_size([b_03, b_08]), "All arrays must be the same size."

    result = np.zeros_like(b_03, dtype=np.float32)
    np.divide(b_03, b_08, out=result, where=b_08 != 0)

    return result

def s2_index_ndbi(b_11: np.ndarray, b_08: np.ndarray):
    """ Normalised difference built-up index
        Calc: (B11 - B08) / (B11 + B08)
    """
    assert _all_arrays_are_same_size([b_11, b_08]), "All arrays must be the same size."

    result = np.zeros_like(b_11, dtype=np.float32)

    add = b_11 + b_08
    sub = b_11 - b_08
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_blfei(b_03: np.ndarray, b_04: np.ndarray, b_11: np.ndarray, b_12: np.ndarray):
    """ Built-up features extraction
        Calc: (bix - B11) / (bix + B11), bix = (B03 + B04 + B12) / 3
    """
    assert _all_arrays_are_same_size([b_03, b_04, b_11, b_12]), "All arrays must be the same size."

    result = np.zeros_like(b_11, dtype=np.float32)

    bix_calc = (b_03 + b_04 + b_12) / 3

    add = bix_calc + b_11
    sub = bix_calc - b_11
    np.divide(sub, add, out=result, where=add != 0)

    return result

def s2_index_ibi(b_03: np.ndarray, b_04: np.ndarray, b_08: np.ndarray, b_11: np.ndarray):
    """ Built-up features extraction
        Calc: (ndbi - ((savi + ndwi2) / 2)) / (ndbi + ((savi + ndwi2) / 2))
            savi = ((B08 - B04) / ((B08 + B04) + 0.428)) * 1.856
            ndwi2 = (B03 - B08) / (B03 + B08)
            ndbi = (B11 - B08) / (B11 + B08
    """
    assert _all_arrays_are_same_size([b_03, b_04, b_08, b_11]), "All arrays must be the same size."

    savi_calc = s2_index_savi(b_08, b_04)
    ndwi2_calc = s2_index_ndwi(b_03, b_08)
    ndbi_calc = s2_index_ndbi(b_11, b_08)

    result = np.zeros_like(b_11, dtype=np.float32)

    add = ndbi_calc + (savi_calc + ndwi2_calc) / 2
    sub = ndbi_calc - (savi_calc + ndwi2_calc) / 2

    np.divide(sub, add, out=result, where=add != 0)

    return result
