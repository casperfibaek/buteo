"""
This module provides varoius indices for Sentinel 2 imagery.

TODO:
    Convert to numpy functions.
    Handle resampling
"""

import numpy as np


def chlre(b_05, b_08):
    """ Red Edge Chlorophyll Index
        Calc: B05 / B08
    """
    return np.divide(b_05, b_08)


def rendvi(b_08, b_06):
    """ Red Edge NDVI
        Calc: (B08 - B06) / (B08 + B06)
    """
    return np.divide((b_08 - b_06), (b_08 + b_06))


def s2_rep(b_07, b_04, b_05, b_06):
    """ Sentinel 2 Red Edge Position
        Calc: 705 + 35 * ((((B07 + B04) / 2) - B05) / (B06 - B05))
    """
    return 705 + 35 * ((((b_07 + b_04) / 2) - b_05) / (b_06 - b_05))


def ireci(b_07, b_04, b_05, b_06):
    """ Red Edge Chlorophyll Index
        Calc: (B07 - B04) * B06 / B05
    """
    return (b_07 - b_04) * b_06 / b_05


def mcari(b_05, b_04, b_03):
    """ Modified Chlorophyll Absorption in Reflectance Index
        Calc: (B05 - B04) - 0.2 * (B05 - B03) * (B05 / B04)
    """
    return (b_05 - b_04) - 0.2 * (b_05 - b_03) * (b_05 / b_04)


def arvi(b_08, b_04, b_02):
    """ Atmospherically Resistant Vegetation Index
        Calc: (B08 - b) / (B08 + b), b = 2 * B04 - B02
    """
    b_calc = 2 * b_04 - b_02
    return (b_08 - b_calc) / (b_08 + b_calc)


def savi(b_08, b_04):
    """ Soil Adjusted Vegetation Index
        Calc: ((B08 - B04) / ((B08 + B04) + 0.428)) * 1.856
    """
    return ((b_08 - b_04) / ((b_08 + b_04) + 0.428)) * 1.856


def msavi2(b_08, b_04):
    """ Modified Soil Adjusted Vegetation Index v2
        Calc: (2 * B08 + 1 - sqrt((2 * B08 + 1) ** 2) - 8 * (B08 - B04)) / 2
    """
    return (2 * b_08 + 1 - np.sqrt((2 * b_08 + 1) ** 2) - 8 * (b_08 - b_04)) / 2


def gndvi(b_08, b_03):
    """ Green Normalized Difference Vegetation Index
        Calc: (B08 - B03) / (B08 + B03)
    """
    return (b_08 - b_03) / (b_08 + b_03)


def ndvi(b_08, b_04):
    """ Normalized Difference Vegetation Index
        Calc: (B08 - B04) / (B08 + B04)
    """
    return (b_08 - b_04) / (b_08 + b_04)


def moist(b_8A, b_11):
    """ Soil Moisture Index
        Calc: (B8A - B11) / (B8A + B11)
    """
    return (b_8A - b_11) / (b_8A + b_11)


def ndwi(b_08, b_11):
    """ Normalized Difference Water Index
        Calc: (B08 - B11) / (B08 + B11)
    """
    return (b_08 - b_11) / (b_08 + b_11)


def ndwi(b_03, b_08):
    """ Normalized Difference Water Index v2
        Calc: (B03 - B08) / (B03 + B08)
    """
    return (b_03 - b_08) / (b_03 + b_08)


def nbr(b_08, b_12):
    """ Normalized Burn Ratio
        Calc: (B08 - B12) / (B08 + B12)
    """
    return (b_08 - b_12) / (b_08 + b_12)


def nvei(b_02, b_04):
    """ Non-elimination vegetation index
        Calc: (B02 - B04) / (B08 + B04)
    """
    return (b_02 - b_04) / (b_02 + b_04)


def nbai(b_02, b_08, b_12): # Built-up area index
    """ Built-up area index
        Calc: (B12 - d) / (B12 + d), d = B08 / B02
    """
    d_calc = b_08 / b_02
    return (b_12 - d_calc) / (b_12 + d_calc)


def brba(b_03, b_08):
    """ Band ratio for built-up areas
        Calc: (B03 / B08)
    """
    return b_03 / b_08


def ndbi(b_11, b_08):
    """ Normalised difference built-up index
        Calc: (B11 - B08) / (B11 + B08)
    """
    return (b_11 - b_08) / (b_11 + b_08)


def blfei(b_03, b_04, b_11, b_12):
    """ Built-up features extraction
        Calc: (bix - B11) / (bix + B11), bix = (B03 + B04 + B12) / 3
    """
    bix_calc = (b_03 + b_04 + b_12) / 3
    return (bix_calc - b_11) / (bix_calc + b_11)


def ibi(b_03, b_04, b_08, b_11):
    """ Built-up features extraction
        Calc: (ndbi - ((savi + ndwi2) / 2)) / (ndbi + ((savi + ndwi2) / 2))
            savi = ((B08 - B04) / ((B08 + B04) + 0.428)) * 1.856
            ndwi2 = (B03 - B08) / (B03 + B08)
            ndbi = (B11 - B08) / (B11 + B08
    """
    savi_calc = savi(b_08, b_04)
    ndwi2_calc = ndwi(b_03, b_08)
    ndbi_calc = ndbi(b_11, b_08)

    return (ndbi_calc - (savi_calc + ndwi2_calc) / 2) / (ndbi_calc + (savi_calc + ndwi2_calc) / 2)
