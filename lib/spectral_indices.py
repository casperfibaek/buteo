import math
import numpy as np


np.seterr(divide='ignore', invalid='ignore')  # disable warnings for division by zero


# Red Edge Chlorophyll Index
def chlre(B08, B05):
    return np.divide(B05, B08)


# Red Edge NDVI
def rendvi(B08, B06):
    return np.divide((B08 - B06), (B08 + B06))


# Sentinel 2 Red Edge Position
def s2rep(B07, B06, B05, B04):
    return 705 + 35 * np.divide((np.divide((B07 + B04), 2) - B05), (B06 - B05))


# Modified Chlorophyll Absorption in Reflectance Index
def mcari(B05, B04, B03):
    return ((B05 - B04) - 0.2 * (B05 - B03)) * (np.divide(B05, B04))


# Inverted Red Edge Chlorophyll Index
def ireci(B07, B06, B05, B04):
    return np.divide((B07 - B04) * B06, B05)


# Atmospherically Resistant Vegetation Index
def arvi(B08, B04, B02):
    base = (2 * B04 - B02)
    return np.divide((B08 - base), (B08 + base))


# Enhanced vegetation index
def evi(B08, B04, B02):
    return 2.5 * np.divide((B08 - B04), ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0))


# Enhanced vegetation index v2
def evi2(B08, B04):
    return 2.4 * np.divide((B08 - B04), (B08 + B04 + 1.0))


# Soil adjusted vegetation index
def savi(B08, B04):
    return np.divide(B08 - B04, B08 + B04 + 0.428) * 1.856


# Modified soil adjusted vegetation index v2
def msavi2(B08, B04):
    return (np.divide(2 * B08 + 1 - np.sqrt(np.power((2 * B08 + 1), 2) - 8 * (B08 - B04)), 2))


# Normalised difference vegetation index
def ndvi(B08, B04):
    return np.divide((B08 - B04), (B08 + B04))


# Green normalised difference vegetation index
def gndvi(B08, B03):
    return np.divide((B08 - B03), (B08 + B03))


# Soil moisture index
def moist(B11, B08A):
    return np.divide((B08A - B11), (B08A + B11))


# Normalised difference water index
def ndwi(B11, B08):
    return np.divide((B08 - B11), (B08 + B11))


# Normalised difference water index v2
def ndwi2(B08, B03):
    return np.divide((B03 - B08), (B03 + B08))


# Normalised difference burn ratio
def nbr(B12, B08):
    return np.divide((B08 - B12), (B08 + B12))


# Non-elimination vegetation index
def nvei(B08, B04, B02):
    return np.divide(B02 - B04, B08 + B04)


# Built-up area index
def nbai(B12, B08, B02):
    return np.divide(B12 - np.divide(B08, B02), B12 + np.divide(B08, B02))


# Band ratio for built-up areas
def brba(B08, B03):
    return np.divide(B03, B08)


# Normalised difference built-up index
def ndbi(B11, B08):
    return np.divide((B11 - B08), (B11 + B08))


# Built-up features extraction
def blfei(B12, B11, B04, B03):
    base = np.divide(B03 + B04 + B12, 3)
    return np.divide(base - B11, base + B11)


# Index-based built-up index
def ibi(B12, B11, B08, B04, B03):
    savi = np.divide(B08 - B04, B08 + B04 + 0.428) * 1.856
    ndwi2 = np.divide((B03 - B08), (B03 + B08))
    ndbi = np.divide((B11 - B08), (B11 + B08))
    base = np.divide(savi + ndwi2, 2)

    return np.divide(ndbi - base, ndbi + base)


np.errstate(divide='ignore', invalid='ignore')  # enable warnings for division by zero