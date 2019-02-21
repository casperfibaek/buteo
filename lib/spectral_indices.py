import rasterio
import formulas
from rasterio import Affine
import rasterUtils
import time

allIndicies = [
    "chlre",    # Red Edge Chlorophyll Index
    "rendvi",   # Red Edge NDVI
    "s2rep",    # Sentinel 2 Red Edge Position
    "mcari",    # Modified Chlorophyll Absorption in Reflectance Index
    "ireci",    # Inverted Red Edge Chlorophyll Index

    "arvi",     # Atmospherically Resistant Vegetation Index
    "evi",      # Enhanced vegetation index
    "evi2",     # Enhanced vegetation index v2
    "savi",     # Soil adjusted vegetation index
    "msavi2",   # Modified soil adjusted vegetation index v2
    "ndvi",     # Normalised difference vegetation index
    "gndvi",    # Green normalised difference vegetation index

    "moist",    # Soil moisture index
    "ndwi",     # Normalised difference water index
    "ndwi2",    # Normalised difference water index v2

    "nbr",      # Normalised difference burn ratio

    "nvei",     # Non-elimination vegetation index
    "nbai",     # Built-up area index
    'brba',     # Band ratio for built-up areas
    'ndbi',     # Normalised difference built-up index
    'blfei',    # Built-up features extraction 
    'ibi',      # Index-based built-up index
]


def indexBands(index):
    if index == 'chlre': return [('B08', 10), ('B05', 10)]
    if index == 'rendvi': return [('B08', 10), ('B06', 10)]
    if index == 's2rep': return [('B07', 10), ('B06', 10), ('B05', 10), ('B03', 10)]
    if index == 'mcari': return [('B05', 10), ('B04', 10), ('B03', 10)]
    if index == 'ireci': return [('B07', 10), ('B06', 10), ('B05', 10), ('B04', 10)]

    if index == 'arvi': return [('B08', 10), ('B04', 10), ('B02', 10)]
    if index == 'evi': return [('B08', 10), ('B04', 10), ('B02', 10)]
    if index == 'evi2': return [('B08', 10), ('B04', 10)]
    if index == 'savi': return [('B08', 10), ('B04', 10)]
    if index == 'msavi2': return [('B08', 10), ('B04', 10)]
    if index == 'ndvi': return [('B08', 10), ('B04', 10)]
    if index == 'gndvi': return [('B08', 10), ('B03', 10)]

    if index == 'moist': return [('B11', 10), ('B8A', 10)]
    if index == 'ndwi': return [('B11', 10), ('B08', 10)]
    if index == 'ndwi2': return [('B08', 10), ('B03', 10)]

    if index == 'nbr': return [('B12', 10), ('B08', 10)]

    if index == 'nvei': return [('B08', 10), ('B04', 10), ('B02', 10)]
    if index == 'nbai': return [('B12', 10), ('B08', 10), ('B02', 10)]
    if index == 'brba': return [('B08', 10), ('B03', 10)]
    if index == 'ndbi': return [('B11', 10), ('B08', 10)]
    if index == 'blfei': return [('B12', 10), ('B11', 10), ('B04', 10), ('B03', 10)]
    if index == 'ibi': return [('B12', 10), ('B11', 10), ('B08', 10), ('B04', 10), ('B03', 10)]
    return []


def calc( yellowObj, arrOfIndices, dst='./indices/'):
    nativeResolution = { 'B02': 10, 'B03': 10, 'B04': 10,'B05': 20, 'B06': 20, 'B07': 20, 'B08': 10, 'B8A': 20, 'B11': 20, 'B12': 20 }
    holder = {}
    metaHolder = {}
    allBands = []

    indiciesToCalculate = arrOfIndices
    if 'ALL' in arrOfIndices:
        indiciesToCalculate = allIndicies
    else:
        indiciesToCalculate = arrOfIndices

    for index in indiciesToCalculate:
        indexName = index
        bands = indexBands(index)

        for band in bands:
            allBands.append(band)
    
    uniqueBands = list(set(allBands))

    for uBand in uniqueBands:
        bandName = uBand[0]
        resoInt = uBand[1]
        natResoStr = f"{nativeResolution[bandName]}m"
        holderName = f"{bandName}_{resoInt}m"

        band = rasterio.open(yellowObj[natResoStr][bandName], driver='JP2OpenJPEG')
        metaHolder[holderName] = band.meta

        if resoInt == nativeResolution[bandName]:
            holder[holderName] = band.read().astype(rasterio.uint16)
        else:
            ratio = nativeResolution[bandName] / resoInt
            holder[holderName] = rasterUtils.resample(band, ratio, 'nearest')
            metaHolder[holderName].update(width=round(metaHolder[holderName]['width']  * ratio))
            metaHolder[holderName].update(height=round(metaHolder[holderName]['height'] * ratio))
            metaHolder[holderName].update(transform=Affine(
                metaHolder[holderName]['transform'][0] / ratio,
                metaHolder[holderName]['transform'][1],
                metaHolder[holderName]['transform'][2],
                metaHolder[holderName]['transform'][3],
                metaHolder[holderName]['transform'][4] / ratio,
                metaHolder[holderName]['transform'][5])
            )

    for index in indiciesToCalculate:
        bandsInIndex = []
        indexBandsTuple = indexBands(index)
        firstBand = f"{indexBandsTuple[0][0]}_{indexBandsTuple[0][1]}m"
        profile = metaHolder[holderName] # Take profile of first band

        for band in indexBands(index):
            bandsInIndex.append(holder[f"{band[0]}_{band[1]}m"])

        profile.update(driver = 'GTiff')
        profile.update(dtype = rasterio.float32)

        calculatedIndex = getattr(formulas, index)(*bandsInIndex)

        with rasterio.open(f'{dst}index_{index}.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads='ALL_CPUS') as output:
            output.write(calculatedIndex.astype(rasterio.float32))

import numpy as np
from numba import jit
import math

np.seterr(divide='ignore', invalid='ignore')  # disable warnings for division by zero


# Red Edge Chlorophyll Index
@jit(nopython=True, parallel=True, fastmath=True)
def chlre(B08, B05):
    return np.divide(B05, B08)


# Red Edge NDVI
@jit(nopython=True, parallel=True, fastmath=True)
def rendvi(B08, B06):
    return np.divide((B08 - B06), (B08 + B06))


# Sentinel 2 Red Edge Position
@jit(nopython=True, parallel=True, fastmath=True)
def s2rep(B07, B06, B05, B04):
    return 705 + 35 * np.divide((np.divide((B07 + B04), 2) - B05), (B06 - B05))


# Modified Chlorophyll Absorption in Reflectance Index
@jit(nopython=True, parallel=True, fastmath=True)
def mcari(B05, B04, B03):
    return ((B05 - B04) - 0.2 * (B05 - B03)) * (np.divide(B05, B04))


# Inverted Red Edge Chlorophyll Index
@jit(nopython=True, parallel=True, fastmath=True)
def ireci(B07, B06, B05, B04):
    return np.divide((B07 - B04) * B06, B05)


# Atmospherically Resistant Vegetation Index
@jit(nopython=True, parallel=True, fastmath=True)
def arvi(B08, B04, B02):
    base = (2 * B04 - B02)
    return np.divide((B08 - base), (B08 + base))


# Enhanced vegetation index
@jit(nopython=True, parallel=True, fastmath=True)
def evi(B08, B04, B02):
    return 2.5 * np.divide((B08 - B04), ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0))


# Enhanced vegetation index v2
@jit(nopython=True, parallel=True, fastmath=True)
def evi2(B08, B04):
    return 2.4 * np.divide((B08 - B04), (B08 + B04 + 1.0))


# Soil adjusted vegetation index
@jit(nopython=True, parallel=True, fastmath=True)
def savi(B08, B04):
    return np.divide(B08 - B04, B08 + B04 + 0.428) * 1.856


# Modified soil adjusted vegetation index v2
@jit(nopython=True, parallel=True, fastmath=True)
def msavi2(B08, B04):
    return (np.divide(2 * B08 + 1 - np.sqrt(np.power((2 * B08 + 1), 2) - 8 * (B08 - B04)), 2))


# Normalised difference vegetation index
@jit(nopython=True, parallel=True, fastmath=True)
def ndvi(B08, B04):
    return np.divide((B08 - B04), (B08 + B04))


# Green normalised difference vegetation index
@jit(nopython=True, parallel=True, fastmath=True)
def gndvi(B08, B03):
    return np.divide((B08 - B03), (B08 + B03))


# Soil moisture index
@jit(nopython=True, parallel=True, fastmath=True)
def moist(B11, B08A):
    return np.divide((B08A - B11), (B08A + B11))


# Normalised difference water index
@jit(nopython=True, parallel=True, fastmath=True)
def ndwi(B11, B08):
    return np.divide((B08 - B11), (B08 + B11))


# Normalised difference water index v2
@jit(nopython=True, parallel=True, fastmath=True)
def ndwi2(B08, B03):
    return np.divide((B03 - B08), (B03 + B08))


# Normalised difference burn ratio
@jit(nopython=True, parallel=True, fastmath=True)
def nbr(B12, B08):
    return np.divide((B08 - B12), (B08 + B12))


# Non-elimination vegetation index
@jit(nopython=True, parallel=True, fastmath=True)
def nvei(B08, B04, B02):
    return np.divide(B02 - B04, B08 + B04)


# Built-up area index
@jit(nopython=True, parallel=True, fastmath=True)
def nbai(B12, B08, B02):
    return np.divide(B12 - np.divide(B08, B02), B12 + np.divide(B08, B02))


# Band ratio for built-up areas
@jit(nopython=True, parallel=True, fastmath=True)
def brba(B08, B03):
    return np.divide(B03, B08)


# Normalised difference built-up index
@jit(nopython=True, parallel=True, fastmath=True)
def ndbi(B11, B08):
    return np.divide((B11 - B08), (B11 + B08))


# Built-up features extraction
@jit(nopython=True, parallel=True, fastmath=True)
def blfei(B12, B11, B04, B03):
    base = np.divide(B03 + B04 + B12, 3)
    return np.divide(base - B11, base + B11)


# Index-based built-up index
@jit(nopython=True, parallel=True, fastmath=True)
def ibi(B12, B11, B08, B04, B03):
    savi = np.divide(B08 - B04, B08 + B04 + 0.428) * 1.856
    ndwi2 = np.divide((B03 - B08), (B03 + B08))
    ndbi = np.divide((B11 - B08), (B11 + B08))
    base = np.divide(savi + ndwi2, 2)

    return np.divide(ndbi - base, ndbi + base)


np.errstate(divide='ignore', invalid='ignore')  # enable warnings for division by zero
