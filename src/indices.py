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
