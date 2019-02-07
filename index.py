import rasterio
import utils
import indexFunc
from rasterio import Affine
import time

threads = utils.threads()

def indexBands(index):
    if index == 'msavi2': return [('B08', 10), ('B04', 10)]
    if index == 'mcari': return [('B05', 10), ('B04', 10), ('B03', 10)]
    if index == 's2rep': return [('B07', 20), ('B06', 20), ('B05', 20), ('B03', 20)]
    if index == 'ndwi': return [('B11', 10), ('B08', 10)]
    if index == 'ndvi': return [('B08', 10), ('B04', 10)]
    if index == 'cre': return [('B07', 20), ('B05', 20)]
    if index == 'moist': return [('B11', 20), ('B8A', 20)]
    if index == 'evi': return [('B08', 10), ('B04', 10), ('B02', 10)]
    if index == 'evi2': return [('B08', 10), ('B04', 10)]
    if index == 'nbr': return [('B12', 10), ('B08', 10)]
    if index == 'ari': return [('B05', 10), ('B03', 10)]
    return []


def calc( yellowObj, arrOfIndices, dst='./indices/'):
    nativeResolution = { 'B02': 10, 'B03': 10, 'B04': 10,'B05': 20, 'B06': 20, 'B07': 20, 'B08': 10, 'B8A': 20, 'B11': 20, 'B12': 20 }
    holder = {}
    metaHolder = {}
    allBands = []

    for index in arrOfIndices:
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
            holder[holderName] = utils.resample(band, ratio, 'nearest')
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

    for index in arrOfIndices:
        bandsInIndex = []
        indexBandsTuple = indexBands(index)
        firstBand = f"{indexBandsTuple[0][0]}_{indexBandsTuple[0][1]}m"
        profile = metaHolder[holderName] # Take profile of first band

        for band in indexBands(index):
            bandsInIndex.append(holder[f"{band[0]}_{band[1]}m"])

        profile.update(driver = 'GTiff')
        profile.update(dtype = rasterio.float32)

        calculatedIndex = getattr(indexFunc, index)(*bandsInIndex)

        with rasterio.open(f'{dst}{index}.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as output:
            output.write(calculatedIndex.astype(rasterio.float32))
