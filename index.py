import rasterio
import utils
import indexFunc

threads = utils.threads()

def native(band):
    if band == 'B02': return 10
    if band == 'B03': return 10
    if band == 'B04': return 10
    if band == 'B05': return 20
    if band == 'B06': return 20
    if band == 'B07': return 20
    if band == 'B08': return 10
    if band == 'B8A': return 20
    if band == 'B11': return 20
    if band == 'B12': return 20
    return 10

def req(index):
    if index == 'msavi2': return [('B08', 10), ('B04', 10)]
    if index == 'mcari': return [('B05', 10), ('B04', 10), ('B03', 10)]
    if index == 's2rep': return [('B07', 20), ('B06', 20), ('B05', 20), ('B03', 20)]
    if index == 'ndwi': return [('B11', 10), ('B08', 10)]
    if index == 'ndvi': return [('B08', 10), ('B04', 10)]
    if index == 'chlRedEdge': return [('B07', 20), ('B05', 20)]
    if index == 'moist': return [('B11', 20), ('B08A', 20)]
    if index == 'evi': return [('B08', 10), ('B04', 10), ('B02', 10)]
    if index == 'evi2': return [('B08', 10), ('B04', 10)]
    if index == 'nbr': return [('B12', 10), ('B08', 10)]
    if index == 'ari': return [('B05', 10), ('B03', 10)]
    return []


def calc( yellowObj, arrOfIndices, dst='./indices/'):
    holder = {}

    allBands = []
    for index in arrOfIndices:
        allBands.extend(req(index))

    uniqueBands = list(set(allBands))
    
    for uniqueBand in uniqueBands:
        if len(uniqueBand.split('_')) == 1:
            with rasterio.open(yellowObj['10m'][uniqueBand], driver='JP2OpenJPEG') as band:
                profile = band.meta
                profile.update(driver = 'GTiff')
                profile.update(dtype = rasterio.float32) # handle this somewhere proper
                holder[uniqueBand] = band.read().astype(rasterio.uint16)
        else:
            with rasterio.open(yellowObj['20m'][uniqueBand], driver='JP2OpenJPEG') as band:
                lowres = band.read().astype(rasterio.uint16)
                holder[uniqueBand] = utils.resample(lowres, 2, 'bilinear').astype(rasterio.float32)   

    for index in arrOfIndices:
        h = []
        highestResolution = None
        for band in req(index):
            h.append(f"holder['{band}']")
        joined = ', '.join(h)
        s = f"getattr(indexFunc, index)({joined})"
        
        calculated = eval(s)

        with rasterio.open(f'{dst}{index}.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as output:
            output.write(calculated.astype(rasterio.float32))
