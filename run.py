from rasterstats import zonal_stats
import numpy as np
import os
import rasterio
from rasterio import Affine
import sentinelReader
import indices
import time

start = time.time()

threads = os.cpu_count() * 2
img = sentinelReader.readS2("S2A_MSIL2A_20180727T104021_N0208_R008_T32VNH_20180727T134459.SAFE")
indicesToCalculate = [
    "chlre",    # Red Edge Chlorophyll Index
    # "rendvi",   # Red Edge NDVI
    "s2rep",    # Sentinel 2 Red Edge Position
    # "mcari",    # Modified Chlorophyll Absorption in Reflectance Index
    "ireci",    # Inverted Red Edge Chlorophyll Index

    "arvi",     # Atmospherically Resistant Vegetation Index
    "evi",      # Enhanced vegetation index
    "evi2",     # Enhanced vegetation index v2
    "savi",     # Soil adjusted vegetation index
    "msavi2",   # Modified soil adjusted vegetation index v2
    "ndvi",     # Normalised difference vegetation index
    "gndvi",    # Green normalised difference vegetation index
    
    "moist",    # Soil moisture index
    # "ndwi",     # Normalised difference water index
    # "ndwi2",    # Normalised difference water index v2

    "nbr",      # Normalised difference burn ratio

    # "nvei",     # Non-elimination vegetation index
    "nbai",     # Built-up area index
    'brba',     # Band ratio for built-up areas
    # 'ndbi',     # Normalised difference built-up index
    'blfei',    # Built-up features extraction 
    # 'ibi',      # Index-based built-up index
]

# indicesToCalculate = ['ndvi', 'ndwi2']

indices.calc(img, indicesToCalculate, "./indices/")

b2 = rasterio.open(img['10m']['B02'], driver="JP2OpenJPEG")
avgZscore = np.empty(shape=(1, b2.shape[0], b2.shape[1]), dtype = rasterio.float32)

profile = b2.meta
profile.update(driver = 'GTiff')
profile.update(dtype = rasterio.float32)

for index in indicesToCalculate:
    a = rasterio.open(f"./indices/index_{index}.tif", driver="GTiff")
    b = a.read()
    stats = zonal_stats("./geometry/fieldTest_01.shp", f"./indices/index_{index}.tif", stats="median mean std min max range")
    median = stats[0]["median"]
    mean = stats[0]["mean"]
    std = stats[0]["std"]
    min = stats[0]["min"]
    max = stats[0]["max"]
    range = stats[0]["range"]

    deviations = np.abs(median - b)

    with rasterio.open(f"./indices/deviation_{index}.tif", "w", **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
        dst.write(deviations)

    mad_std = zonal_stats("./geometry/fieldTest_01.shp", f"./indices/deviation_{index}.tif", stats="median")[0]["median"] * 1.4826

    nonParametricSkew = (mean - median) / std
    nonParametricSkewMad = (mean - median) / mad_std
    simpleSkewRatio = abs(1 - (mean / median))
    minZscore = (min - median) / mad_std
    maxZscore = (max - median) / mad_std
    
    print('    ')
    print(f"{index}:")
    print(f"    non-parametric skew       = {round(nonParametricSkew, 3)}")
    print(f"    non-parametric skew (mad) = {round(nonParametricSkewMad, 3)}")
    print(f"    standard deviation        = {round(std, 3)}")
    print(f"    standard deviation (mad)  = {round(mad_std, 3)}")
    print(f"    minimum training z-score  = {round(minZscore, 3)}")
    print(f"    maximum training z-score  = {round(maxZscore, 3)}")
    print(f"    simple skew ratio         = {round(simpleSkewRatio, 3)}")
    print(f"    range                     = {round(range, 3)}")
    print(f"    range std ratio           = {round(range / std, 3)}")
    print(f"    range std ratio (mad)     = {round(range / mad_std, 3)}")

    os.remove(f"./indices/deviation_{index}.tif")

    zscore = (b - median) / mad_std

    # with rasterio.open(f"./indices/zscore_{index}.tif", "w", **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    #     dst.write(zscore)

    avgZscore += np.abs(zscore)

avgZscore = np.divide(avgZscore, len(indicesToCalculate))

with rasterio.open("./indices/zscore.tif", "w", **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(avgZscore)

print(f"Execution took: {round(time.time() - start, 2)}s")