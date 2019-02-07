from rasterstats import zonal_stats
import numpy as np
import os
import rasterio
import sentinelReader
import indices
import time

start = time.time()

threads = os.cpu_count() * 2
img = sentinelReader.readS2("S2A_MSIL2A_20180727T104021_N0208_R008_T32VNH_20180727T134459.SAFE")
indicesToCalculate = [
    "ari",      # Anthocyanin Reflectance Index
    "cre",      # Chlorophyll Red-Edge
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

indices.calc(img, indicesToCalculate, "./indices/")

ndvi = rasterio.open(f"./indices/ndvi.tif", driver="GTiff")
zscore = np.empty(shape=(1, ndvi.shape[0], ndvi.shape[1]), dtype = rasterio.float32)

for index in indicesToCalculate:
    a = rasterio.open(f"./indices/{index}.tif", driver="GTiff")
    b = a.read()
    s = zonal_stats("./geometry/fieldTest_01.shp", f"./indices/{index}.tif", stats="mean std")

    d = (b - s[0]["mean"]) / s[0]["std"]

    zscore = zscore + d

zscore = np.absolute(np.divide(zscore, len(indicesToCalculate)))
profile = ndvi.meta
profile.update(driver = 'GTiff')
profile.update(dtype = rasterio.float32)

with rasterio.open("./indices/zscore.tif", "w", **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(zscore)

print(f"Execution took: {round(time.time() - start, 2)}s")