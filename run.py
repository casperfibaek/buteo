from rasterstats import zonal_stats
import numpy as np
import os
import rasterio
from rasterio import Affine
import lib.sentinelHelper as sentinelHelper
import lib.indices as indices
import time

start = time.time()

threads = os.cpu_count() * 2
img = sentinelHelper.readS2("S2A_MSIL2A_20180727T104021_N0208_R008_T32VNH_20180727T134459.SAFE")

indicesToCalculate = ['arvi', 'evi', 'evi2', 'savi', 'msavi2', 'ndvi', 'gndvi']

indices.calc(img, 'ALL', "./indices/")

b2 = rasterio.open(img['10m']['B02'])
avgZscore = np.empty(shape=(1, b2.shape[0], b2.shape[1]), dtype = rasterio.float32)

profile = b2.meta
profile.update(driver = 'GTiff')
profile.update(dtype = rasterio.float32)

for index in indicesToCalculate:
    a = rasterio.open(f"./indices/index_{index}.tif", driver="GTiff")
    b = a.read()
    stats = zonal_stats("./geometry/fieldTest_01.shp", f"./indices/index_{index}.tif", stats="median mean std min max range")

    deviations = np.abs(stats[0]["median"] - b)

    with rasterio.open(f"./indices/deviation_{index}.tif", "w", **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
        dst.write(deviations)

    mad_std = zonal_stats("./geometry/fieldTest_01.shp", f"./indices/deviation_{index}.tif", stats="median")[0]["median"] * 1.4826

    normalRange = stats[0]["range"]
    normalMean = stats[0]['mean']
    nonParametricSkewMad = (stats[0]["mean"] - stats[0]["median"]) / mad_std
    simpleSkewRatio = 1 - (stats[0]["mean"] / stats[0]["median"])
    minZscore = (stats[0]["min"] - stats[0]["median"]) / mad_std
    maxZscore = (stats[0]["max"] - stats[0]["median"]) / mad_std
    rangeStdRatio = stats[0]["range"] / mad_std
    
    print('    ')
    print(f"{index}:")
    print(f"    non-parametric skew (mad) = {round(nonParametricSkewMad, 3)}")
    print(f"    standard deviation (mad)  = {round(mad_std, 3)}")
    print(f"    coef. of variation (mad)  = {round((mad_std / normalMean) * 100, 3)}")
    print(f"    range std ratio (mad)     = {round(rangeStdRatio, 3)}")
    print(f"    range                     = {round(normalRange, 3)}")
    print(f"    minimum training z-score  = {round(minZscore, 3)}")
    print(f"    maximum training z-score  = {round(maxZscore, 3)}")
    print(f"    simple skew ratio         = {round(simpleSkewRatio, 3)}")

    os.remove(f"./indices/deviation_{index}.tif")

    zscore = (b - stats[0]["median"]) / mad_std

    # with rasterio.open(f"./indices/zscore_{index}.tif", "w", **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    #     dst.write(zscore)

    avgZscore += np.abs(zscore)

avgZscore = np.divide(avgZscore, len(indicesToCalculate))

with rasterio.open("./indices/zscore.tif", "w", **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(avgZscore)

print(f"Execution took: {round(time.time() - start, 2)}s")