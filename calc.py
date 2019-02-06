import rasterio
import numpy as np
import yellow
import time
import utils

start = time.time()
threads = utils.threads()
img = yellow.readS2('D:\pythonScripts\yellow\S2A_MSIL2A_20180719T112121_N0208_R037_T30UXC_20180719T141416.SAFE')

with rasterio.open(img['10m']['B02'], driver='JP2OpenJPEG') as b2:
    B2 = b2.read().astype(rasterio.float32)
with rasterio.open(img['10m']['B03'], driver='JP2OpenJPEG') as b3:
    B3 = b3.read().astype(rasterio.float32)
with rasterio.open(img['10m']['B04'], driver='JP2OpenJPEG') as b4:
    B4 = b4.read().astype(rasterio.float32)
with rasterio.open(img['10m']['B08'], driver='JP2OpenJPEG') as b8:
    B8 = b8.read().astype(rasterio.float32)

with rasterio.open(img['20m']['B05'], driver='JP2OpenJPEG') as b5:
    B5 = b5.read().astype(rasterio.float32)
    B5_10m = utils.resample(b5, 2)
with rasterio.open(img['20m']['B06'], driver='JP2OpenJPEG') as b6:
    B6 = b6.read().astype(rasterio.float32)
    B6_10m = utils.resample(b6, 2)
with rasterio.open(img['20m']['B07'], driver='JP2OpenJPEG') as b7:
    B7 = b7.read().astype(rasterio.float32)
    B7_10m = utils.resample(b7, 2).astype(rasterio.float32)
with rasterio.open(img['20m']['B8A'], driver='JP2OpenJPEG') as b8a:
    B8A = b8a.read().astype(rasterio.float32)
    B8A_10m = utils.resample(b8a, 2).astype(rasterio.float32)    
with rasterio.open(img['20m']['B11'], driver='JP2OpenJPEG') as b11:
    B11 = b11.read().astype(rasterio.float32)
    B11_10m = utils.resample(b11, 2, 'cubic').astype(rasterio.float32)
with rasterio.open(img['20m']['B12'], driver='JP2OpenJPEG') as b12:
    B12 = b12.read().astype(rasterio.float32)
    B12_10m = utils.resample(b12, 2, 'cubic').astype(rasterio.float32)    

# np.seterr(divide='ignore', invalid='ignore') # disable warnings for division by zero

msavi2 = ((2 * B8 + 1 - np.sqrt( np.power((2 * B8 + 1), 2) - 8 * (B8 - B4))) / 2)
mcari = ((B5_10m - B4) - 0.2 * (B5_10m - B3)) * (np.divide(B5_10m, B4))
s2rep = 705 + 35 * (np.divide((np.divide((B7_10m + B4), 2) - B5_10m), (B6_10m - B5_10m)))
ndwi = np.divide((B8 - B11_10m), (B8 + B11_10m))
ndvi = np.divide((B8 - B4), (B8 + B4))
chlRededge = np.power(np.divide(B7_10m, B5_10m), (-1.0))
moist = np.divide((B8A_10m - B11_10m), (B8A_10m + B11_10m))
evi = 2.5 * np.divide((B8 - B4), ((B8 + 6.0 * B4 - 7.5 * B2) + 1.0))
evi2 = 2.4 * np.divide((B8 - B4), (B8 + B4 + 1.0))
nbr = ((B8 - B12_10m), (B8 + B12_10m))
ari = np.divide(1.0,  B3) - np.divide(1.0, B5_10m)

# np.errstate(divide='ignore', invalid='ignore') # enable warnings for division by zero

profile = b4.meta
profile.update(driver = 'GTiff')
profile.update(dtype = rasterio.float32)

with rasterio.open('index_msavi2.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(msavi2.astype(rasterio.float32))

with rasterio.open('index_mcari.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(mcari.astype(rasterio.float32))

with rasterio.open('index_s2rep.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(s2rep.astype(rasterio.float32))

with rasterio.open('index_ndwi.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(ndwi.astype(rasterio.float32))

with rasterio.open('index_ndvi.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(ndvi.astype(rasterio.float32))

with rasterio.open('index_chlRededge.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(chlRededge.astype(rasterio.float32))

with rasterio.open('index_moist.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(moist.astype(rasterio.float32))

with rasterio.open('index_evi.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(evi.astype(rasterio.float32))

with rasterio.open('index_evi2.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(evi2.astype(rasterio.float32))

with rasterio.open('index_nbr.tif', 'w', **profile, compress='DEFLATE', predictor=3, num_threads=threads) as dst:
    dst.write(nbr.astype(rasterio.float32))

end = time.time()
print(f'Script execution took: {round(end - start)}s')
