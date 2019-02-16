import numpy as np
from rasterUtils.arrayToRaster import arrayToRaster
from rasterUtils.resample import resample
import time


arr = np.random.randint(0, high=255, size=(2500, 5000), dtype='uint8')
path = '../raster/S2B_MSIL2A_20180702T104019_N0208_R008_T32VNJ_20180702T150728.SAFE/GRANULE/L2A_T32VNJ_A006898_20180702T104021/IMG_DATA/R20m/T32VNJ_20180702T104019_B08_20m.tif'

# bob = arrayToRaster(
#     arr,
#     outRaster='../raster/test_02.tif',
#     referenceRaster=path,
#     resample=True,
#     # topLeft=[499980, 6.40002e+06],
#     # pixelSize=[10, 10],
#     # projection="+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
# )

before = time.perf_counter()
resample(
    '../raster/test_02_resampled.tif',
    referenceRaster=path,
    outRaster='../raster/test_03_resampled.tif',
)
print(f'arrayToRaster took: {time.perf_counter() - before}s')
