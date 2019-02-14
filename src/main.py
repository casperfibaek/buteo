from rasterUtils.rasterToArray import rasterToArray
from rasterUtils.clipRaster import clipRaster
from rasterUtils.rasterStats import rasterStats
import time


before = time.perf_counter()
stats = rasterStats(
    '../raster/S2B_MSIL2A_20180702T104019_N0208_R008_T32VNJ_20180702T150728.SAFE/GRANULE/L2A_T32VNJ_A006898_20180702T104021/IMG_DATA/R10m/T32VNJ_20180702T104019_B08_10m.jp2',
    cutline='../geometry/roses.geojson',
    histogram=True,
    statistics=('min', 'max')
    # cutlineAllTouch=False,
    # quiet=True,
)
print(f'rasterStats took: {time.perf_counter() - before}s')
print(stats)
