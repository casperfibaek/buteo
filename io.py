from rstats import rstats
from evaluate import evaluate


shape = './geometry/threeUrban.shp'
inRaster = './geometry/index_nbai.tif'

stats = rstats(shape, inRaster, nodata=0, histogram=True,
               statistics=('mean', 'std', 'median', 'kurtosis', 'variation',
                           'skew', 'npskew', 'skewratio', 'mad', 'madstd',
                           'min', 'max', 'count', 'range'))

print(stats)

evaluated = evaluate(shape, inRaster, outRaster='./geometry/index_nbai_evaluated.tif', inMemory=False , nodata=0)
# print(evaluated)

# obs weird ndvi values around buildings