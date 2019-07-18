import sys
from time import time
from glob import glob

sys.path.append('../lib')
from zonal_statistics import calc_zonal


zones = 'E:\\SATF\\phase_IV_urban-classification\\urban_segmentation.shp'
rasters = glob('E:\\SATF\\phase_IV_urban-classification\\data\\*.tif')


before = time()

for i, r in enumerate(rasters):
    calc_zonal(zones, r, prefix=f'{i}_', stats=['mean', 'med', 'mad', 'std', 'kurt', 'skew', 'iqr'], shape_attributes=True)

after = time()
dif = after - before
hours = int(dif / 3600)
minutes = int((dif % 3600) / 60)
seconds = "{0:.2f}".format(dif % 60)
print(f"Zonal_stats took: {hours}h {minutes}m {seconds}s")
