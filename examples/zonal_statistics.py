import sys; sys.path.append('..')
from time import time
from glob import glob

from lib.stats_zonal import calc_zonal


# zones = 'E:\\SATF\\phase_IV_urban-classification\\urban_segmentation.shp'
zones = 'C:\\Users\\CFI\\Desktop\\satf_training\\data\\training_data.shp'

rasters = [
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\dry_b02.tif', 'name': 'd2'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\dry_b03.tif', 'name': 'd3'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\dry_b04.tif', 'name': 'd4'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\dry_b04_tex.tif', 'name': 'd4t'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\dry_b08.tif', 'name': 'd8'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\dry_b08_tex.tif', 'name': 'd8t'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\dry_b12.tif', 'name': 'd12'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\dry_b12_tex.tif', 'name': 'd12t'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\wet_b02.tif', 'name': 'w2'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\wet_b03.tif', 'name': 'w3'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\wet_b04.tif', 'name': 'w4'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\wet_b04_tex.tif', 'name': 'w4t'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\wet_b08.tif', 'name': 'w8'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\wet_b08_tex.tif', 'name': 'w8t'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\wet_b12.tif', 'name': 'w12'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\wet_b12_tex.tif', 'name': 'w12t'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\surf_dry.tif', 'name': 'sd'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\surf_wet.tif', 'name': 'sw'},
    {'path': 'E:\\SATF\\phase_IV_urban-classification\\data\\viirs.tif', 'name': 'vir'},
]

before = time()

for r in rasters:
    calc_zonal(zones, r["path"], prefix=f'{r["name"]}_', stats=['mean', 'med', 'mad', 'std', 'kurt', 'skew', 'iqr'], shape_attributes=False)

after = time()
dif = after - before
hours = int(dif / 3600)
minutes = int((dif % 3600) / 60)
seconds = "{0:.2f}".format(dif % 60)
print(f"Zonal_stats took: {hours}h {minutes}m {seconds}s")
