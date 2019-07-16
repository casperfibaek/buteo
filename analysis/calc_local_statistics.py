import sys
import numpy as np
from time import time
sys.path.append('../lib')
# sys.path.append('../gefolki')


# from algorithm import GEFolki
# from tools import wrapData

# from zscores import calc_zscores
from local_statistics import local_statistics2
# from crosslayer_math import layers_math
from array_to_raster import array_to_raster
from raster_to_array import raster_to_array

# before1 = time()
# i1, i2 = GEFolki(raster_to_array('E:\\SATF\\data\\folki_test\\dry_b04_test.tif'), raster_to_array('E:\\SATF\\data\\folki_test\\coh_gauss_test.tif'))
# N = np.sqrt(i1**2 + i2**2)
# after1 = time() - before1
# print(after1)

# array_to_raster(i1, reference_raster='E:\\SATF\\data\\folki_test\\dry_b04_test.tif', out_raster='E:\\SATF\\data\\folki_test\\dry_b04_test_gefolki.tif')
# array_to_raster(i2, reference_raster='E:\\SATF\\data\\folki_test\\coh_gauss_test.tif', out_raster='E:\\SATF\\data\\folki_test\\coh_gauss_test_gefolki.tif')


# import pdb; pdb.set_trace()

# before1 = time()
# local_statistics('E:\\SATF\\data\\db.tif', radius=1, out_raster='E:\\SATF\\data\\db_mean1.tif')
# after1 = time() - before1

before2 = time()
med = local_statistics2('E:\\SATF\\data\\coh_dry.tif', out_raster='E:\\SATF\\data\\coh_dry_med.tif', radius=1, stat='median')
# m1 = local_statistics2('E:\\SATF\\data\\coh_dry_med.tif', radius=1, stat='max')
# m2 = local_statistics2('E:\\SATF\\data\\coh_dry_med.tif', radius=2, stat='max')
# m3 = local_statistics2('E:\\SATF\\data\\coh_dry_med.tif', radius=3, stat='max')

# images_array = np.array([m1, m2, m3])
# array_to_raster(np.mean(images_array, axis=0), reference_raster='E:\\SATF\\data\\coh_dry.tif', out_raster='E:\\SATF\\data\\coh_dry_max3.tif')

after2 = time() - before2
print(after2)
