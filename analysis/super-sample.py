import sys
import os
import numpy as np
import time
from glob import glob

sys.path.append('../lib')
from sentinel_super_sample import super_sample_s2

before = time.time()

base = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\data\\s2\\mosaic\\'
out_folder = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\'

b4 = os.path.join(base, 'mosaic_R10m_B04.tif')
b5_20m = os.path.join(base, 'mosaic_R20m_B05.tif')
b6_20m = os.path.join(base, 'mosaic_R20m_B06.tif')
b7_20m = os.path.join(base, 'mosaic_R20m_B07.tif')
b8a_20m = os.path.join(base, 'mosaic_R20m_B8A.tif')
b8 = os.path.join(base, 'mosaic_R10m_B08.tif')

super_sample_s2(b4, b8, B5=b5_20m, B6=b6_20m, B7=b7_20m, B8A=b8a_20m, out_folder=out_folder, suffix='_ss')

after = time.time()

print((after - before) / 60)
