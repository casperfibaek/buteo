import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_clip import clip_raster
from lib.raster_reproject import reproject
from glob import glob
import os

from sen1mosaic.preprocess import correctionGraph

folder = "C:/Users/caspe/Desktop/noise_test/"
output_folder = folder + "processed/"
images = glob(folder + "*.zip")

for image in images:
    base = os.path.basename(image)
    basename = os.path.splitext(base)[0]
    correctionGraph(image, f"{basename}_step1", output_folder, graph='backscatter_step1.xml')
    correctionGraph(f"{output_folder}/{basename}_step1.dim", f"{basename}_step2", output_folder, graph='backscatter_step2.xml')

