import sys; sys.path.append('..'); sys.path.append('../lib/')
from osgeo import gdal
import geopandas as gpd
import pandas as pd
import os
import shutil
from glob import glob
from lib.raster_io import raster_to_metadata, clip_raster
from lib.vector_io import intersection_rasters

in_dir = '/mnt/c/Users/caspe/Desktop/tests/'
images = glob(in_dir + '*.tif')

bob = intersection_rasters(images[0], images[1])
carl = clip_raster(images[0], out_raster=in_dir + 'clip2.tif', cutline=bob)
