import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import robust_scaler_filter
from lib.stats_zonal import calc_zonal
from glob import glob
import os

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\"
images = glob(folder + "*.tif")

names = []

for img in images:
    names.append(os.path.basename(img).split(".")[0] + "_")

vect = folder + "grid_80m.gpkg"
calc_zonal(vect, images, names, ["mean", "med", "std", "min", "max"])

# for img in images:
#     name = os.path.basename(img)
#     array_to_raster(robust_scaler_filter(raster_to_array(img)), folder + "scaled\\"+ name, img)

