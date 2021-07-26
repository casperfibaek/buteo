import sys

sys.path.append("../../")
from glob import glob
from buteo.raster.clip import clip_raster

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/"

vector = folder + "vector/roskilde_predict.gpkg"
rasters = glob(folder + "raster/*.tif")

clip_raster(rasters, vector, folder + "predictions/roskilde_rasters/", postfix="")
