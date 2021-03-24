import sys; sys.path.append('../../')
from glob import glob
import numpy as np

folder = "C:/Users/caspe/Desktop/test/"

from buteo.raster.clip import clip_raster

raster = folder + "fyn_wgs84_utm32.tif"

target = folder + "odense.gpkg"

clip_raster(
    raster,
    target,
    out_path=folder + "fyn_clip.tif",
    crop_to_geom=False,
    overwrite=True,
)

# import pdb; pdb.set_trace()