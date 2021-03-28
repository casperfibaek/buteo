import sys; sys.path.append('../../')
from glob import glob
import numpy as np


# from buteo.raster.warp import warp_raster
from buteo.raster.align import align_rasters
from buteo.raster.io import raster_to_array, array_to_raster
# from buteo.raster.clip import clip_raster
# from buteo.vector.io import vector_to_reference

folder = "C:/Users/caspe/Desktop/test/align/comp/"
rasters = glob(folder + "*.tif")

sars = []
for raster in rasters:
    sars.append(raster_to_array(raster))

stacked = np.ma.masked_equal(np.dstack(sars), 0) 
array_to_raster(np.ma.median(stacked, axis=2), rasters[0], folder + "mosaic.tif")

# import pdb; pdb.set_trace()

# warp_raster(
#     raster,
#     out_path=folder + "dtm_warp.tif",
#     projection=vector_odense,
#     target_size=(10, 10),
#     clip_geom=vector,
#     crop_to_geom=True,
# )

# vector_df = vector_to_reference(vector)

# import pdb; pdb.set_trace()

# align_rasters(
#     rasters,
#     output=folder + "comp/",
#     src_nodata=0.0,
#     bounding_box="union",
# )