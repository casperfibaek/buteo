import sys; sys.path.append('../../')

from buteo.raster.warp import warp_raster
from buteo.raster.clip import clip_raster

folder = "C:/Users/caspe/Desktop/test/"
raster = folder + "dtm.tif"
patches = folder + "dtm_patches.npy"
geom = folder + "patches_64_patches.gpkg"
vector = folder + "aeroe.gpkg"
vector_odense = folder + "odense.gpkg"


# all_touch does not work..
# warp_raster(
#     raster,
#     out_path=folder + "dtm_warp.tif",
#     projection=vector_odense,
#     target_size=(10, 10),
#     clip_geom=vector,
#     crop_to_geom=True,
# )

clip_raster(
    raster,
    vector,
    out_path=folder + "dtm_clipped3.tif",
    all_touch=True,
)