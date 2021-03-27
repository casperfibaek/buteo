import sys; sys.path.append('../../')

# from buteo.raster.warp import warp_raster
from buteo.raster.warp import warp_raster
from buteo.vector.io import vector_to_reference

folder = "C:/Users/caspe/Desktop/test/"
raster = folder + "dtm.tif"
vector = folder + "aeroe.gpkg"
# patches = folder + "dtm_patches.npy"
# geom = folder + "patches_64_patches.gpkg"
vector_odense = folder + "odense.gpkg"


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

warp_raster(
    raster,
    out_path=folder + "dtm_clipped2.tif",
    clip_geom=vector,
    # projection=vector_odense,
    all_touch=True,
)