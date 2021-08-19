import sys

sys.path.append("../../")
from buteo.vector.rasterize import rasterize_vector
from buteo.vector.clip import clip_vector
from buteo.raster.clip import internal_clip_raster
from buteo.raster.resample import internal_resample_raster
from buteo.raster.io import raster_to_array, array_to_raster
from osgeo import ogr, gdal
from glob import glob
import os

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/vector/"
out_folder = (
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/raster/people/"
)
tile_folder = folder + "processing_tiles/"
buildings = folder + "buildings_ppl_04.gpkg"


# for region in glob(tile_folder + "fid_*.gpkg"):
#     name = os.path.basename(region)
#     number = os.path.splitext(name.split("_")[1])[0]

#     print(f"Processing tile: {name}")

#     clipped_path = f"/vsimem/fid_{number}_clipped_buildings.gpkg"
#     rasterized_m1_path = f"/vsimem/fid_{number}_rasterized_1m.tif"
#     rasterized_m10_path = f"/vsimem/fid_{number}_resampled_10m.tif"

#     print(f"Clipping..")
#     clip_vector(buildings, clip_geom=region, out_path=clipped_path)

#     print(f"Rasterizing to small GSD..")
#     rasterize_vector(
#         buildings,
#         1.0,
#         out_path=rasterized_m1_path,
#         extent=region,
#         attribute="ppl_per_sq_meter",
#     )
#     driver = ogr.GetDriverByName("GPKG")
#     driver.DeleteDataSource(clipped_path)

#     print(f"Resampling to large GSD..")
#     internal_resample_raster(
#         rasterized_m1_path,
#         10.0,
#         resample_alg="average",
#         out_path=rasterized_m10_path,
#     )
#     gdal.Unlink(rasterized_m1_path)

#     print(f"Writing output..")
#     array_to_raster(
#         (raster_to_array(rasterized_m10_path) / 10).astype("float32"),
#         reference=rasterized_m10_path,
#         out_path=out_folder + f"fid_{number}_rasterized.tif",
#     )
#     gdal.Unlink(rasterized_m10_path)


vrt_path = out_folder + "merged.vrt"
vrt = raster_to_array(vrt_path)

current = vrt.sum()
total = 5792202 - 39499 - 84  # bornholm og ertholmene

scale = total / current

vrt = (vrt * scale).astype("float32")

scale_path = array_to_raster(vrt, vrt_path)
internal_clip_raster(
    scale_path,
    folder + "denmark_proper_hull.gpkg",
    out_path=out_folder + "people_v2.tif",
    adjust_bbox=False,
    all_touch=False,
    postfix="",
)
