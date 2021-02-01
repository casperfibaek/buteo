import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_clip import clip_raster
from lib.raster_reproject import reproject
from glob import glob
import os

from sen1mosaic.preprocess import processFiles

base_folder = "/home/cfi/data/sentinel1_paper2/"
geom_path = "../geometry/studyArea100mBuffer.gpkg"
ascending_grd = glob(base_folder + "ascending/S1*_*_GRDH*.zip")
descending_grd = glob(base_folder + "descending/S1*_*_GRDH*.zip")

ascending_dst_folder = base_folder + "ascending_processed_grd/"
descending_dst_folder = base_folder + "descending_processed_grd/"

processed_folder = base_folder + "processed/"

proj = 'PROJCS["ETRS89 / UTM zone 32N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","25832"]]'

import pdb; pdb.set_trace()

# processFiles(ascending_grd, ascending_dst_folder)
# processFiles(descending_grd, descending_dst_folder)

# ascending_images = glob(ascending_dst_folder + "*/*.img")
# for img in ascending_images:
#     name = base_folder + "processed/asc_"+ img.split("/")[-2].split(".")[0] + ".tif"
#     try:
#         clip_raster(
#             reproject(img, target_projection=proj),
#             out_raster=name,
#             cutline=geom_path,
#             cutline_all_touch=True,
#             src_nodata=0,
#             dst_nodata=0,
#         )
#     except:
#         print(f"{img} did not overlap..")
#         pass

# descending_images = glob(descending_dst_folder + "*/*.img")
# for img in descending_images:
#     name = base_folder + "processed/desc_"+ img.split("/")[-2].split(".")[0] + ".tif"
#     try:
#         clip_raster(
#             reproject(img, target_projection=proj),
#             out_raster=name,
#             cutline=geom_path,
#             cutline_all_touch=True,
#             src_nodata=0,
#             dst_nodata=0,
#         )
#     except:
#         print(f"{img} did not overlap..")
#         pass

bsa_images = glob(processed_folder + "asc_*.tif")
bsd_images = glob(processed_folder + "desc_*.tif")

cmd_a = "pkcomposite" +  " -i " + " -i ".join(bsa_images) + f" -cr median -msknodata 0 -srcnodata 0 -dstnodata 0 -o {base_folder}mosaic_asc.tif"
cmd_d = "pkcomposite" +  " -i " + " -i ".join(bsd_images) + f" -cr median -msknodata 0 -srcnodata 0 -dstnodata 0 -o {base_folder}mosaic_desc.tif"

import pdb; pdb.set_trace()