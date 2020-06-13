import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_clip import clip_raster
from lib.raster_reproject import reproject
from glob import glob
import os

from sen1mosaic.preprocess import processFiles
# from sen1mosaic.mosaic import buildComposite

base_folder = "/mnt/c/users/caspe/Desktop/Paper_2_StruturalDensity/Data/sentinel1/"
ascending_grd = glob(base_folder + "ascending/S1*_*_GRDH*.zip")
descending_grd = glob(base_folder + "descending/S1*_*_GRDH*.zip")

ascending_dst_folder = base_folder + "ascending_processed_grd/"
descending_dst_folder = base_folder + "descending_processed_grd/"

proj = 'PROJCS["ETRS89 / UTM zone 32N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","25832"]]'

# ascending_images = glob(ascending_dst_folder + "*/*.img")
# for img in ascending_images:
#     name = base_folder + "processed/asc_"+ img.split("/")[-2].split(".")[0] + ".tif"
#     reproject(array_to_raster(raster_to_array(img), None, img), name, target_projection=proj)

# descdending_images = glob(descending_dst_folder + "*/*.img")
# for img in descdending_images:
#     name = base_folder + "processed/desc_"+ img.split("/")[-2].split(".")[0] + ".tif"
#     reproject(array_to_raster(raster_to_array(img), None, img), name, target_projection=proj)


# processFiles(ascending_grd, ascdending_dst_folder)
# processFiles(descending_grd, descdending_dst_folder)
# folder = "/mnt/c/users/caspe/Desktop/Paper_2_StruturalDensity/Data/sentinel1/processed/"
# folder_geom = "/mnt/c/users/caspe/Desktop/Paper_2_StruturalDensity/Data/"

# images = glob(folder + "*.tif")
# for img in images:
#     output = folder + "clipped/" + os.path.basename(img)
#     # import pdb; pdb.set_trace()
#     try:
#         clipped = clip_raster(img, cutline=folder_geom + "studyArea100mBuffer.gpkg", cutline_all_touch=True)
#         array_to_raster(raster_to_array(clipped), output, clipped, src_nodata=0, dst_nodata=0)
#     except:
#         print(f"{img} did not overlap..")
#         pass


folder = "/mnt/c/users/caspe/Desktop/Paper_2_StruturalDensity/Data/sentinel1/processed/clipped/"
out_folder = "/mnt/c/users/caspe/Desktop/Paper_2_StruturalDensity/Data/sentinel1/processed/"

bsa_images = glob(folder + "asc_*.tif")
bsd_images = glob(folder + "desc_*.tif")

cmd_a = "pkcomposite" +  " -i " + " -i ".join(bsa_images) + f" -cr median -msknodata 0 -srcnodata 0 -dstnodata 0 -o {out_folder}mosaic_asc.tif"
cmd_d = "pkcomposite" +  " -i " + " -i ".join(bsd_images) + f" -cr median -msknodata 0 -srcnodata 0 -dstnodata 0 -o {out_folder}mosaic_desc.tif"

import pdb; pdb.set_trace()