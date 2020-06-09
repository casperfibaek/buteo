import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from glob import glob

from sen1mosaic.preprocess import processFiles
# from sen1mosaic.mosaic import buildComposite

base_folder = "/mnt/c/users/caspe/Desktop/Paper_2_StruturalDensity/Data/sentinel1/"
ascending_grd = glob(base_folder + "ascending/S1*_*_GRDH*.zip")
descending_grd = glob(base_folder + "descending/S1*_*_GRDH*.zip")

ascdending_dst_folder = base_folder + "ascending_processed_grd/"
descdending_dst_folder = base_folder + "descending_processed_grd/"

ascending_images = glob(ascdending_dst_folder + "*/*.img")
for img in ascending_images:
    name = img.split("/")[-2].split(".")[0] + "_asc.tif"
    array_to_raster(raster_to_array(img), ascdending_dst_folder + name, img)

descdending_images = glob(descdending_dst_folder + "*/*.img")
for img in descdending_images:
    name = img.split("/")[-2].split(".")[0] + "_desc.tif"
    array_to_raster(raster_to_array(img), descdending_dst_folder + name, img)

# processFiles(ascending_grd, ascdending_dst_folder)
# processFiles(descending_grd, descdending_dst_folder)

