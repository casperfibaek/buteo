import sys; sys.path.append('..'); sys.path.append('../lib/')
from glob import glob

from sen1mosaic.preprocess import processFiles
# from sen1mosaic.mosaic import buildComposite

base_folder = "/mnt/c/users/caspe/Desktop/Paper_2_StruturalDensity/Data/sentinel1/"
ascending_grd = glob(base_folder + "ascending/*.zip")
descending_grd = glob(base_folder + "descending/*.zip")

ascdending_dst_folder = base_folder + "ascending_processed_grd/"
descdending_dst_folder = base_folder + "descending_processed_grd/"

processFiles(ascending_grd, ascdending_dst_folder)
processFiles(descending_grd, descdending_dst_folder)