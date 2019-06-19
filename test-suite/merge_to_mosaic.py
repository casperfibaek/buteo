import sys
from glob import glob
sys.path.append('../lib')

from orfeo_toolbox import merge_rasters

in_folder = 'E:\\sentinel_2_data\\ghana\\wet_season_2019\\'

B02 = glob(f"{in_folder}*B02*.tif")
B03 = glob(f"{in_folder}*B03*.tif")
B04 = glob(f"{in_folder}*B04*.tif")
B08 = glob(f"{in_folder}*B08*.tif")

merge_rasters(B02, in_folder + 'MOSAIC_B02.tif')
merge_rasters(B03, in_folder + 'MOSAIC_B03.tif')
merge_rasters(B04, in_folder + 'MOSAIC_B04.tif')
merge_rasters(B08, in_folder + 'MOSAIC_B08.tif')

in_folder = 'E:\\sentinel_2_data\\ghana\\dry_season_2019\\'

B02 = glob(f"{in_folder}*B02*.tif")
B03 = glob(f"{in_folder}*B03*.tif")
B04 = glob(f"{in_folder}*B04*.tif")
B08 = glob(f"{in_folder}*B08*.tif")

merge_rasters(B02, in_folder + 'MOSAIC_B02.tif')
merge_rasters(B03, in_folder + 'MOSAIC_B03.tif')
merge_rasters(B04, in_folder + 'MOSAIC_B04.tif')
merge_rasters(B08, in_folder + 'MOSAIC_B08.tif')
