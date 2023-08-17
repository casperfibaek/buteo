# Mikelsons, Karlis; Wang, Menghua; Jiang, Lide; Wang, Xiao-Long (2021), “Global land mask for satellite ocean color remote sensing”, Mendeley Data, V1, doi: 10.17632/9r93m9s7cw.1
import os
import sys; sys.path.append("../")
import buteo as beo
import numpy as np


FOLDER = r"C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/data/global_human_settlement/"

path_land_mask = os.path.join(FOLDER, "GHS_LAND_E2018_GLOBE_R2022A_54009_1000_V1_0.tif")
# path_house_volume = os.path.join(FOLDER, "GHS_BUILT_V_E2025_GLOBE_R2023A_54009_1000_V1_0.tif")

# assert beo.check_rasters_are_aligned([path_land_mask, path_house_volume]), "Rasters are not aligned"

land_mask = (beo.raster_to_array(path_land_mask, filled=True, fill_value=0) > 0).astype(np.uint8)
beo.array_to_raster(land_mask, reference=path_land_mask, out_path=os.path.join(FOLDER, "land_mask_54009.tif"))
import pdb; pdb.set_trace()