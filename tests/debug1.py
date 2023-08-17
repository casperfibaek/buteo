# Mikelsons, Karlis; Wang, Menghua; Jiang, Lide; Wang, Xiao-Long (2021), “Global land mask for satellite ocean color remote sensing”, Mendeley Data, V1, doi: 10.17632/9r93m9s7cw.1
import os
import sys; sys.path.append("../")
import buteo as beo


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/data/masks/"
FOLDER_REF = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/data/nightlights/"

# This is the reference raster for projection, pixel_size, etc.. we are going to use
reference = os.path.join(FOLDER_REF, "VNL_v21_npp_2021_global_vcmslcfg_c202205302300.median.dat_500m_54009.tif")

# Align the water_mask to the reference raster
water_mask = os.path.join(FOLDER, "watermask.tif")

print("Reprojecting...")
reprojected = beo.raster_reproject(water_mask, projection=reference, resample_alg="min", out_path=os.path.join(FOLDER, "watermask_54009.tif"))
print("Resampling...")
resampled = beo.raster_resample(reprojected, target_size=reference, resample_alg="min", out_path=os.path.join(FOLDER, "watermask_500m_54009.tif"))
print()
aligned = beo.raster_align(resampled, reference=reference, resample_alg="min", out_path=os.path.join(FOLDER, "watermask_500m_54009_aligned.tif"))

# # Flip the mask so land is 1 and water is 0
# beo.array_to_raster(
#     beo.raster_to_array(aligned, filled=True, fill_value=0, cast=np.uint8) == 0,
#     reference=reference,
#     out_path=os.path.join(FOLDER, "landmask_500m_54009.tif"),
# )
