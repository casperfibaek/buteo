# Standard library
import sys; sys.path.append("../")

# Standard library
import os

# External
import numpy as np
from glob import glob
import seaborn as sns

from buteo.raster import raster_to_array, array_to_raster, raster_to_metadata, stack_rasters_vrt, stack_rasters
from buteo.utils import split_into_offsets, hsl_to_rgb, rgb_to_hsl

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/Unicef/"

raster = os.path.join(FOLDER, "VNL_v21_npp_2014-2021_global_vcmslcfg_trend_202205302300_median_54009_500m.tif")
raster_mask = os.path.join(FOLDER, "mask_land_54009_500m.tif")
metadata = raster_to_metadata(raster)
offsets = split_into_offsets(metadata["shape"], 3, 3)


generated_images = []

for idx, offset in enumerate(offsets):
    print(f"Processing offset {idx+1} of {len(offsets)}")
    cmap = sns.color_palette("vlag", as_cmap=True)

    trend_robust = raster_to_array(raster, pixel_offsets=offset, bands=3)
    trend_robust.filled(0.0)
    trend_robust = (np.clip(trend_robust, -10.0, 10.0) + 10.0) / 20.0
    trend_robust = np.clip(trend_robust, 0.0, 1.0)

    latest = raster_to_array(raster, pixel_offsets=offset, bands=1)
    latest = np.ma.getdata(latest.filled(0.0))

    latest_10 = np.clip(latest, 0.0, 10.0) / 10.0
    latest_10 = np.clip(latest_10, 0.0, 1.0)
    latest_20 = np.clip(latest, 0.0, 20.0) / 20.0
    latest_20 = np.clip(latest_20, 0.0, 1.0)

    rgb_array = cmap(np.squeeze(trend_robust))[:, :, :3]
    rgb_array = rgb_array.astype(np.float32)

    hsl = rgb_to_hsl(rgb_array)

    hsl[:, :, 1] = hsl[:, :, 1] * latest_10[:, :, 0]
    hsl[:, :, 2] = hsl[:, :, 2] * latest_20[:, :, 0]

    output_arr = hsl_to_rgb(hsl)
    output_arr = np.rint(np.clip(output_arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    out_path = os.path.join(FOLDER, f"VNL_v21_npp_2014-2021_global_vcmslcfg_trend_202205302300_median_54009_500m_rgb_{idx}.tif")
    generated_images.append(out_path)

    array_to_raster(
        output_arr,
        reference=raster,
        out_path=out_path,
        pixel_offsets=offset,
    )

print("Stacking...")
vrt_path = os.path.join(FOLDER, "VNL_v21_npp_2014-2021_global_vcmslcfg_trend_202205302300_median_54009_500m_rgb.vrt")
stack_rasters_vrt(
    glob(FOLDER + "VNL_v21_npp_2014-2021_global_vcmslcfg_trend_202205302300_median_54009_500m_rgb_*.tif"),
    vrt_path,
    separate=False,
)

mask = array_to_raster(np.rint(raster_to_array(raster_mask) * 255).astype(np.uint8), reference=raster_mask)
stack_rasters(
    [vrt_path, mask],
    out_path=os.path.join(FOLDER, "VNL_v21_npp_2014-2021_global_vcmslcfg_trend_202205302300_median_54009_500m_rgb.tif"),
    dtype="uint8",
)
