import os
import sys; sys.path.append("../")
from glob import glob
from tqdm import tqdm

import buteo as beo
import numpy as np


FOLDERS = [
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/egypt/",
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/ghana/",
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/israel_gaza/",
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_dar/",
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_kigoma/",
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_kilimanjaro/",
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_mwanza_Q2/",
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_mwanza_Q3/",
    "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/uganda/",
]

for FOLDER in tqdm(FOLDERS, total=len(FOLDERS)):
    mask = os.path.join(FOLDER, "mask.gpkg")
    label = os.path.join(FOLDER, "labels_10m.tif")

    clip_geom = beo.raster_to_extent(label)
    clipped = beo.vector_clip(mask, clip_geom, to_extent=True)
    masked = beo.vector_rasterize(clipped, label, extent=mask)
    aligned = beo.raster_align(masked, reference=label)[0]

    arr_mask = beo.raster_to_array(aligned, filled=True, fill_value=0, cast="uint8")
    mask_aligned = beo.array_to_raster(arr_mask, reference=label, set_nodata=None, out_path=os.path.join(FOLDER, "mask_10m.tif"))

    if not os.path.exists(os.path.join(FOLDER, "resampled")):
        os.mkdir(os.path.join(FOLDER, "resampled"))

    for img in glob(FOLDER + "*.tif"):
        if "_20m" not in img:
            arr = beo.raster_to_array(img, filled=True, fill_value=0, cast="uint16")
            beo.array_to_raster(
                arr,
                reference=img,
                set_nodata=None,
                out_path=os.path.join(FOLDER, "resampled", os.path.basename(img).replace("_10m", "")),
            )
        else:
            resampled = beo.raster_resample(
                img,
                target_size=mask_aligned,
                resample_alg="bilinear",
            )
            resampled_clip = beo.raster_clip(resampled, mask_aligned)
            resampled_arr = beo.raster_to_array(resampled_clip, filled=True, fill_value=0, cast="uint16")
            beo.array_to_raster(
                resampled_arr,
                reference=mask_aligned,
                set_nodata=None,
                out_path=os.path.join(FOLDER, "resampled", os.path.basename(img).replace("_20m", "")),
            )

    assert beo.check_rasters_are_aligned(glob(FOLDER + "/resampled/*.tif"), same_nodata=True)
