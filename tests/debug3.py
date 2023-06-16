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

# for idx, FOLDER in enumerate(tqdm(FOLDERS, total=len(FOLDERS))):
#     mask = os.path.join(FOLDER, "mask.gpkg")
#     label = os.path.join(FOLDER, "labels_10m.tif")

#     # First we ensure that there is a mask and that it is aligned with the labels.
#     clip_geom = beo.raster_to_extent(label)
#     clipped = beo.vector_clip(mask, clip_geom, to_extent=True)
#     masked = beo.vector_rasterize(clipped, label, extent=mask)
#     aligned = beo.raster_align(masked, reference=label)[0]

#     arr_mask = beo.raster_to_array(aligned, filled=True, fill_value=0, cast="uint8")
#     mask_aligned = beo.array_to_raster(arr_mask, reference=label, set_nodata=None, out_path=os.path.join(FOLDER, "mask_10m.tif"))

#     mask_bbox = beo.raster_to_metadata(mask_aligned)["bbox"]

#     # Then we resample the 20m images to 10m.
#     # The 10m images are loaded and the nodata values are removed and replaced with 0.
#     if not os.path.exists(os.path.join(FOLDER, "resampled")):
#         os.mkdir(os.path.join(FOLDER, "resampled"))

#     for img in glob(FOLDER + "*.tif"):

#         if "_20m" not in img:
#             arr = beo.raster_to_array(img, filled=True, fill_value=0, cast="uint16")
#             beo.array_to_raster(
#                 arr,
#                 reference=img,
#                 set_nodata=None,
#                 out_path=os.path.join(FOLDER, "resampled", os.path.basename(img).replace("_10m", "")),
#             )
#         else:
#             resampled = beo.raster_resample(
#                 img,
#                 target_size=mask_aligned,
#                 resample_alg="bilinear",
#             )
#             resampled_arr = beo.raster_to_array(resampled, filled=True, fill_value=0, cast="uint16", bbox=mask_bbox)
#             beo.array_to_raster(
#                 resampled_arr,
#                 reference=mask_aligned,
#                 set_nodata=None,
#                 out_path=os.path.join(FOLDER, "resampled", os.path.basename(img).replace("_20m", "")),
#             )

#     # Are all the images aligned?
#     assert beo.check_rasters_are_aligned(glob(FOLDER + "/resampled/*.tif"), same_nodata=True)


# for FOLDER in FOLDERS:
#     images = sorted(glob(FOLDER + "/subset/EGY1_label*.tif"))

#     beo.raster_get_footprints(
#         images,
#         latlng=True,
#         out_path=FOLDER + "subset/",
#         suffix="_footprint",
#     )


# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/egypt/",
# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/ghana/",
# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/israel_gaza/",
# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_dar/",
# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_kigoma/",
# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_kilimanjaro/",
# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_mwanza_Q2/",
# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_mwanza_Q3/",
# "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/uganda/",

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/egypt/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/EGY_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/ghana/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/GHA_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/israel_gaza/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/ISR_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_dar/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/TZA1_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_kigoma/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/TZA2_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_kilimanjaro/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/TZA3_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_mwanza_Q2/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/TZA4_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/tanzania_mwanza_Q3/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/TZA5_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)

b08 = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/uganda/B08.tif"
b8A = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/B8A/UGA_B8A.tif"

beo.array_to_raster(
    beo.raster_to_array(b8A, filled=True, fill_value=0, cast="uint16"),
    reference=b08,
    out_path=b08.replace("B08", "B8A")
)
