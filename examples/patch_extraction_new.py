import sys, os

sys.path.append("..")
import numpy as np

from glob import glob
from buteo.raster.align import rasters_are_aligned
from buteo.raster.io import raster_to_metadata
from buteo.raster.clip import clip_raster
from buteo.machine_learning.patch_extraction import extract_patches, create_labels
from buteo.machine_learning.ml_utils import preprocess_optical, preprocess_sar

folder = "D:/training_data_buteo/tanzania_dar/"
folder_tmp = "D:/training_data_buteo/tmp/"
convert_sar_to_db = True

reference = folder + "B12_20m.tif"
# clip_raster(
#     glob(folder + "B*_10m.tif"),
#     clip_geom=reference,
#     out_path=folder,
#     to_extent=True,
#     all_touch=False,
#     adjust_bbox=True,
#     postfix="_aligned",
#     dst_nodata=0,
# )

# clip_raster(
#     glob(folder + "V*_10m.tif"),
#     clip_geom=reference,
#     out_path=folder,
#     to_extent=True,
#     all_touch=False,
#     adjust_bbox=True,
#     postfix="_aligned",
#     dst_nodata=-9999.9,
# )

# exit()

print("Validating input data.")
if not rasters_are_aligned(glob(folder + "*_10m.tif")):
    raise Exception("10m not aligned.")

if not rasters_are_aligned(glob(folder + "*_20m.tif")):
    raise Exception("20m not aligned.")

size_20m_meta = raster_to_metadata(folder + "B12_20m.tif")
size_20m = (int(size_20m_meta["width"] * 2), int(size_20m_meta["height"] * 2))
for img in glob(folder + "*_10m.tif"):
    meta = raster_to_metadata(img)
    if meta["width"] != size_20m[0]:
        raise Exception(f"{img} has wrong size.")
    elif meta["height"] != size_20m[1]:
        raise Exception(f"{img} has wrong size.")


grid = sorted(glob(folder + "grid/*.gpkg"))
if len(grid) == 0:
    grid = [folder + "mask.gpkg"]

print("Creating rasterized labels.")
create_labels(
    folder + "buildings.gpkg",
    reference=folder + "B04_10m.tif",
    grid=grid,
    out_path=folder + "labels_10m.tif",
    tmp_folder=folder_tmp,
    round_label=2,
    resample_from=0.4,
)

print("Extracting 10m patches.")
extract_patches(
    glob(folder + "*_10m.tif"),
    folder + "patches/",
    tile_size=32,
    zones=folder + "mask.gpkg",
    options={
        "output_zone_masks": False,
        "mask_reference": folder + "B12_20m.tif",
    }
)

print("Extracting 20m patches.")
extract_patches(
    glob(folder + "*_20m.tif"),
    folder + "patches/",
    tile_size=16,
    zones=folder + "mask.gpkg",
    options={
        "output_zone_masks": False,
    }
)

shuffle_mask = np.random.permutation(np.load(folder + "patches/B12_20m.npy").shape[0])

print("Preprocessing RGBN.")
rgbn = preprocess_optical(
    np.stack(
        [
            np.load(folder + "patches/B02_10m.npy"),
            np.load(folder + "patches/B03_10m.npy"),
            np.load(folder + "patches/B04_10m.npy"),
            np.load(folder + "patches/B08_10m.npy"),
        ],
        axis=3,
    )[:, :, :, :, 0],
    cutoff_low=0,
    cutoff_high=10000,
    target_low=0,
    target_high=1,
)

print("Saving RGBN.")
np.savez_compressed(folder + "patches/RGBN", rgbn=rgbn[shuffle_mask])
rgbn = None

print("Preprocessing RESWIR.")
reswir = preprocess_optical(
    np.stack(
        [
            np.load(folder + "patches/B05_20m.npy"),
            np.load(folder + "patches/B06_20m.npy"),
            np.load(folder + "patches/B07_20m.npy"),
            np.load(folder + "patches/B11_20m.npy"),
            np.load(folder + "patches/B12_20m.npy"),
        ],
        axis=3,
    )[:, :, :, :, 0],
    cutoff_low=0,
    cutoff_high=10000,
    target_low=0,
    target_high=1,
)

print("Saving RESWIR.")
np.savez_compressed(folder + "patches/RESWIR", reswir=reswir[shuffle_mask])
reswir = None

print("Preprocessing SAR.")
sar = preprocess_sar(
    np.stack(
        [
            np.load(folder + "patches/VV_10m.npy"),
            np.load(folder + "patches/VH_10m.npy"),
        ],
        axis=3,
    )[:, :, :, :, 0],
    convert_db=convert_sar_to_db,
    cutoff_low=-30,
    cutoff_high=20,
    target_low=0,
    target_high=1,
)

print("Saving SAR.")
np.savez_compressed(folder + "patches/SAR", reswir=sar[shuffle_mask])
sar = None

print("Compressing labels")
np.savez_compressed(
    folder + "patches/LABEL",
    label=np.load(folder + "patches/labels_10m.npy")[shuffle_mask],
)

for f in glob(folder + "patches/*.npy"):
    try:
        os.remove(f)
    except:
        pass