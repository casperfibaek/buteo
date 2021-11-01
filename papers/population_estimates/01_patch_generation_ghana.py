import sys
from typing import Type

sys.path.append("../../")
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.resample import internal_resample_raster
from buteo.raster.clip import clip_raster
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.align import align_rasters
from buteo.utils import progress
from buteo.machine_learning.patch_extraction import extract_patches
from glob import glob
from osgeo import gdal

import os
import numpy as np

# rasterize buildings @ 50cm, use grid as extent
# resample to 10m, use summation
# clip all images to resampled
# convert to rgbn, swir, sar
# preprocess rgbn, swir, sar

folder = (
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/grid_cells/"
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/grid_cells_student/"
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/volume_rasters/"
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/comparisions/"
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/grid_cells_test/"
)
# raster_folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster_v2/"
predictions = (
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/results/ghana_area_float32.tif"
)

# folder2 = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/comparisions/prediction/"

# y_pred = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/predictions/Ghana_float32_v5_teacher.tif"

grids = glob(folder + "grid_id_*.tif")

for idx, grid_cell in enumerate(grids):
    progress(idx, len(grids), "Rasterizing")

    name = os.path.basename(grid_cell)
    number = os.path.splitext(name)[0].split("_")[-1]

    # clip_raster(
    #     y_pred,
    #     grid_cell,
    #     out_path=folder + f"fid_{number}_rasterized.tif",
    #     all_touch=False,
    #     adjust_bbox=False,
    #     postfix="",
    # )

    buildings = folder + "building_id_" + number + ".gpkg"

    try:
        rasterize_vector(
            buildings,
            0.2,
            out_path=f"/vsimem/fid_{number}_rasterized.tif",
            extent=grid_cell,
        )
    except:
        rasterize_vector(
            grid_cell,
            0.2,
            out_path=f"/vsimem/fid_{number}_rasterized.tif",
            extent=grid_cell,
            fill_value=0,
            burn_value=0,
        )

    internal_resample_raster(
        f"/vsimem/fid_{number}_rasterized.tif",
        10.0,
        resample_alg="average",
        out_path=f"/vsimem/fid_{number}_resampled.tif",
    )

    array_to_raster(
        (raster_to_array(f"/vsimem/fid_{number}_resampled.tif") * 100).astype(
            "float32"
        ),
        reference=f"/vsimem/fid_{number}_resampled.tif",
        # out_path=folder + f"fid_{number}_rasterized.tif",
        out_path=folder + f"Google00_{number}.tif",
    )

    gdal.Unlink(f"/vsimem/fid_{number}_rasterized.tif")
    gdal.Unlink(f"/vsimem/fid_{number}_resampled.tif")

    progress(idx + 1, len(grids), "Rasterizing")

exit()


# for cell in glob(folder + "fid_*_rasterized.tif"):
# for cell in glob(folder + "class_*.tif"):
for cell in glob(folder + "volume_*.tif"):
    # number = os.path.basename(os.path.basename(cell)).split("_")[1]
    number = os.path.splitext(os.path.basename(cell))[0]
    number = number.split("_")[1]

    # vector_cell = folder + "grid_id_" + number + ".gpkg"
    vector_cell = cell

    clipped_rgbn = clip_raster(
        [
            raster_folder + "2021_B02_10m.tif",
            raster_folder + "2021_B03_10m.tif",
            raster_folder + "2021_B04_10m.tif",
            raster_folder + "2021_B08_10m.tif",
        ],
        cell,
        all_touch=False,
        adjust_bbox=False,
        postfix="",
    )

    clipped_reswir = clip_raster(
        [
            raster_folder + "2021_B05_20m.tif",
            raster_folder + "2021_B06_20m.tif",
            raster_folder + "2021_B07_20m.tif",
            raster_folder + "2021_B11_20m.tif",
            raster_folder + "2021_B12_20m.tif",
        ],
        cell,
        all_touch=False,
        adjust_bbox=False,
        postfix="",
    )

    clipped_sar = clip_raster(
        [
            raster_folder + "2021_VV_10m.tif",
            raster_folder + "2021_VH_10m.tif",
        ],
        cell,
        all_touch=False,
        adjust_bbox=False,
        postfix="",
    )

    swir = align_rasters(
        clipped_reswir,
        bounding_box=cell,
        postfix="",
    )

    rgbn = align_rasters(
        clipped_rgbn,
        target_in_pixels=cell,
        bounding_box=cell,
        postfix="",
    )
    sar = align_rasters(
        clipped_sar,
        target_in_pixels=cell,
        bounding_box=cell,
        postfix="",
    )

    m10 = [cell]
    m20 = []

    for path in rgbn:
        m10.append(path)

    for path in sar:
        m10.append(path)

    for path in swir:
        m20.append(path)

    extract_patches(
        m10,
        out_dir=folder + "patches/",
        prefix=number + "_",
        postfix="",
        size=32,
        offsets=[
            (0, 8),
            (0, 16),
            (0, 24),
            (8, 0),
            (8, 8),
            (8, 16),
            (8, 24),
            (16, 0),
            (16, 8),
            (16, 16),
            (16, 24),
            (24, 0),
            (24, 8),
            (24, 16),
            (24, 24),
        ],
        generate_grid_geom=True,
        generate_zero_offset=True,
        generate_border_patches=True,
        clip_geom=vector_cell,
        verify_output=False,
        verification_samples=100,
        verbose=1,
    )

    path_np, path_geom = extract_patches(
        m20,
        out_dir=folder + "patches/",
        prefix=number + "_",
        postfix="",
        size=16,
        offsets=[
            (0, 4),
            (0, 8),
            (0, 12),
            (4, 0),
            (4, 4),
            (4, 8),
            (4, 12),
            (8, 0),
            (8, 4),
            (8, 8),
            (8, 12),
            (12, 0),
            (12, 4),
            (12, 8),
            (12, 12),
        ],
        generate_grid_geom=True,
        generate_zero_offset=True,
        generate_border_patches=True,
        clip_geom=vector_cell,
        verify_output=False,
        verification_samples=100,
        verbose=1,
    )


def sortKeyFunc(s):
    return int(os.path.basename(s).split("_")[0])


for band in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "VV", "VH"]:
    band_paths = glob(folder + f"patches/*2021_{band}*.npy")
    band_paths = sorted(band_paths, key=sortKeyFunc)

    # merge all by band
    for index, key in enumerate(band_paths):
        if index == 0:
            loaded = np.load(key)
        else:
            loaded = np.concatenate([loaded, np.load(key)])

    add_list = ["B05", "B06", "B07", "B11", "B12"]
    addition = "10m"
    if band in add_list:
        addition = "20m"

    np.save(folder + f"patches/merged/raw/{band}_{addition}.npy", loaded)

# band_paths = glob(folder + f"patches/*rasterized*.npy")
band_paths = glob(folder + f"patches/*volume*.npy")
# band_paths = glob(folder + f"patches/*class*.npy")
band_paths = sorted(band_paths, key=sortKeyFunc)

# merge all by band
for index, key in enumerate(band_paths):
    if index == 0:
        loaded = np.load(key)
    else:
        loaded = np.concatenate([loaded, np.load(key)])

np.save(folder + "patches/merged/raw/label_volume_10m.npy", loaded)
