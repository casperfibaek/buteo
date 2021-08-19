import sys

sys.path.append("../../")
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.resample import internal_resample_raster
from buteo.raster.clip import clip_raster
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.align import align_rasters
from buteo.machine_learning.patch_extraction import extract_patches
from glob import glob
import os

# rasterize buildings @ 50cm, use grid as extent
# resample to 10m, use summation
# clip all images to resampled
# convert to rgbn, swir, sar
# preprocess rgbn, swir, sar

folder = (
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/grid_cells/"
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/grid_cells4/"
)
raster_folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/"

# for cell in glob(folder + "fid*.gpkg"):
for cell in glob(folder + "grid_id_*.gpkg"):
    name = os.path.basename(cell)
    # number = os.path.splitext(name.split("_")[1])[1]
    number = os.path.splitext(name)[0].split("_")[-1]

    grid = cell
    # buildings = folder + "grid_fid_" + number + ".gpkg"
    buildings = folder + "building_grid_fid_" + number + ".gpkg"

    rasterize_vector(
        buildings,
        0.2,
        out_path=f"/vsimem/fid_{number}_rasterized.tif",
        extent=grid,
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
        # out_path=f"/vsimem/fid_{number}_scaled.tif",
        out_path=folder + f"fid_{number}_rasterized.tif",
    )

for cell in glob(folder + "fid*_rasterized.tif"):
    number = os.path.basename(os.path.basename(cell)).split("_")[1]
    # number = os.path.splitext(name)[0].split("_")[-1]
    # vector_cell = folder + "fid_" + number + ".gpkg"
    vector_cell = folder + "grid_id_" + number + ".gpkg"

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

    clipped_sar = clip_raster(
        [
            raster_folder + "2021_VH_10m.tif",
            raster_folder + "2021_VV_10m.tif",
        ],
        cell,
        all_touch=False,
        adjust_bbox=False,
        postfix="",
    )

    clipped_swir = clip_raster(
        [
            raster_folder + "2021_B11_20m.tif",
            raster_folder + "2021_B12_20m.tif",
        ],
        cell,
        all_touch=False,
        adjust_bbox=False,
        postfix="",
    )

    swir = align_rasters(
        clipped_swir,
        # folder + "clipped/",
        bounding_box=cell,
        postfix="",
    )

    rgbn = align_rasters(
        clipped_rgbn,
        # folder + "clipped/",
        target_in_pixels=cell,
        bounding_box=cell,
        postfix="",
    )
    sar = align_rasters(
        clipped_sar,
        # folder + "clipped/",
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
        size=64,
        offsets=[
            (0, 16),
            (0, 32),
            (0, 48),
            (16, 0),
            (16, 16),
            (16, 32),
            (16, 48),
            (32, 0),
            (32, 16),
            (32, 32),
            (32, 48),
            (48, 0),
            (48, 16),
            (48, 32),
            (48, 48),
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


import numpy as np


def sortKeyFunc(s):
    return int(os.path.basename(s).split("_")[0])


for band in ["B02", "B03", "B04", "B08", "B11", "B12", "VV", "VH"]:
    band_paths = glob(folder + f"patches/*2021_{band}*.npy")
    band_paths = sorted(band_paths, key=sortKeyFunc)

    # merge all in bob by band
    for index, key in enumerate(band_paths):
        if index == 0:
            loaded = np.load(key)
        else:
            loaded = np.concatenate([loaded, np.load(key)])

    np.save(folder + f"patches/merged/{band}.npy", loaded)

band_paths = glob(folder + f"patches/*rasterized*.npy")
band_paths = sorted(band_paths, key=sortKeyFunc)

# merge all in bob by band
for index, key in enumerate(band_paths):
    if index == 0:
        loaded = np.load(key)
    else:
        loaded = np.concatenate([loaded, np.load(key)])

np.save(folder + "patches/merged/AREA.npy", loaded)
