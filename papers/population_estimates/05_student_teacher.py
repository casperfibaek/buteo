yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import os

sys.path.append(yellow_follow)

import numpy as np
from glob import glob
from osgeo import gdal
from buteo.raster.io import stack_rasters, raster_to_array
from buteo.machine_learning.patch_extraction_v2 import predict_raster
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    stack_rasters_vrt,
)
from buteo.raster.clip import internal_clip_raster, clip_raster
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
    tpe,
    get_offsets,
)

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/"
vector_folder = folder + "vector/grid_cells_student2/"
raster_folder = folder + "raster/"

model = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/models/student1"


for region in glob(vector_folder + "grid_id_*.gpkg"):
    region_name = os.path.splitext(os.path.basename(region))[0]

    print(f"Processing region: {region_name}")

    print("Clipping RESWIR.")
    b20m_clip = internal_clip_raster(
        raster_folder + "2021_B05_20m.tif",
        region,
        adjust_bbox=False,
        all_touch=False,
        out_path="/vsimem/20m_clip.tif",
    )

    reswir = clip_raster(
        [
            raster_folder + "2021_B05_20m.tif",
            raster_folder + "2021_B06_20m.tif",
            raster_folder + "2021_B07_20m.tif",
            raster_folder + "2021_B11_20m.tif",
            raster_folder + "2021_B12_20m.tif",
        ],
        clip_geom=region,
        adjust_bbox=False,
        all_touch=False,
    )

    print("Stacking RESWIR.")
    reswir_stack = []
    for idx, raster in enumerate(reswir):
        reswir_stack.append(
            array_to_raster(
                preprocess_optical(
                    raster_to_array(reswir[idx]),
                    target_low=0,
                    target_high=1,
                    cutoff_high=8000,
                ),
                reference=reswir[idx],
            ),
        )
    reswir_stacked = stack_rasters(reswir_stack, dtype="float32")
    for raster in reswir:
        gdal.Unlink(raster)

    print("Clipping RGBN.")
    b10m_clip = internal_clip_raster(
        raster_folder + "2021_B04_10m.tif",
        b20m_clip,
        adjust_bbox=False,
        all_touch=False,
        out_path="/vsimem/10m_clip.tif",
    )
    rgbn = clip_raster(
        [
            raster_folder + "2021_B02_10m.tif",
            raster_folder + "2021_B03_10m.tif",
            raster_folder + "2021_B04_10m.tif",
            raster_folder + "2021_B08_10m.tif",
        ],
        clip_geom=b20m_clip,
        adjust_bbox=False,
        all_touch=False,
    )

    print("Stacking RGBN.")
    rgbn_stack = []
    for idx, raster in enumerate(rgbn):
        rgbn_stack.append(
            array_to_raster(
                preprocess_optical(
                    raster_to_array(rgbn[idx]),
                    target_low=0,
                    target_high=1,
                    cutoff_high=8000,
                ),
                reference=rgbn[idx],
            ),
        )
    rgbn_stacked = stack_rasters(rgbn_stack, dtype="float32")
    for raster in rgbn:
        gdal.Unlink(raster)

    print("Clipping SAR.")
    sar = clip_raster(
        [
            raster_folder + "2021_VV_10m.tif",
            raster_folder + "2021_VH_10m.tif",
        ],
        clip_geom=b20m_clip,
        adjust_bbox=False,
        all_touch=False,
    )

    print("Stacking SAR.")
    sar_stack = []
    for idx, raster in enumerate(sar):
        sar_stack.append(
            array_to_raster(
                preprocess_sar(raster_to_array(sar[idx]), target_low=0, target_high=1),
                reference=sar[idx],
            ),
        )
    sar_stacked = stack_rasters(sar_stack, dtype="float32")
    for raster in sar:
        gdal.Unlink(raster)

    print("Ready for predictions.")

    number = os.path.splitext(os.path.basename(region))[0].split("_")[2]
    outname = f"fid_{number}_rasterized.tif"

    predict_raster(
        [rgbn_stacked, sar_stacked, reswir_stacked],
        tile_size=[32, 32, 16],
        output_tile_size=32,
        model_path=model,
        reference_raster=b10m_clip,
        out_path=vector_folder + outname,
        offsets=[
            get_offsets(32),
            get_offsets(32),
            get_offsets(16),
        ],
        batch_size=1024,
    )

    try:
        for raster in reswir_stack:
            gdal.Unlink(raster)

        for raster in rgbn_stack:
            gdal.Unlink(raster)

        for raster in sar_stack:
            gdal.Unlink(raster)

        gdal.Unlink(reswir_stacked)
        gdal.Unlink(rgbn_stacked)
        gdal.Unlink(sar_stacked)
        gdal.Unlink(b10m_clip)
    except:
        pass

test_sites = glob(vector_folder + "fid_*_rasterized.tif")

from buteo.raster.align import align_rasters
from buteo.machine_learning.patch_extraction import extract_patches
from glob import glob
from osgeo import gdal

for cell in test_sites:
    number = os.path.splitext(os.path.basename(cell))[0].split("_")[1]

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
        out_dir=vector_folder + "patches/",
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
        clip_geom=cell,
        verify_output=False,
        verification_samples=100,
        verbose=1,
    )

    path_np, path_geom = extract_patches(
        m20,
        out_dir=vector_folder + "patches/",
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
        clip_geom=cell,
        verify_output=False,
        verification_samples=100,
        verbose=1,
    )


def sortKeyFunc(s):
    return int(os.path.basename(s).split("_")[0])


folder = vector_folder

for band in ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "VV", "VH"]:
    band_paths = glob(folder + f"patches/*2021_{band}*.npy")
    band_paths = sorted(band_paths, key=sortKeyFunc)

    # merge all in bob by band
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

band_paths = glob(folder + f"patches/*rasterized*.npy")
band_paths = sorted(band_paths, key=sortKeyFunc)

# merge all in bob by band
for index, key in enumerate(band_paths):
    if index == 0:
        loaded = np.load(key)
    else:
        loaded = np.concatenate([loaded, np.load(key)])

np.save(folder + "patches/merged/raw/label_area.npy", loaded)
