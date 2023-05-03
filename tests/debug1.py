""" This is a debug script, used for ad-hoc testing. """
# disable all of pylint for this file only.
# pylint: disable-all

# Standard library
import sys; sys.path.append("../")
import os
from glob import glob

from buteo.vector.split import vector_split_by_fid
from buteo.raster import (
    raster_clip,
    raster_reproject,
    raster_resample,
    raster_align,
    raster_dem_to_orientation,
    check_rasters_are_aligned,
)
from buteo.vector import vector_reproject

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/ccai_tutorial/egypt/"
FOLDER_OUT = FOLDER + "patches/"
PROCESS_DEM = False
PROCESS_RESAMPLE = False

dem_path = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/ccai_tutorial/DEM_east_mediterranean.tif"
mask_path = os.path.join(FOLDER, "labels_10m.tif")

if PROCESS_DEM:
    resample_method = "cubic_spline"

    raster_clip(
        dem_path,
        mask_path,
        out_path = os.path.join(FOLDER, "DEM_10m_clipped.tif"),
        resample_alg=resample_method,
    )

    raster_reproject(
        os.path.join(FOLDER, "DEM_10m_clipped.tif"),
        projection=mask_path,
        out_path=os.path.join(FOLDER, "DEM_10m_reprojected.tif"),
        resample_alg=resample_method,
    )

    raster_align(
        os.path.join(FOLDER, "DEM_10m_reprojected.tif"),
        reference=mask_path,
        out_path=os.path.join(FOLDER, "DEM_10m.tif"),
        resample_alg=resample_method,
    )

    raster_dem_to_orientation(
        os.path.join(FOLDER, "DEM_10m.tif"),
        os.path.join(FOLDER, "ORIENTATION_10m.tif"),
        include_height=True,
    )

if PROCESS_RESAMPLE:
    for img in glob(FOLDER + "*_20m.tif"):
        resampled = raster_resample(
            img,
            target_size=mask_path,
            resample_alg="bilinear",
        )
        raster_align(
            resampled,
            reference=mask_path,
            out_path=img.replace("_20m.tif", "_10m.tif"),
            resample_alg="bilinear",
        )

mask_raster_path = os.path.join(FOLDER, "labels_10m.tif")
s1_stacked = os.path.join(FOLDER, "s1.vrt")
s2_stacked = os.path.join(FOLDER, "s2.vrt")
dem = os.path.join(FOLDER, "ORIENTATION_10m.tif")

split_files = vector_split_by_fid(
    vector_reproject(os.path.join(FOLDER, "mask.gpkg"), mask_raster_path),
)

for idx, split_file in enumerate(split_files):
    idx_offset = idx + 9
    mask_clipped = raster_clip(
        mask_raster_path,
        split_file,
        out_path=os.path.join(FOLDER_OUT, f"label_{idx_offset}.tif"),
        adjust_bbox=True,
    )

    s1_clipped = raster_clip(
        s1_stacked,
        split_file,
        out_path=os.path.join(FOLDER_OUT, f"s1_{idx_offset}.tif"),
        adjust_bbox=True,
    )

    s2_clipped = raster_clip(
        s2_stacked,
        split_file,
        out_path=os.path.join(FOLDER_OUT, f"s2_{idx_offset}.tif"),
        adjust_bbox=True,
    )

    dem_clipped = raster_clip(
        dem,
        split_file,
        out_path=os.path.join(FOLDER_OUT, f"dem_{idx_offset}.tif"),
        adjust_bbox=True,
    )

    if not check_rasters_are_aligned([mask_clipped, s1_clipped, s2_clipped, dem_clipped]):
        print("Rasters are not aligned: ", idx_offset)
