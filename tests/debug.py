""" This is a debug script, used for ad-hoc testing. """

# Standard library
import sys; sys.path.append("../")
import os
from glob import glob
from osgeo import gdal

from buteo.vector.split import split_vector_by_fid
from buteo.raster import (
    raster_to_array,
    array_to_raster,
    clip_raster,
    reproject_raster,
    resample_raster,
    align_rasters,
    raster_to_metadata,
    raster_dem_to_slope,
    raster_dem_to_aspect,
    raster_dem_to_orientation,
)
from buteo.raster.convolution import simple_blur_kernel_2d_3x3, convolve_array_simple

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/ccai_tutorial/gaza_israel/"
FOLDER_OUT = FOLDER + "patches/"
PROCESS_DEM = False

dem_path = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/ccai_tutorial/DEM_east_mediterranean.tif"
mask_path = os.path.join(FOLDER, "labels_10m.tif")

if PROCESS_DEM:
    resample_method = "cubic_spline"

    clip_raster(
        dem_path,
        mask_path,
        out_path = os.path.join(FOLDER, "DEM_10m_clipped.tif"),
        resample_alg=resample_method,
    )

    reproject_raster(
        os.path.join(FOLDER, "DEM_10m_clipped.tif"),
        projection=mask_path,
        out_path=os.path.join(FOLDER, "DEM_10m_resampled.tif"),
        resample_alg=resample_method,
    )

    align_rasters(
        os.path.join(FOLDER, "DEM_10m_resampled.tif"),
        reference=mask_path,
        out_path=os.path.join(FOLDER, "DEM_10m.tif"),
        resample_alg=resample_method,
    )

    raster_dem_to_orientation(
        os.path.join(FOLDER, "DEM_10m.tif"),
        os.path.join(FOLDER, "ORIENTATION_10m.tif"),
    )

for img in glob(FOLDER + "*_20m.tif"):
    resampled = resample_raster(
        img,
        target_size=mask_path,
        resample_alg="bilinear",
    )
    align_rasters(
        resampled,
        reference=mask_path,
        out_path=img.replace("_20m.tif", "_10m.tif"),
        resample_alg="bilinear",
    )

mask_raster_path = os.path.join(FOLDER, "labels_10m.tif")

split_files = split_vector_by_fid(
    os.path.join(FOLDER, "mask.gpkg"),
)
for idx, split_file in enumerate(split_files):
    mask_clipped = clip_raster(
        mask_raster_path,
        split_file,
        adjust_bbox=True,
    )

    mask_arr = raster_to_array(mask_clipped, filled=True, fill_value=0)

    array_to_raster(
        mask_arr,
        reference=mask_clipped,
        out_path=os.path.join(FOLDER_OUT, f"label_{idx}.tif"),
    )

    bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]
    s2_clipped = clip_raster(
        [os.path.join(FOLDER, f"{band}_10m.tif") for band in bands],
        split_file,
        adjust_bbox=True,
    )

    s2_arr = raster_to_array(s2_clipped, filled=True, fill_value=0)

    array_to_raster(
        s2_arr,
        reference=s2_clipped,
        out_path=os.path.join(FOLDER_OUT, f"features_s2_{idx}.tif"),
    )

    bands = ["VV", "VH"]
    s1_clipped = clip_raster(
        [os.path.join(FOLDER, f"{band}_10m.tif") for band in bands],
        split_file,
        adjust_bbox=True,
    )

    s1_arr = raster_to_array(s1_clipped, filled=True, fill_value=0)

    array_to_raster(
        s1_arr,
        reference=s1_clipped,
        out_path=os.path.join(FOLDER_OUT, f"features_s1_{idx}.tif"),
    )

    dem_clipped = clip_raster(
        os.path.join(FOLDER, "ORIENTATION_10m.tif"),
        split_file,
        adjust_bbox=True,
    )

    array_to_raster(
        dem_clipped,
        reference=dem_clipped,
    )