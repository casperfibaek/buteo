# Standard library
import sys; sys.path.append("../")

import os
import numpy as np
from buteo import (
    raster_to_array,
    array_to_raster,
    reproject_raster,
    split_into_offsets,
    raster_to_metadata,
    delete_if_in_memory,
    stack_rasters_vrt,
    align_rasters,
)
from numba import jit, prange

FOLDER = r"C:/Users/casper.fibaek/OneDrive - ESA/Desktop/Unicef/"
path_mask_54009 = os.path.join(FOLDER, "VNL_Land_Mask_54009.tif")
path_mask_4326 = os.path.join(FOLDER, "VNL_Land_Mask_4326.tif")


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_width(lng, lng_max):
    """ Longitude goes from -180 to 180 """

    encoded_sin = ((np.sin(2 * np.pi * (lng / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_arr_position(arr):
    """ Fast encoding of lng coordinates """
    result = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)

    col_end = arr.shape[0] - 1
    row_end = arr.shape[1] - 1

    col_range = np.arange(0, arr.shape[0]).astype(np.float32)
    row_range = np.arange(0, arr.shape[1]).astype(np.float32)

    col_encoded = np.zeros((col_range.shape[0], 1), dtype=np.float32)
    row_encoded = np.zeros((row_range.shape[0], 2), dtype=np.float32)

    for col in prange(col_range.shape[0]):
        col_encoded[col, :] = col_range[col] / col_end

    for row in prange(row_range.shape[0]):
        row_encoded[row, :] = encode_width(row_range[row], row_end)

    for col in prange(arr.shape[0]):
        for row in range(arr.shape[1]):
            result[col, row, 0] = row_encoded[row, 0]
            result[col, row, 1] = row_encoded[row, 1]
            result[col, row, 2] = col_encoded[col, 0]

    return result

if not os.path.exists(os.path.join(FOLDER, "encoded_coordinates_4326_v01.tif")):
    array_to_raster(
        encode_arr_position(raster_to_array(path_mask_4326)),
        reference=path_mask_4326,
        out_path=os.path.join(FOLDER, "encoded_coordinates_4326_v01.tif"),
    )

path_encoded_4326 = os.path.join(FOLDER, "encoded_coordinates_4326_v01.tif")

merged = []

shape_4326 = raster_to_metadata(path_encoded_4326)["shape"]
for idx, offset in enumerate(split_into_offsets(shape_4326, 3, 3, 3, 3)):
    out_path = os.path.join(FOLDER, f"encoded_coordinates_54009_v01_{offset[0]}_{offset[1]}.tif")

    if os.path.exists(out_path):
        merged.append(out_path)
        continue

    arr_chunk = raster_to_array(path_encoded_4326, pixel_offsets=offset)
    raster_chunk = array_to_raster(arr_chunk, reference=path_encoded_4326, pixel_offsets=offset)

    print(f"Creating: {out_path}")
    reproject_raster(
        raster_chunk,
        projection=path_mask_54009,
        out_path=out_path,
        dst_nodata=-1.0,
    )

    merged.append(out_path)
    delete_if_in_memory(raster_chunk)

print("Stacking")
stacked = stack_rasters_vrt(
    merged,
    seperate=False,
    VRTNodata=-1.0,
    srcNodata=-1.0,
    out_path=os.path.join(FOLDER, "encoded_coordinates_54009_v02.vrt")
)

print("Aligning")
aligned = align_rasters(
    stacked,
    master=path_mask_54009,
    out_path=os.path.join(FOLDER, "encoded_coordinates_54009_v02.tif"),
)

print("Done")
