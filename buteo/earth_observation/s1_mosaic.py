import sys

sys.path.append("..")
sys.path.append("../../")

from glob import glob
from numba import jit, prange
from buteo.raster.io import (
    internal_raster_to_metadata,
    raster_to_array,
    array_to_raster,
    rasters_intersect,
)
from buteo.vector.io import internal_vector_to_metadata
from buteo.gdal_utils import parse_projection, ogr_bbox_intersects
from buteo.raster.align import rasters_are_aligned, align_rasters
from buteo.raster.clip import clip_raster
from buteo.raster.reproject import reproject_raster
from buteo.filters.kernel_generator import create_kernel
from buteo.utils import timing, progress
from osgeo import gdal, osr
from time import time
import numpy as np
import os
import datetime
from uuid import uuid4


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_quantile(values, weights, quant):
    sort_mask = np.argsort(values)
    sorted_data = values[sort_mask]
    sorted_weights = weights[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quant, intersect, sorted_data)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def s1_collapse(
    arr,
    offsets,
    weights,
    quantile=0.5,
    nodata=True,
    nodata_value=-9999.0,
    weighted=True,
):
    x_adj = arr.shape[0] - 1
    y_adj = arr.shape[1] - 1
    z_adj = (arr.shape[2] - 1) // 2

    hood_size = len(offsets)
    if nodata:
        result = np.full(arr.shape[:2], nodata_value, dtype="float32")
    else:
        result = np.zeros(arr.shape[:2], dtype="float32")

    for x in prange(arr.shape[0]):
        for y in range(arr.shape[1]):

            hood_values = np.zeros(hood_size, dtype="float32")
            hood_weights = np.zeros(hood_size, dtype="float32")
            weight_sum = np.array([0.0], dtype="float32")
            normalise = False

            for n in range(hood_size):
                offset_x = x + offsets[n][0]
                offset_y = y + offsets[n][1]
                offset_z = offsets[n][2]

                outside = False

                if offset_z < -z_adj:
                    offset_z = -z_adj
                    outside = True
                elif offset_z > z_adj:
                    offset_z = z_adj
                    outside = True

                if offset_x < 0:
                    offset_x = 0
                    outside = True
                elif offset_x > x_adj:
                    offset_x = x_adj
                    outside = True

                if offset_y < 0:
                    offset_y = 0
                    outside = True
                elif offset_y > y_adj:
                    offset_y = y_adj
                    outside = True

                value = arr[offset_x, offset_y, offset_z]

                if outside or (nodata and value == nodata_value):
                    normalise = True
                    hood_weights[n] = 0
                else:
                    hood_values[n] = value
                    weight = weights[n]

                    hood_weights[n] = weight
                    weight_sum[0] += weight

            if normalise:
                hood_weights = np.divide(hood_weights, weight_sum[0])

            if weighted:
                result[x, y] = hood_quantile(hood_values, hood_weights, quantile)
            else:
                result[x, y] = np.median(hood_values[np.nonzero(hood_weights)])

    return result


def name_to_date(path):
    timetag = os.path.basename(path).split("_")[5]
    return datetime.datetime.strptime(timetag, "%Y%m%dT%H%M%S").replace(
        tzinfo=datetime.timezone.utc
    )


def sort_rasters(rasters):
    by_date = sorted(rasters, key=name_to_date)
    copy = list(range(len(rasters)))
    midpoint = len(rasters) // 2
    copy[midpoint] = by_date[0]

    add = 1
    left = True
    for idx, raster in enumerate(by_date):
        if idx == 0:
            continue

        if left:
            copy[midpoint - add] = raster
            left = False
        else:
            copy[midpoint + add] = raster
            left = True

            add += 1

    return copy


def process_aligned(
    aligned_rasters, out_path, folder_tmp, chunks, master_raster, nodata_value
):
    kernel_size = 3

    _kernel, offsets, weights = create_kernel(
        (kernel_size, kernel_size, len(aligned_rasters)),
        distance_calc="gaussian",
        sigma=1,
        spherical=True,
        radius_method="ellipsoid",
        offsets=True,
        edge_weights=True,
        normalised=True,
        remove_zero_weights=True,
    )

    arr_aligned = raster_to_array(aligned_rasters)
    arr_aligned.mask = arr_aligned == nodata_value

    if not rasters_are_aligned(aligned_rasters):
        raise Exception("Rasters not aligned")

    if chunks > 1:
        chunks_list = []
        print("Chunking rasters")

        for chunk in range(chunks):
            print(f"Chunk {chunk + 1} of {chunks}")
            if chunk == 0:
                chunk_start = 0
            else:
                chunk_start = chunk * arr_aligned.shape[0] // chunks

            if chunk == chunks - 1:
                chunk_end = arr_aligned.shape[0]
            else:
                chunk_end = (chunk + 1) * arr_aligned.shape[0] // chunks

            arr_chunk = arr_aligned[chunk_start:chunk_end]

            arr_collapsed = s1_collapse(
                arr_chunk,
                offsets,
                weights,
                weighted=True,
                nodata_value=nodata_value,
                nodata=True,
            )

            chunk_path = folder_tmp + f"{uuid4()}_chunk_{chunk}.npy"
            chunks_list.append(chunk_path)

            np.save(chunk_path, arr_collapsed)

            arr_chunk = None
            arr_collapsed = None

        print("Merging Chunks")
        arr_aligned = None

        merged = []
        for chunk in chunks_list:
            merged.append(np.load(chunk))

        merged = np.concatenate(merged)
        merged = np.ma.masked_array(merged, mask=merged == nodata_value)
        merged.fill_value = nodata_value

        print("Writing raster.")
        array_to_raster(
            merged,
            master_raster,
            out_path=out_path,
        )

        merged = None
        return out_path

    else:

        print("Collapsing rasters")
        arr_collapsed = s1_collapse(
            arr_aligned,
            offsets,
            weights,
            weighted=True,
            nodata_value=nodata_value,
            nodata=True,
        )

        arr_collapsed = np.ma.masked_array(
            arr_collapsed, mask=arr_collapsed == nodata_value
        )
        arr_collapsed.fill_value = nodata_value

        arr_aligned = None

        print("Writing raster.")
        array_to_raster(
            arr_collapsed,
            master_raster,
            out_path=out_path,
        )

        arr_collapsed = None


def mosaic_s1(
    vv_paths,
    vh_paths,
    folder_out,
    folder_tmp,
    master_raster,
    nodata_value=-9999.0,
    chunks=1,
):
    preprocessed = vv_paths + vh_paths

    clipped_vv = []
    clipped_vh = []
    for idx, img in enumerate(preprocessed):
        progress(idx, len(preprocessed), "Clipping Rasters")
        name = os.path.splitext(os.path.basename(img))[0] + "_clipped.tif"
        reprojected = reproject_raster(
            img,
            master_raster,
            copy_if_already_correct=False,
        )

        if not rasters_intersect(reprojected, master_raster):
            print("")
            print(f"{img} does not intersect {master_raster}, continuing\n")
            progress(idx + 1, len(preprocessed), "Clipping Rasters")
            gdal.Unlink(reprojected)
            continue

        clipped_raster = clip_raster(
            reprojected,
            master_raster,
            out_path=folder_tmp + name,
            postfix="",
            adjust_bbox=True,
            all_touch=False,
        )
        if "Gamma0_VV" in name:
            clipped_vv.append(clipped_raster)
        else:
            clipped_vh.append(clipped_raster)

        progress(idx + 1, len(preprocessed), "Clipping Rasters")
        gdal.Unlink(reprojected)

    outpath_vv = None
    outpath_vh = None

    if len(clipped_vv) > 0:
        print("Aligning VV rasters to master")
        arr_vv_aligned_rasters = align_rasters(
            clipped_vv,
            folder_tmp,
            postfix="_aligned",
            master=master_raster,
        )

        print("Processing VV")
        outpath_vv = process_aligned(
            arr_vv_aligned_rasters,
            folder_out + "VV_10m.tif",
            folder_tmp,
            chunks,
            master_raster,
            nodata_value,
        )

    if len(clipped_vh) > 0:
        print("Aligning VH rasters to master")
        arr_vh_aligned_rasters = align_rasters(
            clipped_vh,
            folder_tmp,
            postfix="_aligned",
            master=master_raster,
        )

        print("Processing VH")
        outpath_vh = process_aligned(
            arr_vh_aligned_rasters,
            folder_out + "VH_10m.tif",
            folder_tmp,
            chunks,
            master_raster,
            nodata_value,
        )

    return (outpath_vv, outpath_vh)


s2_mosaic_b04 = "C:/Users/caspe/Desktop/test_area/S2_mosaic/B04_10m.tif"
folder = "C:/Users/caspe/Desktop/test_area/tmp2/"
vv_paths = sort_rasters(glob(folder + "*Gamma0_VV*.tif"))
vh_paths = sort_rasters(glob(folder + "*Gamma0_VH*.tif"))

out_dir = folder + "out/"
tmp_dir = folder + "tmp/"

mosaic_s1(vv_paths, vh_paths, out_dir, tmp_dir, s2_mosaic_b04, chunks=2)
