import sys

sys.path.append("..")
sys.path.append("../../")

from glob import glob
from numba import jit, prange
from buteo.raster.io import (
    internal_raster_to_metadata,
    raster_to_array,
    array_to_raster,
)
from buteo.vector.io import internal_vector_to_metadata
from buteo.gdal_utils import parse_projection, ogr_bbox_intersects
from buteo.raster.align import rasters_are_aligned, align_rasters
from buteo.filters.kernel_generator import create_kernel
from buteo.utils import timing
from osgeo import gdal, osr
from time import time
import numpy as np
import os
import datetime


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
    nodata=False,
    nodata_value=0,
):
    x_adj = arr.shape[0] - 1
    y_adj = arr.shape[1] - 1
    z_adj = (arr.shape[2] - 1) // 2

    hood_size = len(offsets)
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

                if outside or (nodata and nodata_value == value):
                    normalise = True
                    hood_weights[n] = 0
                else:
                    hood_values[n] = value
                    weight = weights[n]

                    hood_weights[n] = weight
                    weight_sum[0] += weight

            if normalise:
                hood_weights = np.divide(hood_weights, weight_sum[0])

            result[x, y] = hood_quantile(hood_values, hood_weights, quantile)

    return result


def name_to_date(path):
    timetag = os.path.basename(path).split("_")[5]
    return datetime.datetime.strptime(timetag, "%Y%m%dT%H%M%S").replace(
        tzinfo=datetime.timezone.utc
    )


def mosaic_sentinel1(
    folder,
    output_folder,
    tmp_folder,
    interest_area=None,
    use_tiles=True,
    target_projection=None,
    step_size=1.0,
    polarization="VV",
    epsilon: float = 1e-9,
    overlap=0.05,
    target_size=[10.0, 10.0],
    kernel_size=3,
    max_images=0,
    quantile=0.5,
    overwrite=False,
    prefix="",
    postfix="",
    high_memory=True,
):
    start = time()

    metadatas = []
    images = glob(folder + f"*Gamma0_{polarization}.tif")
    used_projections = []
    used_metadatas = []
    extent = []

    for image in images:
        metadata = internal_raster_to_metadata(image, create_geometry=True)
        metadatas.append(metadata)

    if interest_area is not None:
        tile_extent = internal_vector_to_metadata(interest_area)["extent_ogr_latlng"]

    for idx, image in enumerate(images):
        metadata = metadatas[idx]

        if interest_area is not None:
            if not ogr_bbox_intersects(tile_extent, metadata["extent_ogr_latlng"]):
                continue

        x_min, x_max, y_min, y_max = metadata["extent_ogr_latlng"]

        if len(extent) == 0:
            extent = metadata["extent_ogr_latlng"]
        else:
            if x_min < extent[0]:
                extent[0] = x_min
            if x_max > extent[1]:
                extent[1] = x_max
            if y_min < extent[2]:
                extent[2] = y_min
            if y_max > extent[3]:
                extent[3] = y_max

        used_metadatas.append(metadata)
        used_projections.append(metadata["projection"])

    use_projection = None
    if target_projection is None:
        projection_counter: dict = {}
        for proj in used_projections:
            if proj in projection_counter:
                projection_counter[proj] += 1
            else:
                projection_counter[proj] = 1

        # Choose most common projection
        most_common_projection = sorted(
            projection_counter, key=projection_counter.__getitem__, reverse=True
        )
        use_projection = parse_projection(most_common_projection[0])
    else:
        use_projection = parse_projection(target_projection)

    tile_extents = []

    tiles = 1
    if not use_tiles and interest_area is not None:
        tile_extents.append(
            [
                tile_extent[0] - overlap,
                tile_extent[1] + overlap,
                tile_extent[2] - overlap,
                tile_extent[3] + overlap,
            ]
        )
    elif not use_tiles:
        tile_extents.append(extent)
    else:
        use_area = None
        if interest_area is None:
            use_area = extent
        else:
            interest_area_metadata = internal_vector_to_metadata(interest_area)
            use_area = interest_area_metadata["extent_ogr_latlng"]

            use_area[0] -= use_area[0] % step_size
            use_area[1] += 1 - (use_area[1] % step_size)

            use_area[2] -= use_area[2] % step_size
            use_area[3] += 1 - (use_area[3] % step_size)

        x_size = round((use_area[1] - use_area[0]) / step_size)
        y_size = round((use_area[3] - use_area[2]) / step_size)

        xr = np.arange(use_area[0], use_area[1] + epsilon, step_size)
        yr = np.arange(use_area[2], use_area[3] + epsilon, step_size)

        tiles = int(x_size * y_size)
        coord_grid = np.empty((tiles, 2), dtype="float64")

        oxx, oyy = np.meshgrid(xr[:-1], yr[:-1])
        oxr = oxx.ravel()
        oyr = oyy.ravel()

        coord_grid[:, 0] = oxr
        coord_grid[:, 1] = oyr

        bottom_left = coord_grid.tolist()
        for coord in bottom_left:
            tile_extents.append(
                [
                    coord[0] - overlap,
                    coord[0] + step_size + overlap,
                    coord[1] - overlap,
                    coord[1] + step_size + overlap,
                ]
            )

    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)

    if high_memory:
        driver = gdal.GetDriverByName("GTiff")

    tile_nr = 0
    created_tiles = []
    for tile_extent in tile_extents:
        tile_path = output_folder + f"{prefix}{polarization}_{tile_nr}{postfix}.tif"
        print(f"Started: {tile_path}")

        if not overwrite and os.path.exists(tile_path):
            tile_nr += 1
            created_tiles.append(tile_path)
            print(f"Created: {tile_nr}/{tiles}")
            continue

        overlapping_images = []
        clipped_images_holder = []
        clipped_images = []
        for meta in used_metadatas:
            image_extent = meta["extent_ogr_latlng"]
            if ogr_bbox_intersects(tile_extent, image_extent):
                overlapping_images.append(meta)

                if high_memory:
                    tile_img_path = "/vsimem/" + meta["name"] + ".tif"
                else:
                    tile_img_path = tmp_folder + meta["name"] + ".tif"

                if not overwrite and os.path.exists(tile_img_path):
                    clipped_images.append(tile_img_path)
                    continue

                clip_bounds = [
                    tile_extent[0],
                    tile_extent[2],
                    tile_extent[1],
                    tile_extent[3],
                ]

                warped = gdal.Warp(
                    tile_img_path,
                    meta["path"],
                    outputBounds=clip_bounds,
                    outputBoundsSRS=wgs84,
                    targetAlignedPixels=True,
                    xRes=target_size[0],
                    yRes=target_size[1],
                    dstSRS=use_projection,
                    srcSRS=meta["projection_osr"],
                    multithread=True,
                )

                if warped == None:
                    raise Exception("Error while warping..")

                clipped_images.append(tile_img_path)

        if len(clipped_images) == 0:
            tile_nr += 1
            created_tiles.append(tile_path)
            print(f"Created: {tile_nr}/{tiles}")
            continue

        if not rasters_are_aligned(clipped_images):
            target_folder = None if high_memory else tmp_folder
            clipped_images = align_rasters(
                clipped_images,
                target_folder,
                target_size=target_size,
                projection=use_projection,
                bounding_box="union",
            )

        if max_images != 0:
            tmp_clipped_images = []
            use_idx = []
            added_images = 0
            used_images = []

            for idx, image in enumerate(clipped_images):
                valid_sum = (raster_to_array(image, filled=True, output_2d=True) != 0).sum()

                if not expand_if_missing:
                    valid_mask = None

                use_idx.append(
                    {"idx": idx, "valid": valid_sum, "image": image}
                )

            use_idx = sorted(use_idx, key=lambda i: i["valid"], reverse=True)

            combined_mask = None
            combined_sum = 0
            for nr in range(max_images):
                try:
                    ordered_image = use_idx[nr]["image"]
                    tmp_clipped_images.append(ordered_image)

                    added_images += 1
                except:
                    break

            for img in clipped_images:
                if img not in tmp_clipped_images:
                    driver.Delete(img)

            clipped_images = tmp_clipped_images
            use_idx = None

        clipped_images.sort(reverse=False, key=name_to_date)

        image_count = len(clipped_images)

        merge_images = raster_to_array(clipped_images).filled(0)

        if high_memory:
            for idx, img in enumerate(clipped_images):
                if idx > 0:
                    driver.Delete(img)

        _kernel, offsets, weights = create_kernel(
            (kernel_size, kernel_size, image_count),
            distance_calc=False,
            spherical=True,
            radius_method="ellipsoid",
            offsets=True,
            edge_weights=True,
            normalised=True,
            remove_zero_weights=True,
        )

        print("Merging tiles.")
        image = s1_collapse(
            merge_images,
            offsets,
            weights,
            quantile=quantile,
            nodata=True,
            nodata_value=0,
        )

        merge_images = None

        array_to_raster(
            image,
            reference=clipped_images[0],
            out_path=tile_path,
        )

        if high_memory:
            driver.Delete(clipped_images[0])

        image = None

        created_tiles.append(tile_path)

        if not high_memory:
            tmp_files = glob(tmp_folder + "*.tif")
            for f in tmp_files:
                try:
                    os.remove(f)
                except:
                    pass

        tile_nr += 1

        print(f"Created: {tile_nr}/{tiles} - {polarization}, {prefix[:-1]}")
        timing(start)

    return created_tiles


if __name__ == "__main__":
    # data_folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
    folder = "/home/cfi/Desktop/sentinel1/"
    # folder = data_folder + "sentinel1/"
    tiles = folder + "tiles/"
    processed = folder + "mosaic_2021/"
    tmp = folder + "tmp/"

    mosaic_sentinel1(
        processed,
        tiles,
        tmp,
        interest_area=folder + "ghana_buffered_1280.gpkg",
        target_projection=folder + "ghana_buffered_1280.gpkg",
        kernel_size=3,
        overlap=0.05,
        step_size=1.0,
        quantile=0.5,
        max_images=16,
        overwrite=False,
        use_tiles=True,
        high_memory=True,
        polarization="VH",
        prefix="2021_",
    )

    mosaic_sentinel1(
        processed,
        tiles,
        tmp,
        interest_area=folder + "ghana_buffered_1280.gpkg",
        target_projection=folder + "ghana_buffered_1280.gpkg",
        kernel_size=3,
        overlap=0.05,
        step_size=1.0,
        quantile=0.5,
        max_images=16,
        overwrite=False,
        use_tiles=True,
        high_memory=True,
        polarization="VV",
        prefix="2021_",
    )
