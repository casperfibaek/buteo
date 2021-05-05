import sys

from numpy.lib.function_base import quantile

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
from buteo.raster.clip import internal_clip_raster
from buteo.raster.align import rasters_are_aligned, align_rasters
from buteo.filters.kernel_generator import create_kernel
from buteo.filters.filter import filter_array
from osgeo import gdal, osr
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


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
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
    border = True

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

                if border == True and outside == True:
                    normalise = True
                    hood_weights[n] = 0
                elif nodata and value == nodata_value:
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


# phase_cross_correlatio
# https://github.com/scikit-image/scikit-image/blob/main/skimage/registration/_phase_cross_correlation.py#L109-L276
def mosaic_sentinel1(
    folder,
    output_folder,
    tmp_folder,
    interest_area=None,
    target_projection=None,
    step_size=1.0,
    polarization="VV",
    epsilon: float = 1e-9,
):
    metadatas = []
    images = glob(folder + f"*Gamma0_{polarization}.tif")
    used_projections = []
    extent = []

    for idx, image in enumerate(images):
        metadata = internal_raster_to_metadata(image, create_geometry=True)
        used_projections.append(metadata["projection"])

        x_min, x_max, y_min, y_max = metadata["extent_ogr_latlng"]

        if idx == 0:
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

        metadatas.append(metadata)

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

    tile_extents = []
    bottom_left = coord_grid.tolist()
    for coord in bottom_left:
        tile_extents.append(
            [
                coord[0],
                coord[0] + step_size,
                coord[1],
                coord[1] + step_size,
            ]
        )

    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)

    tile_nr = 0
    for tile_extent in tile_extents:
        overlapping_images = []
        clipped_images = []
        for meta in metadatas:
            image_extent = meta["extent_ogr_latlng"]
            if ogr_bbox_intersects(tile_extent, image_extent):
                overlapping_images.append(meta)

                tile_img_path = tmp_folder + meta["name"] + ".tif"

                gdal.Translate(
                    tile_img_path,
                    meta["path"],
                    projWin=[
                        tile_extent[0],
                        tile_extent[3],
                        tile_extent[1],
                        tile_extent[2],
                    ],
                    projWinSRS=wgs84,
                    outputSRS=target_projection,
                )

                clipped_images.append(tile_img_path)

        if not rasters_are_aligned(clipped_images):
            clipped_images = align_rasters(
                clipped_images,
                tmp_folder,
                projection=use_projection,
                target_size=[10.0, 10.0],
            )

        clipped_images.sort(reverse=False, key=name_to_date)

        image_count = len(clipped_images)

        if (image_count % 2) == 0:
            clipped_images.append(clipped_images[0])

            image_count += 1

        merge_images = raster_to_array(clipped_images).filled(0)

        _kernel, offsets, weights = create_kernel(
            (image_count, 7, 7),
            sigma=2,
            distance_calc="gaussian",
            radius_method="ellipsoid",
            offsets=True,
            spherical=True,
            edge_weights=True,
            normalised=True,
            remove_zero_weights=True,
        )

        image = s1_collapse(
            merge_images,
            offsets,
            weights,
            quantile=0.333,
            nodata=True,
            nodata_value=0,
        )

        array_to_raster(
            image,
            reference=clipped_images[0],
            out_path=output_folder + f"tile_{tile_nr}.tif",
        )

        import pdb

        pdb.set_trace()

    # 1.  save all metadatas from images
    # 2.  find sentinel 2 tiles that intersect interest area
    # 3.  for tile in sentinel2_tiles:
    #         test does extent intersect tile
    #         clip to tile and reproject to tmp_folder

    #         base = the most central date and overlap > 50

    #         coregister images to base

    #         order images by time distance to base

    #         3D elipsodial weighted median filter

    #         save to output_folder

    # 4.  update and coregister all tiles
    # 5.  orfeo-toolbox mosaic tiles.


if __name__ == "__main__":
    data_folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
    folder = data_folder + "sentinel1/"
    merged = folder + "merged/"
    processed = folder + "mosaic_2020/"
    tmp = folder + "tmp/"

    mosaic_sentinel1(
        processed,
        merged,
        tmp,
        interest_area=data_folder + "denmark_polygon_border_region_removed.gpkg",
    )
