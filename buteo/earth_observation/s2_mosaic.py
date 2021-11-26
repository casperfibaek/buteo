import sys

from buteo.raster.reproject import reproject_raster

sys.path.append("../../")

import os
import numpy as np
from glob import glob
from time import time
import datetime
from buteo.raster.resample import internal_resample_raster
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.align import match_projections
from buteo.raster.clip import clip_raster
from buteo.earth_observation.s2_utils import (
    get_tile_files_from_safe,
    get_metadata,
)
from buteo.earth_observation.s2_quality_assessment import (
    assess_quality,
    smooth_mask,
    erode_mask,
    feather,
)
from buteo.gdal_utils import destroy_raster
from buteo.utils import timing
from buteo.orfeo_toolbox import merge_rasters


def resample_array(arr, original_reference, target_reference, resample_alg="average"):
    return raster_to_array(
        internal_resample_raster(
            array_to_raster(
                arr,
                reference=original_reference,
            ),
            target_size=target_reference,
            resample_alg=resample_alg,
        ),
        filled=True,
        output_2d=True,
    )


def harmonise_band(
    slave_arr,
    metadata,
    size,
    name,
    master_arr,
    master_quality,
    max_harmony=50,
    quality_to_include=75,
    method="mean_std_match",
):
    slave_quality = resample_array(
        metadata["quality"],
        metadata["paths"]["20m"]["SCL"],
        metadata["paths"]["60m"]["B04"],
    )
    slave_raster = internal_resample_raster(
        metadata["paths"][size][name], target_size=metadata["paths"]["60m"]["B04"]
    )
    slave_arr_60 = raster_to_array(slave_raster, output_2d=True, filled=True)
    destroy_raster(slave_raster)

    overlap = np.logical_and(
        master_quality > quality_to_include, slave_quality > quality_to_include
    )

    if overlap.sum() < 100:
        overlap = slave_quality > quality_to_include
    if overlap.sum() < 100:
        overlap = np.ones_like(overlap)

    slave_arr_60 = slave_arr_60[overlap]
    slave_quality_60 = slave_quality[overlap]

    master_arr_60 = master_arr[overlap]
    master_quality_60 = master_quality[overlap]

    if method == "mean_std_match":
        slave_med = np.ma.average(slave_arr_60, weights=slave_quality_60)
        slave_std = np.ma.sqrt(
            np.ma.average((slave_arr_60 - slave_med) ** 2, weights=slave_quality_60)
        )

        master_med = np.ma.average(master_arr_60, weights=master_quality_60)
        master_std = np.ma.sqrt(
            np.ma.average((master_arr_60 - master_med) ** 2, weights=master_quality_60)
        )
    else:
        slave_med = np.ma.median(slave_arr_60)
        slave_absdev = np.ma.abs(np.ma.subtract(slave_arr_60, slave_med))
        slave_std = np.ma.multiply(np.ma.median(slave_absdev), 1.4826)

        master_med = np.ma.median(master_arr_60)
        master_absdev = np.ma.abs(np.ma.subtract(master_arr_60, master_med))
        master_std = np.ma.multiply(np.ma.median(master_absdev), 1.4826)

    with np.errstate(divide="ignore", invalid="ignore"):
        harmony = master_med + (slave_arr - slave_med) * (master_std / slave_std)

    if max_harmony != 0:
        negative_limit = slave_arr * (1 - (max_harmony / 100))
        positive_limit = slave_arr * (1 + (max_harmony / 100))
    else:
        negative_limit = np.zeros_like(slave_arr)
        positive_limit = np.full_like(slave_arr, 65534.0)

    negative_limit = np.where(negative_limit < 0, 0, negative_limit)
    positive_limit = np.where(positive_limit > 65534.0, 65534.0, positive_limit)

    ret_arr = np.where(
        harmony < negative_limit,
        negative_limit,
        np.where(harmony > positive_limit, positive_limit, harmony),
    )

    return ret_arr


def mosaic_tile_s2(
    folder,
    tile_name,
    out_folder,
    max_time_delta=60.0,
    time_penalty=7,
    quality_threshold=105,
    quality_to_update=5,
    min_improvement=0.5,
    feather_dist=7,
    ideal_date=None,
    use_image=None,
    harmonise=True,
    max_harmony=50,
    max_images=6,
    output_scl=False,
    output_tracking=False,
    output_quality=False,
    process_bands=None,
):
    start = time()

    tiles = get_tile_files_from_safe(folder, tile_name)

    print(f"Finding best image for tile: {tile_name}, {len(tiles)} candidates.")

    metadatas = []
    best_score = 0
    best_idx = 0
    best_date = None
    best_time = 9999999999.9

    for index, tile in enumerate(tiles):
        metadata = get_metadata(tile)
        quality = assess_quality(tile)
        tile_score = np.average(quality)
        metadata["quality_score"] = tile_score

        if ideal_date is not None or use_image is not None:
            comp_time = None
            if use_image is not None:
                comp_time = use_image
            else:
                comp_time = ideal_date

            time_delta = abs(
                (
                    metadata["PRODUCT_STOP_TIME"]
                    - datetime.datetime.strptime(comp_time, "%Y%m%d").replace(
                        tzinfo=datetime.timezone.utc
                    )
                ).total_seconds()
                / 86400
            )

        quality_adjustment = 1
        if ideal_date is not None:
            # 1 % reduction in quality for every x days.
            quality_adjustment = (100 - time_delta / time_penalty) / 100

        if use_image:
            if time_delta < best_time:
                best_score = tile_score
                best_idx = index
                best_date = metadata["PRODUCT_STOP_TIME"]
                best_time = time_delta
        elif (tile_score * quality_adjustment) > best_score:
            best_score = tile_score
            best_idx = index
            best_date = metadata["PRODUCT_STOP_TIME"]

        metadata["quality"] = quality
        metadatas.append(metadata)

    print(
        f"Best image found: {os.path.basename(tiles[best_idx])} @ {round(best_score, 3)}"
    )

    metadatas_thresholded = []
    print("Adjusting scores by temporal distance.")
    for index, tile in enumerate(tiles):
        metadata = metadatas[index]
        recording_time = metadata["PRODUCT_STOP_TIME"]

        time_delta = abs((best_date - recording_time).total_seconds() / 86400)

        # 1 % reduction in quality for every x days.
        quality_adjustment = (100 - time_delta / time_penalty) / 100

        metadatas[index]["quality"] = np.rint(
            (metadatas[index]["quality"].astype("float32") * quality_adjustment)
        ).astype("uint8")
        metadatas[index]["quality_score"] = np.average(metadatas[index]["quality"])

        if time_delta < max_time_delta:
            metadatas_thresholded.append(metadatas[index])

    # Ordering metadatas by quality
    metadatas = metadatas_thresholded
    metadatas = sorted(metadatas, key=lambda i: i["quality_score"], reverse=True)

    # if central_images, move it to the front
    if use_image is not None:
        for index, meta in enumerate(metadatas):
            if meta["PRODUCT_STOP_TIME"] == best_date:
                metadatas.insert(0, metadatas.pop(index))
                break

    scl_array = raster_to_array(
        metadatas[0]["paths"]["20m"]["SCL"], filled=True, output_2d=True
    )
    tracking_array = np.zeros_like(scl_array, dtype="uint8")

    current_valid_mask = erode_mask(scl_array != 0, feather_dist)
    current_quality = metadatas[0]["quality"] * current_valid_mask
    current_quality_score = np.average(current_quality)

    print("Tracking..")
    print(f"Current quality: {round(current_quality_score, 4)}")
    used_images = [0]
    tested_images = 1
    while (
        current_quality_score < quality_threshold
        and len(used_images) < max_images
        and tested_images < len(metadatas)
    ):

        best_idx = None
        best_improvement = None
        best_valid_sum = None
        best_tile_quality = None
        best_tile_scl = None
        best_improvement_mask = None

        first = True
        for index, metadata in enumerate(metadatas):
            if index in used_images:
                continue

            tile_quality = metadata["quality"]
            tile_scl = raster_to_array(
                metadata["paths"]["20m"]["SCL"], filled=True, output_2d=True
            )

            valid_mask = erode_mask(tile_scl != 0, feather_dist)

            improvement_mask = valid_mask & smooth_mask(
                tile_quality > (current_quality * (1 + (quality_to_update / 100)))
            )
            improvement_percent = np.average(improvement_mask) * 100

            if first:
                best_idx = index
                best_improvement = improvement_percent
                best_valid_sum = valid_mask.sum()
                best_tile_quality = tile_quality
                best_tile_scl = tile_scl
                best_improvement_mask = improvement_mask
                first = False
            else:
                if improvement_percent > best_improvement:
                    best_idx = index
                    best_improvement = improvement_percent
                    best_valid_sum = valid_mask.sum()
                    best_tile_quality = tile_quality
                    best_tile_scl = tile_scl
                    best_improvement_mask = improvement_mask

        tile_quality = np.where(
            best_improvement_mask, best_tile_quality, current_quality
        )
        tile_quality_score = np.average(tile_quality)

        if (tile_quality_score - current_quality_score) < min_improvement and (
            best_valid_sum <= current_valid_mask.sum()
        ):
            break

        # Update tracking arrays
        tracking_array = np.where(best_improvement_mask, best_idx, tracking_array)
        scl_array = np.where(best_improvement_mask, best_tile_scl, scl_array)
        current_quality = tile_quality
        current_quality_score = tile_quality_score
        used_images.append(best_idx)
        tested_images += 1

        print(f"Current quality: {round(current_quality_score, 4)}")

    bands_to_process = (
        [
            {"size": "10m", "band": "B02"},
            {"size": "10m", "band": "B03"},
            {"size": "10m", "band": "B04"},
            {"size": "20m", "band": "B05"},
            {"size": "20m", "band": "B06"},
            {"size": "20m", "band": "B07"},
            {"size": "20m", "band": "B8A"},
            {"size": "10m", "band": "B08"},
            {"size": "20m", "band": "B11"},
            {"size": "20m", "band": "B12"},
        ]
        if process_bands is None
        else process_bands
    )

    if len(used_images) > 1:
        print("Pre-calculating feathers")
        tracking_20m = feather(
            tracking_array, np.array(used_images, dtype="uint8"), feather_dist
        )

        tracking_10m = raster_to_array(
            internal_resample_raster(
                array_to_raster(
                    tracking_20m, reference=metadatas[0]["paths"]["20m"]["SCL"]
                ),
                target_size=metadatas[0]["paths"]["10m"]["B04"],
                resample_alg="average",
            ),
            filled=True,
        )

        tracking_60m = raster_to_array(
            internal_resample_raster(
                array_to_raster(
                    tracking_20m, reference=metadatas[0]["paths"]["20m"]["SCL"]
                ),
                target_size=metadatas[0]["paths"]["60m"]["B04"],
                resample_alg="average",
            ),
            filled=True,
        )

    created_images = []
    print("Harmonising and merging tiles")
    for pi, process_band in enumerate(bands_to_process):
        size = process_band["size"]
        name = process_band["band"]
        tile_outname = out_folder + f"{tile_name}_{name}_{size}.tif"

        print(f"Now processing {pi + 1}/{len(bands_to_process)}: {name} @ {size}")

        out_arr = None

        master_arr = None
        master_quality = None
        for index, image in enumerate(used_images):
            metadata = metadatas[image]

            band_arr = raster_to_array(
                metadata["paths"][size][name], filled=True, output_2d=True
            )

            if len(used_images) == 1:
                out_arr = band_arr
                continue

            if harmonise and index == 0:
                master_arr = resample_array(
                    band_arr,
                    metadata["paths"][size]["B04"],
                    metadata["paths"]["60m"]["B04"],
                )
                master_quality = resample_array(
                    metadata["quality"],
                    metadata["paths"]["20m"]["SCL"],
                    metadata["paths"]["60m"]["B04"],
                )

            if harmonise and index != 0:

                band_arr = harmonise_band(
                    band_arr,
                    metadata,
                    size,
                    name,
                    master_arr,
                    master_quality,
                    max_harmony=max_harmony,
                )

            feather_scale = None
            if size == "10m":
                feather_scale = tracking_10m[:, :, index]
            elif size == "20m":
                feather_scale = tracking_20m[:, :, index]
            elif size == "60m":
                feather_scale = tracking_60m[:, :, index]
            else:
                raise Exception("Unknown band size.")

            if index == 0:
                out_arr = feather_scale * band_arr
            else:
                out_arr += feather_scale * band_arr

        ref = None
        if size == "10m":
            ref = metadatas[0]["paths"]["10m"]["B04"]
        elif size == "20m":
            ref = metadatas[0]["paths"]["20m"]["B04"]
        elif size == "60m":
            ref = metadatas[0]["paths"]["60m"]["B04"]

        created_images.append(tile_outname)

        array_to_raster(
            np.rint(out_arr).astype("uint16"),
            reference=ref,
            out_path=tile_outname,
        )

    if output_tracking:
        tracking_outname = out_folder + f"{tile_name}_tracking_20m.tif"
        created_images.append(tracking_outname)
        array_to_raster(
            tracking_array,
            reference=metadatas[0]["paths"]["20m"]["SCL"],
            out_path=tracking_outname,
        )

    if output_scl:
        scl_outname = out_folder + f"{tile_name}_SCL_20m.tif"
        created_images.append(scl_outname)
        array_to_raster(
            scl_array,
            reference=metadatas[0]["paths"]["20m"]["SCL"],
            out_path=scl_outname,
        )

    if output_quality:
        quality_outname = out_folder + f"{tile_name}_quality_20m.tif"
        created_images.append(quality_outname)
        array_to_raster(
            current_quality,
            reference=metadatas[0]["paths"]["20m"]["SCL"],
            out_path=quality_outname,
        )

    timing(start)

    return created_images


def join_s2_tiles(
    mosaic_tile_folder,
    out_folder,
    tmp_dir,
    prefix="",
    clip_geom=None,
    harmonisation=False,
    nodata_value=0.0,
    pixel_width=None,
    pixel_height=None,
    bands_to_process=None,
    projection_to_match=25832,
    clean=False,
):
    bands = (
        [
            "B02_10m",
            "B03_10m",
            "B04_10m",
            "B08_10m",
            "B05_20m",
            "B06_20m",
            "B07_20m",
            "B8A_20m",
            "B11_20m",
            "B12_20m",
        ]
        if bands_to_process is None
        else bands_to_process
    )

    created = []
    for band in bands:
        images = glob(mosaic_tile_folder + f"*_{band}.tif")

        # reprojected = match_projections(
        #     images, projection_to_match, tmp_dir, dst_nodata=nodata_value
        # )

        reprojected = reproject_raster(
            images,
            projection_to_match,
            tmp_dir,
            copy_if_already_correct=False,
            dst_nodata=nodata_value,
        )

        if clip_geom is not None:
            reprojected = clip_raster(
                reprojected, clip_geom, out_path=tmp_dir, crop_to_geom=False
            )

        output = merge_rasters(
            reprojected,
            out_folder + prefix + band + ".tif",
            tmp=tmp_dir,
            harmonisation=harmonisation,
            nodata_value=nodata_value,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
        )

        created.append(output)

    if clean:
        tmp_files = glob(tmp_dir + "*.tif")
        for f in tmp_files:
            try:
                os.remove(f)
            except:
                pass

    return created
