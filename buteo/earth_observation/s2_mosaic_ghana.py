import sys

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
    get_tile_files_from_safe_zip,
    unzip_files_to_folder,
)
from buteo.earth_observation.s2_quality_assessment import (
    assess_quality,
    weighted_quantile_2d,
    weighted_std,
    smooth_mask,
    erode_mask,
    feather,
)
from buteo.utils import timing
from buteo.orfeo_toolbox import merge_rasters


def mosaic_tile(
    folder,
    tile_name,
    out_folder,
    max_time_delta=60.0,
    time_penalty=7,
    quality_threshold=105,
    quality_to_update=5,
    min_improvement=1,
    feather_dist=15,
    ideal_date=None,
    use_image=None,
    harmonise=True,
    harmony_type="median",
    max_harmony=10,
    max_images=10,
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
    for index, metadata in enumerate(metadatas):
        if index == 0 or current_quality_score > quality_threshold:
            continue

        if max_images != 0 and index >= max_images:
            break

        tile_quality = metadata["quality"]
        tile_scl = raster_to_array(
            metadatas[index]["paths"]["20m"]["SCL"], filled=True, output_2d=True
        )

        valid_mask = erode_mask(tile_scl != 0, feather_dist)

        improvement_mask = valid_mask & smooth_mask(
            tile_quality > (current_quality * (1 + (quality_to_update / 100)))
        )
        improvement_percent = np.average(improvement_mask) * 100

        # merged_valid_mask = current_valid_mask | valid_mask

        # if improvement_percent < min_improvement and (merged_valid_mask.sum() <= current_valid_mask.sum()):
        if improvement_percent < min_improvement:
            continue

        # Update tracking arrays
        tracking_array = np.where(improvement_mask, index, tracking_array)
        scl_array = np.where(improvement_mask, tile_scl, scl_array)
        current_quality = np.where(improvement_mask, tile_quality, current_quality)
        current_quality_score = np.average(current_quality)
        used_images.append(index)

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

    target_harmony = {}
    created_images = []

    print("Harmonising and merging tiles")
    for pi, process_band in enumerate(bands_to_process):
        size = process_band["size"]
        name = process_band["band"]
        tile_outname = out_folder + f"{tile_name}_{name}_{size}.tif"

        print(f"Now processing {pi + 1}/{len(bands_to_process)}: {name} @ {size}")

        band_data = None
        for index, image in enumerate(used_images):
            metadata = metadatas[image]

            band_arr = raster_to_array(
                metadata["paths"][size][name], filled=True, output_2d=True
            )

            if len(used_images) == 1:
                array_to_raster(
                    band_arr,
                    reference=metadata["paths"][size][name],
                    out_path=tile_outname,
                )
                continue

            if harmonise:
                image_quality = raster_to_array(
                    internal_resample_raster(
                        array_to_raster(
                            metadata["quality"],
                            reference=metadatas[0]["paths"]["20m"]["SCL"],
                        ),
                        target_size=metadatas[0]["paths"]["60m"]["B04"],
                        resample_alg="average",
                    ),
                    filled=True,
                    output_2d=True,
                )
                image_quality_norm = image_quality / image_quality.max()
                weights = (image_quality_norm / image_quality_norm.sum()).astype(
                    "float32"
                )

            scale = None
            med_arr = None
            if size == "10m":
                scale = tracking_10m[:, :, index]
            elif size == "20m":
                scale = tracking_20m[:, :, index]
            elif size == "60m":
                scale = tracking_60m[:, :, index]
            else:
                raise Exception("Unknown band size.")

            if harmonise:

                # The infrared band is only available in 10m resolution.
                if name == "B08":
                    med_arr = raster_to_array(
                        internal_resample_raster(
                            metadata["paths"]["10m"][name],
                            target_size=metadatas[0]["paths"]["60m"]["B04"],
                            resample_alg="average",
                        ),
                        filled=True,
                        output_2d=True,
                    )
                else:
                    med_arr = raster_to_array(
                        metadata["paths"]["60m"][name], filled=True, output_2d=True
                    )

                if index != 0:
                    med_arr = med_arr * (
                        target_harmony[name]["gain"] / metadata["gains"][name]
                    )

                # find weighted_median and mad
                average = (
                    np.average(med_arr * image_quality_norm)
                    if harmony_type != "median"
                    else None
                )
                std = (
                    weighted_std(med_arr, weights) if harmony_type != "median" else None
                )
                median = (
                    weighted_quantile_2d(med_arr, weights, 0.5)
                    if harmony_type == "median"
                    else None
                )
                absdeviation = (
                    np.abs(np.subtract(med_arr, median))
                    if harmony_type == "median"
                    else None
                )
                madstd = (
                    weighted_quantile_2d(absdeviation, weights, 0.5) * 1.4826
                    if harmony_type == "median"
                    else None
                )

            if index == 0:
                if harmonise:
                    target_harmony[name] = {
                        "median": median,
                        "madstd": madstd,
                        "average": average,
                        "std": std,
                        "gain": metadata["gains"][name],
                    }

                readied_tile = scale * band_arr
                band_data = readied_tile
            else:
                if harmonise:
                    band_arr = band_arr * (
                        target_harmony[name]["gain"] / metadata["gains"][name]
                    )
                    if harmony_type == "median":
                        deviation = band_arr - median
                        with np.errstate(divide="ignore", invalid="ignore"):
                            harmony_scale = (
                                np.true_divide(
                                    deviation * target_harmony[name]["madstd"], madstd,
                                )
                            ) + target_harmony[name]["median"]
                    else:
                        deviation = band_arr - average
                        with np.errstate(divide="ignore", invalid="ignore"):
                            harmony_scale = (
                                np.true_divide(
                                    deviation * target_harmony[name]["std"], std
                                )
                            ) + target_harmony[name]["average"]

                    merged = band_arr + harmony_scale
                    negative_limit = band_arr * (1 - (max_harmony / 100))
                    positive_limit = band_arr * (1 + (max_harmony / 100))

                    readied_tile = (
                        np.where(
                            merged < negative_limit,
                            negative_limit,
                            np.where(merged > positive_limit, positive_limit, merged),
                        ).astype("uint16")
                        * scale
                    )
                else:
                    readied_tile = scale * band_arr

                band_data = band_data + readied_tile

            ref = None
            if size == "10m":
                ref = metadatas[0]["paths"]["10m"]["B04"]
            elif size == "20m":
                ref = metadatas[0]["paths"]["20m"]["B04"]
            elif size == "60m":
                ref = metadatas[0]["paths"]["60m"]["B04"]

            created_images.append(tile_outname)

            array_to_raster(
                np.rint(band_data).astype("uint16"),
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


def join_tiles(
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

        reprojected = match_projections(images, 25832, tmp_dir, dst_nodata=nodata_value)

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

    tmp_files = glob(tmp_dir + "*.tif")
    for f in tmp_files:
        os.remove(f)

    return created


# from matplotlib import pyplot as plt; plt.imshow(final); plt.show()
if __name__ == "__main__":
    from shutil import rmtree
    from buteo.vector.attributes import vector_get_attribute_table

    folder = "/home/cfi/Desktop/sentinel2/"

    tmp = folder + "tmp/"
    raw = folder + "raw/"
    dst = folder + "mosaic/"

    vector = folder + "ghana_s2_tiles.gpkg"

    attributes = vector_get_attribute_table(vector)
    tiles = attributes["Name"].values.tolist()

    completed = []

    all_tiles = [
        # '30NVL',
        # '30NVM',
        # '30NVN',
        # '30NWL',
        # '30NWM',
        # '30NWN',
        # '30NWP',
        # '30NXL',
        # '30NXM',
        # '30NXN',
        # '30NXP',
        # '30NYL',
        # '30NYM',
        # '30NYN',
        # '30NYP',
        # '30NZM',
        # '30NZN',
        # '30NZP',
        # '30PWQ',
        # '30PWR',
        # '30PWS',
        # '30PWT',
        # '30PXQ',
        # '30PXR',
        # '30PXS',
        # '30PXT',
        # '30PYQ',
        # '30PYR',
        # '30PYS',
        # '30PYT',
        # '30PZQ',
        # '30PZR',
        # '30PZS',
        # '30PZT',
        # '31NBG',
        # '31NBH',
        # '31NBJ',
        # '31PBK',
        # '31PBL',
        '31PBM',
        '31PBN',
    ]

    for tile in all_tiles:

        if tile in completed:
            continue

        try:
            unzipped = unzip_files_to_folder(
                get_tile_files_from_safe_zip(raw, tile), tmp,
            )

            mosaic_tile(
                tmp,
                tile,
                dst,
                min_improvement=0.1,
                quality_threshold=110,
                time_penalty=14,
                max_time_delta=90.0,
                max_images=10,
                # ideal_date="20210226",
                # use_image="20210226",
                process_bands=[
                    # {"size": "10m", "band": "B02"},
                    # {"size": "10m", "band": "B03"},
                    # {"size": "10m", "band": "B04"},
                    # {"size": "20m", "band": "B05"},
                #     {"size": "20m", "band": "B06"},
                #     {"size": "20m", "band": "B07"},
                #     {"size": "20m", "band": "B8A"},
                #     {"size": "10m", "band": "B08"},
                #     {"size": "20m", "band": "B11"},
                    {"size": "20m", "band": "B12"},
                ],
            )
        except:
            print(f"Error with tile: {tile}")
        finally:
            tmp_files = glob(tmp + "*.SAFE")
            for f in tmp_files:
                rmtree(f)
