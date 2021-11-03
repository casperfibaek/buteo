from logging import makeLogRecord
import sys
import os

sys.path.append("../../")
import numpy as np
from osgeo import ogr, gdal

np.set_printoptions(suppress=True)

import tensorflow as tf
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    is_raster,
    raster_to_metadata,
)
from buteo.raster.align import rasters_are_aligned, align_rasters
from buteo.raster.clip import clip_raster
from buteo.raster.resample import resample_raster
from buteo.vector.io import vector_to_metadata, is_vector, open_vector
from buteo.vector.clip import clip_vector
from buteo.vector.intersect import intersect_vector
from buteo.vector.attributes import vector_get_fids
from buteo.vector.rasterize import rasterize_vector
from buteo.machine_learning.ml_utils import tpe, get_offsets
from buteo.utils import progress
from uuid import uuid4


def array_to_blocks(arr: np.ndarray, tile_size, offset=[0, 0]) -> np.ndarray:
    blocks_y = (arr.shape[0] - offset[1]) // tile_size
    blocks_x = (arr.shape[1] - offset[0]) // tile_size

    cut_y = -((arr.shape[0] - offset[1]) % tile_size)
    cut_x = -((arr.shape[1] - offset[0]) % tile_size)

    cut_y = None if cut_y == 0 else cut_y
    cut_x = None if cut_x == 0 else cut_x

    reshaped = arr[offset[1] : cut_y, offset[0] : cut_x].reshape(
        blocks_y,
        tile_size,
        blocks_x,
        tile_size,
        arr.shape[2],
    )

    swaped = reshaped.swapaxes(1, 2)
    merge = swaped.reshape(-1, tile_size, tile_size, arr.shape[2])

    return merge


def blocks_to_array(blocks, og_shape, tile_size, offset=[0, 0]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        target = np.empty(og_shape) * np.nan

    target_y = ((og_shape[0] - offset[1]) // tile_size) * tile_size
    target_x = ((og_shape[1] - offset[0]) // tile_size) * tile_size

    cut_y = -((og_shape[0] - offset[1]) % tile_size)
    cut_x = -((og_shape[1] - offset[0]) % tile_size)

    cut_x = None if cut_x == 0 else cut_x
    cut_y = None if cut_y == 0 else cut_y

    reshape = blocks.reshape(
        target_y // tile_size,
        target_x // tile_size,
        tile_size,
        tile_size,
        blocks.shape[3],
        1,
    )

    swap = reshape.swapaxes(1, 2)

    destination = swap.reshape(
        (target_y // tile_size) * tile_size,
        (target_x // tile_size) * tile_size,
        blocks.shape[3],
    )

    target[offset[1] : cut_y, offset[0] : cut_x] = destination

    return target


def get_overlaps(arr, offsets, tile_size, border_check=True):
    arr_offsets = []

    for offset in offsets:
        arr_offsets.append(array_to_blocks(arr, tile_size, offset))

    if border_check:
        for end in ["right", "bottom", "corner"]:
            if end == "right" and (arr.shape[1] % tile_size) != 0:
                border = arr[:, -tile_size:]
            elif end == "bottom" and (arr.shape[0] % tile_size) != 0:
                border = arr[-tile_size:, :]
            elif (
                end == "corner"
                and ((arr.shape[1] % tile_size) != 0)
                and ((arr.shape[1] % tile_size) != 0)
            ):
                border = arr[-tile_size:, -tile_size:]

            arr_offsets.append(array_to_blocks(border, tile_size, offset=[0, 0]))

    return arr_offsets


def extract_patches(
    raster_list,
    outdir,
    tile_size=32,
    zones=None,
    options=None,
):
    """
    Generate patches for machine learning from rasters
    """
    if options is None:
        options = {}

    overlaps = True if "overlaps" not in options else options["overlaps"]
    border_check = True if "border_check" not in options else options["border_check"]
    merge_output = True if "merge_output" not in options else options["merge_output"]
    tolerance = 0.0 if "tolerance" not in options else options["tolerance"]
    label_geom = None if "label_geom" not in options else options["label_geom"]
    label_res = 0.2 if "label_res" not in options else options["label_res"]
    label_mult = 100 if "label_mult" not in options else options["label_mult"]
    zone_layer_id = 0 if "zone_layer_id" not in options else options["zone_layer_id"]
    fill_value = 0 if "fill_value" not in options else options["fill_value"]
    force_align = True if "force_align" not in options else options["force_align"]
    output_raster_labels = (
        True
        if "output_raster_labels" not in options
        else options["output_raster_labels"]
    )
    prefix = "" if "prefix" not in options else options["prefix"]
    postfix = "" if "postfix" not in options else options["postfix"]

    if zones is not None and not is_vector(zones):
        raise TypeError("Clip geom is invalid. Did you input a valid geometry?")

    if not isinstance(raster_list, list):
        raise TypeError("raster_list is not a list of rasters.")
    for raster in raster_list:
        if not is_raster(raster):
            raise TypeError("raster_list is not a list of rasters.")

    if not os.path.isdir(outdir):
        raise ValueError(
            "Outdir does not exist. Please create before running the function."
        )

    if not rasters_are_aligned(raster_list, same_extent=True):
        if force_align:
            print(
                "Rasters we not aligned. Realigning rasters due to force_align=True option."
            )
            raster_list = align_rasters(raster_list)
        else:
            raise ValueError("Rasters in raster_list are not aligned.")

    offsets = get_offsets(tile_size) if overlaps else [[0, 0]]
    raster_metadata = raster_to_metadata(raster_list[0], create_geometry=True)
    pixel_size = min(raster_metadata["pixel_height"], raster_metadata["pixel_width"])

    if zones is None:
        zones = raster_metadata["extent_datasource_path"]

    zones_meta = vector_to_metadata(zones)

    mem_driver = ogr.GetDriverByName("ESRI Shapefile")

    if zones_meta["layer_count"] == 0:
        raise ValueError("Vector contains no layers.")

    zones_layer_meta = zones_meta["layers"][zone_layer_id]

    if zones_layer_meta["geom_type"] not in ["Multi Polygon", "Polygon"]:
        raise ValueError("clip geom is not Polygon or Multi Polygon.")

    zones_ogr = open_vector(zones)
    zones_layer = zones_ogr.GetLayer(zone_layer_id)
    fids = vector_get_fids(zones_ogr, zone_layer_id)

    progress(0, len(fids) * len(raster_list), "processing fids")
    processed_fids = []
    processed = 0
    labels_processed = False

    for idx, raster in enumerate(raster_list):
        name = os.path.splitext(os.path.basename(raster))[0]
        list_extracted = []
        list_masks = []
        list_labels = []

        for fid in fids:
            feature = zones_layer.GetFeature(fid)
            fid_path = f"/vsimem/fid_mem_{uuid4().int}_{str(fid)}.shp"
            fid_ds = mem_driver.CreateDataSource(fid_path)
            fid_ds_lyr = fid_ds.CreateLayer(
                "fid_layer",
                geom_type=ogr.wkbPolygon,
                srs=zones_layer_meta["projection_osr"],
            )
            fid_ds_lyr.CreateFeature(feature.Clone())
            fid_ds.SyncToDisk()

            valid_path = f"/vsimem/{prefix}validmask_{str(fid)}{postfix}.tif"

            rasterize_vector(
                fid_ds,
                pixel_size,
                out_path=valid_path,
                extent=fid_ds,
            )
            valid_arr = raster_to_array(valid_path)

            if label_geom is not None and fid not in processed_fids:
                if not is_vector(label_geom):
                    raise TypeError(
                        "label geom is invalid. Did you input a valid geometry?"
                    )

                label_clip_path = f"/vsimem/fid_{uuid4().int}_{str(fid)}_clipped.shp"
                label_ras_path = f"/vsimem/fid_{uuid4().int}_{str(fid)}_rasterized.tif"
                label_warp_path = f"/vsimem/fid_{uuid4().int}_{str(fid)}_resampled.tif"

                intersect_vector(label_geom, fid_ds, out_path=label_clip_path)

                try:
                    rasterize_vector(
                        label_clip_path,
                        label_res,
                        out_path=label_ras_path,
                        extent=valid_path,
                    )

                except Exception:
                    array_to_raster(
                        np.zeros(valid_arr.shape, dtype="float32"),
                        valid_path,
                        out_path=label_ras_path,
                    )

                resample_raster(
                    label_ras_path,
                    pixel_size,
                    resample_alg="average",
                    out_path=label_warp_path,
                )

                labels_arr = (raster_to_array(label_warp_path) * label_mult).astype(
                    "float32"
                )

                if output_raster_labels:
                    array_to_raster(
                        labels_arr,
                        label_warp_path,
                        out_path=f"{outdir}{prefix}label_{str(fid)}{postfix}.tif",
                    )

            raster_clip_path = f"/vsimem/raster_{uuid4().int}_{str(idx)}_clipped.tif"

            try:
                clip_raster(
                    raster,
                    valid_path,
                    raster_clip_path,
                    all_touch=False,
                    adjust_bbox=False,
                )
            except Exception:
                print(f"Warning: {raster} did not intersect geom with fid: {fid}.")

                if label_geom is not None:
                    gdal.Unlink(label_clip_path)
                    gdal.Unlink(label_ras_path)
                    gdal.Unlink(label_warp_path)
                gdal.Unlink(fid_path)

                continue

            arr = raster_to_array(raster_clip_path)

            if arr.shape[:2] != valid_arr.shape[:2]:
                raise Exception(
                    f"Error while matching array shapes. Raster: {arr.shape}, Valid: {valid_arr.shape}"
                )

            arr_offsets = get_overlaps(arr, offsets, tile_size, border_check)

            arr = np.concatenate(arr_offsets)
            valid_offsets = np.concatenate(
                get_overlaps(valid_arr, offsets, tile_size, border_check)
            )

            valid_mask = (
                (1 - (valid_offsets.sum(axis=(1, 2)) / (tile_size * tile_size)))
                <= tolerance
            )[:, 0]

            arr = arr[valid_mask]
            valid_masked = valid_offsets[valid_mask]

            if label_geom is not None and not labels_processed:
                labels_masked = np.concatenate(
                    get_overlaps(labels_arr, offsets, tile_size, border_check)
                )[valid_mask]

            if merge_output:
                list_extracted.append(arr)
                list_masks.append(valid_masked)

                if label_geom is not None and not labels_processed:
                    list_labels.append(labels_masked)
            else:
                np.save(
                    f"{outdir}{prefix}{str(fid)}_{name}{postfix}.npy",
                    arr.filled(fill_value),
                )

                np.save(
                    f"{outdir}{prefix}{str(fid)}_mask_{name}{postfix}.npy",
                    valid_masked.filled(fill_value),
                )

                if label_geom is not None and not labels_processed:
                    np.save(
                        f"{outdir}{prefix}{str(fid)}_label_{name}{postfix}.npy",
                        valid_masked.filled(fill_value),
                    )

            if fid not in processed_fids:
                processed_fids.append(fid)

            processed += 1
            progress(processed, len(fids) * len(raster_list), "processing fids")

            if not merge_output:
                gdal.Unlink(label_clip_path)
                gdal.Unlink(label_ras_path)
                gdal.Unlink(label_warp_path)
                gdal.Unlink(fid_path)

            gdal.Unlink(valid_path)

        if merge_output:
            np.save(
                f"{outdir}{prefix}{name}{postfix}.npy",
                np.concatenate(list_extracted).filled(fill_value),
            )
            np.save(
                f"{outdir}{prefix}mask_{name}{postfix}.npy",
                np.concatenate(list_masks).filled(fill_value),
            )

            if label_geom is not None and not labels_processed:
                np.save(
                    f"{outdir}{prefix}label_{name}{postfix}.npy",
                    np.concatenate(list_labels).filled(fill_value),
                )
                labels_processed = True

    progress(1, 1, "processing fids")

    return 1


def predict_raster(
    raster_list,
    tile_size=[32],
    output_tile_size=32,
    output_channels=1,
    model_path="",
    reference_raster="",
    out_path=None,
    offsets=[[[0, 0]]],
    batch_size=32,
    method="median",
    scale_to_sum=False,
):
    print("Loading Model")
    model = tf.keras.models.load_model(model_path, custom_objects={"tpe": tpe})
    reference_arr = raster_to_array(reference_raster)

    predictions = []
    read_rasters = []

    print("Initialising rasters.")
    for raster_idx, raster in enumerate(raster_list):
        read_rasters.append(raster_to_array(raster).astype("float32"))

    progress(0, len(offsets[0]) + 3, "Predicting")
    for offset_idx in range(len(offsets[0])):
        model_inputs = []
        for raster_idx, raster in enumerate(raster_list):
            array = read_rasters[raster_idx]

            blocks = array_to_blocks(
                array, tile_size[raster_idx], offset=offsets[raster_idx][offset_idx]
            )

            model_inputs.append(blocks)

        prediction_blocks = model.predict(model_inputs, batch_size, verbose=0)

        prediction = blocks_to_array(
            prediction_blocks,
            (reference_arr.shape[0], reference_arr.shape[1], output_channels),
            output_tile_size,
            offset=offsets[0][offset_idx],
        )
        predictions.append(prediction)

        progress(offset_idx + 1, len(offsets[0]) + 3, "Predicting")

    # Predict the border regions and add them as a layer
    with np.errstate(invalid="ignore"):
        borders = (
            np.empty((reference_arr.shape[0], reference_arr.shape[1], output_channels))
            * np.nan
        )

    for end in ["right", "bottom", "corner"]:
        model_inputs = []

        for raster_idx, raster in enumerate(raster_list):
            if end == "right":
                array = read_rasters[raster_idx][:, -tile_size[raster_idx] :]
            elif end == "bottom":
                array = read_rasters[raster_idx][-tile_size[raster_idx] :, :]
            elif end == "corner":
                array = read_rasters[raster_idx][
                    -tile_size[raster_idx] :, -tile_size[raster_idx] :
                ]

            blocks = array_to_blocks(array, tile_size[raster_idx], offset=[0, 0])

            model_inputs.append(blocks)

        prediction_blocks = model.predict(model_inputs, batch_size, verbose=0)

        if end == "right":
            target_shape = (reference_arr.shape[0], output_tile_size, output_channels)
        elif end == "bottom":
            target_shape = (output_tile_size, reference_arr.shape[1], output_channels)
        elif end == "corner":
            target_shape = (output_tile_size, output_tile_size, output_channels)

        prediction = blocks_to_array(
            prediction_blocks, target_shape, output_tile_size, offset=[0, 0]
        )

        if end == "right":
            borders[:, -output_tile_size:, 0:output_channels] = prediction
        elif end == "bottom":
            borders[-output_tile_size:, :, 0:output_channels] = prediction
        elif end == "corner":
            borders[
                -output_tile_size:, -output_tile_size:, 0:output_channels
            ] = prediction

        offset_idx += 1
        progress(offset_idx + 1, len(offsets[0]) + 3, "Predicting")

    predictions.append(borders)

    print("Merging predictions.")
    with np.errstate(invalid="ignore"):
        if method == "mean":
            predicted = np.nanmean(predictions, axis=0).astype("float32")
        elif method == "olympic" or method == "olympic_1":
            sort = np.sort(predictions, axis=0)[1:-1]
            predicted = np.nanmean(sort, axis=0).astype("float32")
        elif method == "olympic_2":
            sort = np.sort(predictions, axis=0)[2:-2]
            predicted = np.nanmean(sort, axis=0).astype("float32")
        else:
            predicted = np.nanmedian(predictions, axis=0).astype("float32")

        if scale_to_sum:
            predicted = predicted / np.reshape(
                np.nansum(predicted, axis=2),
                (predicted.shape[0], predicted.shape[1], 1),
            )

    return array_to_raster(predicted, reference_raster, out_path)


# example:

from glob import glob

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_dar/"
rasters = glob(folder + "B0*_10m*.tif")
vector = folder + "/vector/dar_test_geom.gpkg"
label = folder + "vector/dar_buildings_01_test.gpkg"

extract_patches(rasters, folder + "tmp/", zones=vector, options={"label_geom": label})
