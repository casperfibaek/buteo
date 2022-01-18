import sys
import os
import numpy as np
import tensorflow as tf
from osgeo import ogr, gdal
from uuid import uuid4
from numba import jit, prange

sys.path.append("../../")

np.set_printoptions(suppress=True)

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
from buteo.vector.intersect import intersect_vector
from buteo.vector.attributes import vector_get_fids
from buteo.vector.rasterize import rasterize_vector
from buteo.machine_learning.ml_utils import tpe, get_offsets, mse_sumbias
from buteo.utils import progress


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
        found = False
        border = None
        for end in ["right", "bottom", "corner"]:
            if end == "right" and (arr.shape[1] % tile_size) != 0:
                found = True
                border = arr[:, -tile_size:]
            elif end == "bottom" and (arr.shape[0] % tile_size) != 0:
                found = True
                border = arr[-tile_size:, :]
            elif (
                end == "corner"
                and ((arr.shape[1] % tile_size) != 0)
                and ((arr.shape[1] % tile_size) != 0)
            ):
                found = True
                border = arr[-tile_size:, -tile_size:]

            if found:
                arr_offsets.append(array_to_blocks(border, tile_size, offset=[0, 0]))

    return arr_offsets


# todo: align with size
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
    base_options = {
        "overlaps": True,
        "border_check": True,
        "merge_output": True,
        "force_align": True,
        "output_raster_labels": True,
        "label_geom": None,
        "label_res": 0.2,
        "label_mult": 100,
        "tolerance": 0.0,
        "fill_value": 0,
        "zone_layer_id": 0,
        "align_with_size": 20,
        "prefix": "",
        "postfix": "",
    }

    if options is None:
        options = base_options
    else:
        for key in options:
            if key not in base_options:
                raise ValueError(f"Invalid option: {key}")
            base_options[key] = options[key]
        options = base_options

    if zones is not None and not is_vector(zones):
        raise TypeError("Clip geom is invalid. Did you input a valid geometry?")

    if not isinstance(raster_list, list):
        raster_list = [raster_list]

    for raster in raster_list:
        if not is_raster(raster):
            raise TypeError("raster_list is not a list of rasters.")

    if not os.path.isdir(outdir):
        raise ValueError(
            "Outdir does not exist. Please create before running the function."
        )

    if not rasters_are_aligned(raster_list, same_extent=True):
        if options["force_align"]:
            print(
                "Rasters we not aligned. Realigning rasters due to force_align=True option."
            )
            raster_list = align_rasters(raster_list)
        else:
            raise ValueError("Rasters in raster_list are not aligned.")

    offsets = get_offsets(tile_size) if options["overlaps"] else [[0, 0]]
    raster_metadata = raster_to_metadata(raster_list[0], create_geometry=True)
    pixel_size = min(raster_metadata["pixel_height"], raster_metadata["pixel_width"])

    if zones is None:
        zones = raster_metadata["extent_datasource_path"]

    zones_meta = vector_to_metadata(zones)

    mem_driver = ogr.GetDriverByName("ESRI Shapefile")

    if zones_meta["layer_count"] == 0:
        raise ValueError("Vector contains no layers.")

    zones_layer_meta = zones_meta["layers"][options["zone_layer_id"]]

    if zones_layer_meta["geom_type"] not in ["Multi Polygon", "Polygon"]:
        raise ValueError("clip geom is not Polygon or Multi Polygon.")

    zones_ogr = open_vector(zones)
    zones_layer = zones_ogr.GetLayer(options["zone_layer_id"])
    featureDefn = zones_layer.GetLayerDefn()
    fids = vector_get_fids(zones_ogr, options["zone_layer_id"])

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
            geom = feature.GetGeometryRef()
            fid_path = f"/vsimem/fid_mem_{uuid4().int}_{str(fid)}.shp"
            fid_ds = mem_driver.CreateDataSource(fid_path)
            fid_ds_lyr = fid_ds.CreateLayer(
                "fid_layer",
                geom_type=ogr.wkbPolygon,
                srs=zones_layer_meta["projection_osr"],
            )
            copied_feature = ogr.Feature(featureDefn)
            copied_feature.SetGeometry(geom)
            fid_ds_lyr.CreateFeature(copied_feature)

            fid_ds.FlushCache()
            fid_ds.SyncToDisk()

            valid_path = f"/vsimem/{options['prefix']}validmask_{str(fid)}{options['postfix']}.tif"

            rasterize_vector(
                fid_path,
                pixel_size,
                out_path=valid_path,
                extent=fid_path,
            )
            valid_arr = raster_to_array(valid_path)

            if options["label_geom"] is not None and fid not in processed_fids:
                if not is_vector(options["label_geom"]):
                    raise TypeError(
                        "label geom is invalid. Did you input a valid geometry?"
                    )

                uuid = str(uuid4().int)

                label_clip_path = f"/vsimem/fid_{uuid}_{str(fid)}_clipped.shp"
                label_ras_path = f"/vsimem/fid_{uuid}_{str(fid)}_rasterized.tif"
                label_warp_path = f"/vsimem/fid_{uuid}_{str(fid)}_resampled.tif"

                intersect_vector(
                    options["label_geom"], fid_ds, out_path=label_clip_path
                )

                try:
                    rasterize_vector(
                        label_clip_path,
                        options["label_res"],
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

                labels_arr = (
                    raster_to_array(label_warp_path) * options["label_mult"]
                ).astype("float32")

                if options["output_raster_labels"]:
                    array_to_raster(
                        labels_arr,
                        label_warp_path,
                        out_path=f"{outdir}{options['prefix']}label_{str(fid)}{options['postfix']}.tif",
                    )

            raster_clip_path = f"/vsimem/raster_{uuid}_{str(idx)}_clipped.tif"

            try:
                clip_raster(
                    raster,
                    valid_path,
                    raster_clip_path,
                    all_touch=False,
                    adjust_bbox=False,
                )
            except Exception as e:
                print(f"Warning: {raster} did not intersect geom with fid: {fid}.")
                print(e)

                if options["label_geom"] is not None:
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

            arr_offsets = get_overlaps(arr, offsets, tile_size, options["border_check"])

            arr = np.concatenate(arr_offsets)
            valid_offsets = np.concatenate(
                get_overlaps(valid_arr, offsets, tile_size, options["border_check"])
            )

            valid_mask = (
                (1 - (valid_offsets.sum(axis=(1, 2)) / (tile_size * tile_size)))
                <= options["tolerance"]
            )[:, 0]

            arr = arr[valid_mask]
            valid_masked = valid_offsets[valid_mask]

            if options["label_geom"] is not None and not labels_processed:
                labels_masked = np.concatenate(
                    get_overlaps(
                        labels_arr, offsets, tile_size, options["border_check"]
                    )
                )[valid_mask]

            if options["merge_output"]:
                list_extracted.append(arr)
                list_masks.append(valid_masked)

                if options["label_geom"] is not None and not labels_processed:
                    list_labels.append(labels_masked)
            else:
                np.save(
                    f"{outdir}{options['prefix']}{str(fid)}_{name}{options['postfix']}.npy",
                    arr.filled(options["fill_value"]),
                )

                np.save(
                    f"{outdir}{options['prefix']}{str(fid)}_mask_{name}{options['postfix']}.npy",
                    valid_masked.filled(options["fill_value"]),
                )

                if options["label_geom"] is not None and not labels_processed:
                    np.save(
                        f"{outdir}{options['prefix']}{str(fid)}_label_{name}{options['postfix']}.npy",
                        valid_masked.filled(options["fill_value"]),
                    )

            if fid not in processed_fids:
                processed_fids.append(fid)

            processed += 1
            progress(processed, len(fids) * len(raster_list), "processing fids")

            if not options["merge_output"]:
                gdal.Unlink(label_clip_path)
                gdal.Unlink(label_ras_path)
                gdal.Unlink(label_warp_path)
                gdal.Unlink(fid_path)

            gdal.Unlink(valid_path)

        if options["merge_output"]:
            np.save(
                f"{outdir}{options['prefix']}{name}{options['postfix']}.npy",
                np.ma.concatenate(list_extracted).filled(options["fill_value"]),
            )
            np.save(
                f"{outdir}{options['prefix']}mask_{name}{options['postfix']}.npy",
                np.ma.concatenate(list_masks).filled(options["fill_value"]),
            )

            if options["label_geom"] is not None and not labels_processed:
                np.save(
                    f"{outdir}{options['prefix']}label_{name}{options['postfix']}.npy",
                    np.ma.concatenate(list_labels).filled(options["fill_value"]),
                )
                labels_processed = True

    progress(1, 1, "processing fids")

    return 1


@jit(nopython=True, nogil=True)
def hood_quantile(values, weights, quant):
    sort_mask = np.argsort(values)
    sorted_data = values[sort_mask]
    sorted_weights = weights[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quant, intersect, sorted_data)


@jit(nopython=True, nogil=True)
def count_within(values, lower_limit, upper_limit):
    copy = np.zeros_like(values)

    for x in range(values.shape[0]):
        low = values[x] - lower_limit
        high = values[x] + upper_limit

        for y in range(x, values.shape[0]):
            if values[y] > high:  # its sorted, so we can break
                break

            if values[y] >= low and values[y] <= high:
                copy[x] += 1

    return copy, copy.max()


@jit(nopython=True, parallel=True, nogil=True)
def mad_collapse(predictions, default=0.0):
    pred = np.zeros((predictions.shape[0], predictions.shape[1], 1), dtype=np.float32)

    for x in prange(predictions.shape[0]):
        for y in range(predictions.shape[1]):
            non_nan = predictions[x, y, :][np.isnan(predictions[x, y, :]) == False]
            if non_nan.shape[0] == 0:
                pred[x, y, 0] = default
                continue

            non_nan.sort()
            non_nan_len = non_nan.shape[0]
            median = np.median(non_nan)
            mad = np.nanmedian(np.abs(median - non_nan))

            counts, max_count = count_within(non_nan, mad, mad)

            current_item = 0
            center_items = np.empty(non_nan_len ** 2, dtype="float32")
            center_items_weights = np.zeros(non_nan_len ** 2, dtype="float32")

            for z in range(non_nan_len):
                lower_limit = non_nan[z] - mad
                upper_limit = non_nan[z] + mad
                if counts[z] == max_count:
                    for q in range(non_nan_len):
                        if (
                            non_nan[q] >= lower_limit and non_nan[q] <= upper_limit
                        ) or q == z:
                            center_items[current_item] = non_nan[q]
                            center_items_weights[current_item] = counts[q]
                            current_item += 1

            center_items = center_items[:current_item]
            center_items_weights = center_items_weights[:current_item]
            center_items_weights = center_items_weights / center_items_weights.sum()

            pred[x, y, 0] = hood_quantile(center_items, center_items_weights, 0.5)

    return pred


def predict_raster(
    raster_list,
    tile_size=[32],
    output_tile_size=32,
    output_channels=1,
    model_path="",
    reference_raster="",
    out_path=None,
    out_path_variance=None,
    offsets=True,
    batch_size=32,
    method="median",
    scale_to_sum=False,
):
    print("Loading Model")
    model = tf.keras.models.load_model(
        model_path, custom_objects={"tpe": tpe, "mse_sumbias": mse_sumbias}
    )
    reference_arr = raster_to_array(reference_raster)

    if offsets == False:
        print(
            "No offsets provided. Using offsets greatly increases accuracy. Please provide offsets."
        )
    else:
        offsets = []
        for val in tile_size:
            offsets.append(get_offsets(val))

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
        elif method == "mad":
            predicted = mad_collapse(
                np.concatenate(predictions, axis=2).astype("float32")
            )
        else:
            predicted = np.nanmedian(predictions, axis=0).astype("float32")

        if scale_to_sum:
            predicted = predicted / np.reshape(
                np.nansum(predicted, axis=2),
                (predicted.shape[0], predicted.shape[1], 1),
            )

    if out_path_variance is not None:
        array_to_raster(
            np.nanvar(predictions, axis=0), reference_raster, out_path_variance
        )

    return array_to_raster(predicted, reference_raster, out_path)
