import sys
import numpy as np
import tensorflow as tf
from numba import jit, prange

sys.path.append("../../")

from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
)
from buteo.machine_learning.ml_utils import tpe, get_offsets, mse_sumbias
from buteo.machine_learning.patch_utils import array_to_blocks, blocks_to_array
from buteo.utils import progress


@jit(nopython=True, parallel=True, nogil=True, inline="always")
def weighted_quantile(values, weights, quant):
    sort_mask = np.argsort(values)
    sorted_data = values[sort_mask]
    sorted_weights = weights[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quant, intersect, sorted_data)


@jit(nopython=True, parallel=True, nogil=True, inline="always")
def values_within(values, limit):
    weights = np.zeros_like(values)
    best_mask = np.ones_like(values, dtype=np.bool_)

    max_observations = 0
    for i in range(values.shape[0]):
        lower_limit = values[i] - limit
        upper_limit = values[i] + limit

        observations = 0
        observations_mask = np.zeros_like(values, dtype=np.bool_)
        for j in range(i, values.shape[0]):
            if values[j] > upper_limit:  # its sorted, so we can break
                break
            elif values[j] < lower_limit:
                continue

            observations_mask[j] = True
            observations += 1
            weights[i] += 1

        if observations > max_observations:
            max_observations = observations
            best_mask = observations_mask

    if max_observations > 1:
        weights[best_mask] = weights[best_mask] - 1

    central_values = values[best_mask]
    central_weights = weights[best_mask] / weights[best_mask].sum()

    return central_values, central_weights


@jit(nopython=True, parallel=True, nogil=True)
def mad_interval_merging(predictions, default=0.0, bias=1.0):
    pred = np.zeros((predictions.shape[0], predictions.shape[1], 1), dtype=np.float32)

    for x in prange(predictions.shape[0]):
        for y in range(predictions.shape[1]):
            non_nan = predictions[x, y, :][np.isnan(predictions[x, y, :]) == False]
            if non_nan.size == 0:
                pred[x, y, 0] = default
                continue
            elif non_nan.size <= 2:
                pred[x, y, 0] = np.nanmean(non_nan)
                continue

            non_nan.sort()
            median = np.median(non_nan)
            mad = np.median(np.abs(median - non_nan)) * bias

            values, weights = values_within(non_nan, mad)

            collapsed = weighted_quantile(values, weights, 0.5)

            pred[x, y, 0] = collapsed

    return pred


# Test for classification tasks.
def predict_raster(
    raster_list,
    /,
    *,
    tile_size=[32],
    output_tile_size=32,
    output_channels=1,
    model_path="",
    loaded_model=None,
    reference_raster="",
    out_path=None,
    out_path_variance=None,
    offsets=True,
    batch_size=32,
    method="median",
    scale_to_sum=False,
):
    if loaded_model is None:
        print("Loading Model")
        model = tf.keras.models.load_model(model_path, custom_objects={"tpe": tpe })
    else:
        model = loaded_model

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
        pred_readied = np.concatenate(predictions, axis=2).astype("float32")
        if method == "mean":
            predicted = np.nanmean(pred_readied)
        elif method == "olympic" or method == "olympic_1":
            sort = np.sort(pred_readied)[1:-1]
            predicted = np.nanmean(sort)
        elif method == "olympic_2":
            sort = np.sort(pred_readied)[2:-2]
            predicted = np.nanmean(sort)
        elif method == "mad":
            predicted = mad_interval_merging(pred_readied)
        else:
            predicted = np.nanmedian(pred_readied)

        if scale_to_sum:
            predicted = predicted / np.sum(predicted)

    if out_path_variance is not None:
        array_to_raster(
            np.nanvar(pred_readied),
            reference=reference_raster,
            out_path=out_path_variance,
        )

    return array_to_raster(predicted, reference=reference_raster, out_path=out_path)
