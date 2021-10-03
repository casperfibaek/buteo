import sys

sys.path.append("../../")
import numpy as np

np.set_printoptions(suppress=True)

import tensorflow as tf
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.machine_learning.ml_utils import tpe
from buteo.utils import progress


def array_to_blocks(arr: np.ndarray, tile_size: tuple, offset=[0, 0]) -> np.ndarray:
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

    progress(0, len(offsets[0]) + 1, "Predicting")
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

        progress(offset_idx + 1, len(offsets[0]) + 2, "Predicting")

    # Predict the border regions and add them as a layer
    with np.errstate(invalid="ignore"):
        borders = (
            np.empty((reference_arr.shape[0], reference_arr.shape[1], output_channels))
            * np.nan
        )

    for idx, end in enumerate(["right", "bottom"]):
        model_inputs = []

        for raster_idx, raster in enumerate(raster_list):
            if end == "right":
                array = read_rasters[raster_idx][:, -tile_size[raster_idx] :]
            else:
                array = read_rasters[raster_idx][-tile_size[raster_idx] :, :]

            blocks = array_to_blocks(array, tile_size[raster_idx], offset=[0, 0])

            model_inputs.append(blocks)

        prediction_blocks = model.predict(model_inputs, batch_size, verbose=0)

        target_shape = (reference_arr.shape[0], output_tile_size, output_channels)
        if end == "bottom":
            target_shape = (output_tile_size, reference_arr.shape[1], output_channels)

        prediction = blocks_to_array(
            prediction_blocks, target_shape, output_tile_size, offset=[0, 0]
        )

        if end == "right":
            borders[:, -output_tile_size:, 0:output_channels] = prediction
        else:
            borders[-output_tile_size:, :, 0:output_channels] = prediction

        progress(offset_idx + 1, len(offsets[0]) + idx + 1, "Predicting")

    predictions.append(borders)

    print("Merging predictions.")

    with np.errstate(invalid="ignore"):
        if method == "mean":
            predicted = np.nanmean(predictions, axis=0).astype("float32")
        else:
            predicted = np.nanmedian(predictions, axis=0).astype("float32")

        if scale_to_sum:
            predicted = predicted / np.reshape(
                np.nansum(predicted, axis=2),
                (predicted.shape[0], predicted.shape[1], 1),
            )

    return array_to_raster(predicted, reference_raster, out_path)
