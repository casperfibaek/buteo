import sys

sys.path.append("../../")
import numpy as np
import os
import random
from typing import Any, Dict, Union, Optional, Tuple, List
from osgeo import ogr, gdal
from uuid import uuid4
import rtree

from buteo.project_types import Number
from buteo.raster.io import (
    open_raster,
    to_raster_list,
    raster_to_array,
    internal_raster_to_disk,
    internal_raster_to_metadata,
    array_to_raster,
    rasters_are_aligned,
)
from buteo.vector.io import (
    internal_vector_to_metadata,
    internal_vector_to_disk,
    open_vector,
    vector_add_index,
)
from buteo.vector.reproject import internal_reproject_vector
from buteo.raster.clip import clip_raster, internal_clip_raster
from buteo.raster.resample import internal_resample_raster
from buteo.utils import overwrite_required, remove_if_overwrite, progress, type_check
from buteo.gdal_utils import ogr_bbox_intersects


def reconstitute_raster(
    blocks: np.ndarray,
    raster_height: int,
    raster_width: int,
    size: int,
    offset: Tuple[Number, Number],
    border_patches: bool,
    border_patches_x: bool,
    border_patches_y: bool,
) -> np.ndarray:
    """ Recombines blocks into an array.
    Args:
        blocks (ndarray): A numpy array with the values to recombine. The shape
        should be (blocks, rows, column, channel).

        raster_height (int): height in pixels of target raster.
        
        raster_width (int): width in pixels of target raster.

        size (int): size of patches in pixels. (square patches.)

        offset (tuple): A tuple with the offset of the blocks (x, y)

        border_patches (bool): Does the patches contain border_patches?

        border_patches_x (bool): Does the patches contain border_patches on the x axis?

        border_patches_y (bool): Does the patches contain border_patches on the y axis?

    Returns:
        A reconstituted raster.
    """
    type_check(blocks, [np.ndarray], "blocks")
    type_check(raster_height, [int], "raster_height")
    type_check(raster_width, [int], "raster_width")
    type_check(size, [int], "size")
    type_check(offset, [tuple], "offset")
    type_check(border_patches, [bool], "border_patches")
    type_check(border_patches_x, [bool], "border_patches_x")
    type_check(border_patches_y, [bool], "border_patches_y")

    ref_shape = [raster_height - offset[1], raster_width - offset[0]]

    if offset != (0, 0):
        border_patches = False
        border_patches_x = False
        border_patches_y = False

    if border_patches and (border_patches_x or border_patches_y):
        if border_patches_x:
            ref_shape[1] = ((ref_shape[1] // size) * size) + size
        if border_patches_y:
            ref_shape[0] = ((ref_shape[0] // size) * size) + size

    reshape = blocks.reshape(
        ref_shape[0] // size,
        ref_shape[1] // size,
        size,
        size,
        blocks.shape[3],
        blocks.shape[3],
    )

    swap = reshape.swapaxes(1, 2)

    destination = swap.reshape(
        (ref_shape[0] // size) * size, (ref_shape[1] // size) * size, blocks.shape[3],
    )

    # Order: Y, X, Z
    if border_patches and (border_patches_x or border_patches_y):

        x_offset = 0
        y_offset = 0

        if border_patches_x:
            x_offset = int(ref_shape[1] - raster_width)
            x_edge = destination[:raster_height, -(size - x_offset) :, :]
            destination[:raster_height, -size:-x_offset, :] = x_edge

        if border_patches_y:
            y_offset = int(ref_shape[0] - raster_height)
            y_edge = destination[-(size - y_offset) :, :raster_width, :]
            destination[-size:-y_offset, :raster_width, :] = y_edge

        if border_patches_y and border_patches_y:
            corner = destination[-(size - y_offset) :, -(size - x_offset) :, :]
            destination[-size:-y_offset, -size:-x_offset, :] = corner

        destination = destination[:raster_height, :raster_width, 0 : blocks.shape[3]]

    return destination


def blocks_to_raster(
    blocks: np.ndarray,
    reference: Union[str, gdal.Dataset],
    out_path: Union[str, None] = None,
    offsets: Union[list, tuple, np.ndarray] = [],
    border_patches: bool = True,
    generate_zero_offset: bool = True,
    merge_method: str = "median",
    output_array: bool = False,
    verbose: int = 1,
) -> Union[str, np.ndarray]:
    """ Recombines a series of blocks to a raster. OBS: Does not work if the patch
        extraction was done with clip geom.
    Args:
        blocks (ndarray): A numpy array with the values to recombine. The shape
        should be (blocks, rows, column, channel).

        reference (str, raster): A reference raster to help coax the blocks back
        into shape.

    **kwargs:
        out_path (str | None): Where to save the reconstituted raster. If None
        are memory raster is returned.

        offsets (tuple, list, ndarray): The offsets used in the original. A (0 ,0)
        offset is assumed.

        border_patches (bool): Do the blocks contain border patches?

        generate_zero_offset (bool): if True, an offset is inserted at (0, 0)
        if none is present.

        merge_method (str): How to handle overlapping pixels. Options are:
        median, average, mode, min, max

        output_array (bool): If True the output will be a numpy array instead of a
        raster.

        verbose (int): If 1 will output messages on progress.

    Returns:
        A reconstituted raster.
    """
    type_check(blocks, [str, np.ndarray], "blocks")
    type_check(reference, [str, gdal.Dataset], "reference")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(offsets, [list, tuple, np.ndarray], "offsets", allow_none=True)
    type_check(border_patches, [bool], "border_patches")
    type_check(generate_zero_offset, [bool], "bool")
    type_check(merge_method, [str], "merge_method")
    type_check(output_array, [bool], "output_array")
    type_check(verbose, [int], "verbose")

    if isinstance(blocks, str):
        try:
            blocks = np.load(blocks)
        except:
            raise ValueError(f"Failed to parse blocks: {blocks}")

    if verbose == 1:
        print("Reconstituting blocks into target raster.")

    metadata = internal_raster_to_metadata(reference)

    border_patches_x = False
    border_patches_y = False

    if blocks.shape[1] != blocks.shape[2]:
        raise ValueError(
            "The input blocks must be square. Rectangles might supported in the future."
        )

    size = blocks.shape[1]

    if metadata["width"] % size != 0 and border_patches:
        border_patches_x = True
    if metadata["height"] % size != 0 and border_patches:
        border_patches_y = True

    # internal offset array. Avoid manipulating the og array.
    in_offsets = []

    if generate_zero_offset:
        if offsets is not None:
            if (0, 0) not in offsets:
                in_offsets.append((0, 0))
        else:
            in_offsets.append((0, 0))

    for offset in offsets:
        if offset != (0, 0):
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise ValueError(
                    f"offset must be a list or tuple of two integers. Recieved: {offset}"
                )
            in_offsets.append((offset[0], offset[1]))
        elif not generate_zero_offset:
            in_offsets.append((offset[0], offset[1]))

    # Easier to read this way.
    has_offsets = False
    if generate_zero_offset and len(in_offsets) > 1:
        has_offsets = True

    if not generate_zero_offset and len(in_offsets) > 0:
        has_offsets = True

    if has_offsets:
        passes = []

        previous = 0
        largest_x = 0
        largest_y = 0
        for index, offset in enumerate(in_offsets):
            passes.append(
                np.ma.masked_all(
                    (metadata["height"], metadata["width"], blocks.shape[3],),
                    dtype=metadata["datatype"],
                ),
            )

            if index == 0:
                x_blocks = ((metadata["width"] - offset[0]) // size) + border_patches_x
                y_blocks = ((metadata["height"] - offset[1]) // size) + border_patches_x
            else:
                x_blocks = (metadata["width"] - offset[0]) // size
                y_blocks = (metadata["height"] - offset[1]) // size

            block_size = x_blocks * y_blocks

            raster_pass = reconstitute_raster(  # pylint: disable=too-many-function-args
                blocks[previous : block_size + previous, :, :, :],
                metadata["height"],
                metadata["width"],
                size,
                offset,
                border_patches,
                border_patches_x,
                border_patches_y,
            )

            if raster_pass.shape[1] > largest_x:
                largest_x = raster_pass.shape[1]

            if raster_pass.shape[0] > largest_y:
                largest_y = raster_pass.shape[0]

            previous += block_size

            passes[index][
                offset[1] : raster_pass.shape[0] + offset[1],
                offset[0] : raster_pass.shape[1] + offset[0],
                :,
            ] = raster_pass

            passes[index] = passes[index].filled(np.nan)

        if merge_method == "median":
            raster = np.nanmedian(passes, axis=0)
        elif merge_method == "mean" or merge_method == "average":
            raster = np.nanmean(passes, axis=0)
        elif merge_method == "min" or merge_method == "minumum":
            raster = np.nanmin(passes, axis=0)
        elif merge_method == "max" or merge_method == "maximum":
            raster = np.nanmax(passes, axis=0)
        elif merge_method == "mode" or merge_method == "majority":
            for index, _ in enumerate(passes):
                passes[index] = np.rint(passes[index]).astype(int)
            raster = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=passes
            )
        else:
            raise ValueError(f"Unable to parse merge_method: {merge_method}")

    else:
        raster: np.ndarray = reconstitute_raster(
            blocks,
            metadata["height"],
            metadata["width"],
            size,
            (0, 0),
            border_patches,
            border_patches_x,
            border_patches_y,
        )

    if output_array:
        return raster

    return array_to_raster(raster, reference, out_path=out_path)


def shape_to_blockshape(shape: tuple, block_shape: tuple, offset: tuple) -> list:
    """ Calculates the shape of the output array.
    Args:
        shape (tuple | list): The shape if the original raster. (1980, 1080, 3)

        block_shape (tuple | list): The size of the blocks eg. (64, 64)

        offset (tuple, list): An initial offset for the array eg. (32, 32)

    Returns:
        A list with the modified shape.
    """
    type_check(shape, [tuple], "shape")
    type_check(block_shape, [tuple], "block_shape")
    type_check(offset, [tuple], "offset")

    # import pdb

    # pdb.set_trace()

    assert len(offset) == 2, "Offset has to be two dimensional."
    assert len(shape) == 3, "Shape has to be three dimensional."
    assert len(block_shape) == 2, "Shape of block has to be two dimensional."

    base_shape = list(shape)
    for index, value in enumerate(offset):
        base_shape[index] = base_shape[index] - value

    sizes = [base_shape[0] // block_shape[0], base_shape[1] // block_shape[1]]

    return sizes


def array_to_blocks(
    array: np.ndarray,
    block_shape: tuple,
    offset: tuple,
    border_patches_x: bool = False,
    border_patches_y: bool = False,
) -> np.ndarray:
    """ Turns an array into a series of blocks. The array can be offset.
    Args:
        array (ndarray): The array to turn to blocks. (Channel last format: 1920x1080x3)

        block_shape (tuple | list | ndarray): The size of the blocks eg. (64, 64)

        offset (tuple, list, ndarray): An initial offset for the array eg. (32, 32)

    Returns:
        A modified view into the array.
    """
    type_check(array, [np.ndarray], "array")
    type_check(block_shape, [tuple], "block_shape")
    type_check(offset, [tuple], "offset")
    type_check(border_patches_x, [bool], "border_patches_x")
    type_check(border_patches_y, [bool], "border_patches_y")

    assert array.ndim == 3, "Input raster must be three dimensional"

    arr = array[
        offset[1] : int(
            array.shape[0] - ((array.shape[0] - offset[1]) % block_shape[0])
        ),
        offset[0] : int(
            array.shape[1] - ((array.shape[1] - offset[0]) % block_shape[1])
        ),
        :,
    ]

    # If border patches are needed. A new array must be created to hold the border values.
    if border_patches_x or border_patches_y:
        x_shape = arr.shape[1]
        y_shape = arr.shape[0]

        if border_patches_x:
            x_shape = (
                (array.shape[1] // block_shape[0]) * block_shape[0]
            ) + block_shape[0]

        if border_patches_y:
            y_shape = (
                (array.shape[0] // block_shape[1]) * block_shape[1]
            ) + block_shape[1]

        # Expand and copy the original array
        arr_exp = np.empty((y_shape, x_shape, array.shape[2]), dtype=array.dtype)
        arr_exp[0 : arr.shape[0], 0 : arr.shape[1], 0 : arr.shape[2]] = arr

        if border_patches_x:
            arr_exp[: array.shape[0], -block_shape[1] :, :] = array[
                :, -block_shape[0] :, :
            ]

        if border_patches_y:
            arr_exp[-block_shape[0] :, : array.shape[1], :] = array[
                -block_shape[1] :, :, :
            ]

        # The bottom right corner will still have empty values if both a True
        if border_patches_x and border_patches_y:
            border_square = array[-block_shape[0] :, -block_shape[1] :, :]
            arr_exp[-block_shape[0] :, -block_shape[1] :, :] = border_square

        arr = arr_exp

    # This only creates views into the array, so should still be fast.
    reshaped = arr.reshape(
        arr.shape[0] // block_shape[0],
        block_shape[0],
        arr.shape[1] // block_shape[1],
        block_shape[1],
        arr.shape[2],
    )
    swaped = reshaped.swapaxes(1, 2)
    merge = swaped.reshape(-1, block_shape[0], block_shape[1], array.shape[2])

    return merge


def test_extraction(
    rasters: Union[list, str, gdal.Dataset],
    arrays: Union[list, np.ndarray],
    grid: Union[ogr.DataSource, str],
    samples: int = 1000,  # if 0, all
    grid_layer_index: int = 0,
    verbose: int = 1,
) -> bool:
    """ Validates the output of the patch_extractor. Useful if you need peace of mind.
    Set samples to 0 to tests everything.
    Args:
        rasters (list of rasters | path | raster): The raster(s) used.

        arrays (list of arrays | ndarray): The arrays generated.

        grid (vector | vector_path): The grid generated.

    **kwargs:
        samples (int): The amount of patches to randomly test. If 0 all patches will be
        tested. This is a long process, so consider only testing everything if absolutely
        necessary.

        grid_layer_index (int): If the grid is part of a multi-layer vector, specify the
        index of the grid.

        verbose (int): If 1 will output messages on progress.

    Returns:
        True if the extraction is valid. Raises an error otherwise.
    """
    type_check(rasters, [list, str, gdal.Dataset], "rasters")
    type_check(arrays, [list, str, np.ndarray], "arrays")
    type_check(grid, [list, str, ogr.DataSource], "grid")
    type_check(samples, [int], "samples")
    type_check(grid_layer_index, [int], "clip_layer_index")
    type_check(verbose, [int], "verbose")

    in_rasters = to_raster_list(rasters)
    in_arrays = arrays

    if verbose == 1:
        print("Verifying integrity of output grid..")

    # grid_memory = open_vector(internal_vector_to_memory(grid))
    grid_memory = open_vector(grid)
    grid_metadata = internal_vector_to_metadata(grid)
    grid_projection = grid_metadata["projection_osr"]

    if grid_layer_index > (grid_metadata["layer_count"] - 1):
        raise ValueError(f"Requested non-existing layer index: {grid_layer_index}")

    grid_layer = grid_memory.GetLayer(grid_layer_index)

    # Select sample fids
    feature_count = grid_metadata["layers"][grid_layer_index]["feature_count"]
    test_samples = samples if samples > 0 else feature_count
    max_test = min(test_samples, feature_count) - 1
    test_fids = np.array(
        random.sample(range(0, feature_count), max_test), dtype="uint64"
    )

    mem_driver = ogr.GetDriverByName("ESRI Shapefile")
    for index, raster in enumerate(in_rasters):
        test_rast = open_raster(raster)

        test_array = in_arrays[index]
        if isinstance(test_array, str):
            if not os.path.exists(test_array):
                raise ValueError(f"Numpy array does not exist: {test_array}")

            try:
                test_array = np.load(in_arrays[index])
            except:
                raise Exception(
                    f"Attempted to read numpy raster from: {in_arrays[index]}"
                )

        base = os.path.basename(raster)
        basename = os.path.splitext(base)[0]

        if verbose == 1:
            print(f"Testing: {basename}")

        tested = 0

        for test in test_fids:
            feature = grid_layer.GetFeature(test)

            if feature is None:
                raise Exception(f"Feature not found: {test}")

            test_ds_path = f"/vsimem/test_mem_grid_{uuid4().int}.gpkg"
            test_ds = mem_driver.CreateDataSource(test_ds_path)
            test_ds_lyr = test_ds.CreateLayer(
                "test_mem_grid_layer", geom_type=ogr.wkbPolygon, srs=grid_projection
            )
            test_ds_lyr.CreateFeature(feature.Clone())
            test_ds.SyncToDisk()

            clipped = internal_clip_raster(
                test_rast,
                test_ds_path,
                adjust_bbox=False,
                crop_to_geom=True,
                all_touch=False,
            )

            if clipped is None:
                raise Exception("Error while clipping raster. Likely a bad extraction.")

            ref_image = raster_to_array(clipped, filled=True)
            image_block = test_array[test]
            if not np.array_equal(ref_image, image_block):
                # from matplotlib import pyplot as plt; plt.imshow(ref_image[:,:,0]); plt.show()
                raise Exception(f"Image {basename} and grid cell did not match..")

            if verbose == 1:
                progress(tested, len(test_fids) - 1, "Verifying..")

            tested += 1

    return True


# TODO: Initial clip to extent of clip.
def extract_patches(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    out_dir: Optional[str] = None,
    prefix: str = "",
    postfix: str = "_patches",
    size: int = 32,
    offsets: Union[list, None] = [],
    generate_border_patches: bool = True,
    generate_zero_offset: bool = True,
    generate_grid_geom: bool = True,
    clip_geom: Optional[Union[str, ogr.DataSource, gdal.Dataset]] = None,
    clip_layer_index: int = 0,
    verify_output=True,
    verification_samples=100,
    overwrite=True,
    epsilon: float = 1e-9,
    verbose: int = 1,
) -> tuple:
    """ Extracts square tiles from a raster.
    Args:
        raster (list of rasters | path | raster): The raster(s) to convert.

    **kwargs:
        out_dir (path | none): Folder to save output. If None, in-memory
        arrays and geometries are outputted.

        prefix (str): A prefix for all outputs.

        postfix (str): A postfix for all outputs.

        size (int): The size of the tiles in pixels.

        offsets (list of tuples): List of offsets to extract. Example:
        offsets=[(16, 16), (16, 0), (0, 16)]. Will offset the initial raster
        and extract from there.

        generate_border_patches (bool): The tiles often do not align with the
        rasters which means borders are trimmed somewhat. If generate_border_patches
        is True, an additional tile is added where needed.

        generate_zero_offset (bool): if True, an offset is inserted at (0, 0)
        if none is present.

        generate_grid_geom (bool): Output a geopackage with the grid of tiles.
        
        clip_geom (str, raster, vector): Clip the output to the
        intersections with a geometry. Useful if a lot of the target
        area is water or similar.

        epsilon (float): How much for buffer the arange array function. This
        should usually just be left alone.

        verbose (int): If 1 will output messages on progress.

    Returns:
        A tuple with paths to the generated items. (numpy_array, grid_geom)
    """
    type_check(raster, [str, list, gdal.Dataset], "raster")
    type_check(out_dir, [str], "out_dir", allow_none=True)
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(size, [int], "size")
    type_check(offsets, [list], "offsets", allow_none=True)
    type_check(generate_grid_geom, [bool], "generate_grid_geom")
    type_check(
        clip_geom,
        [str, ogr.DataSource, gdal.Dataset],
        "clip_layer_index",
        allow_none=True,
    )
    type_check(clip_layer_index, [int], "clip_layer_index")
    type_check(overwrite, [bool], "overwrite")
    type_check(epsilon, [float], "epsilon")
    type_check(verbose, [int], "verbose")

    in_rasters = to_raster_list(raster)

    if out_dir is not None and not os.path.isdir(out_dir):
        raise ValueError(f"Output directory does not exists: {out_dir}")

    if not rasters_are_aligned(in_rasters):
        raise ValueError(
            "Input rasters must be aligned. Please use the align function."
        )

    output_geom = None

    metadata = internal_raster_to_metadata(in_rasters[0])

    if verbose == 1:
        print("Generating blocks..")

    # internal offset array. Avoid manipulating the og array.
    if offsets is None:
        offsets = []

    in_offsets = []
    if generate_zero_offset and (0, 0) not in offsets:
        in_offsets.append((0, 0))

    for offset in offsets:
        if offset != (0, 0):
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise ValueError(
                    f"offset must be a list or tuple of two integers. Recieved: {offset}"
                )
            in_offsets.append((offset[0], offset[1]))

    border_patches_needed_x = True
    border_patches_needed_y = True

    if clip_geom is not None:
        border_patches_needed_x = False
        border_patches_needed_y = False

    shapes = []
    for offset in in_offsets:
        block_shape = shape_to_blockshape(metadata["shape"], (size, size), offset)

        if block_shape[0] * size == metadata["width"]:
            border_patches_needed_x = False

        if block_shape[1] * size == metadata["height"]:
            border_patches_needed_y = False

        shapes.append(block_shape)

    if generate_border_patches:
        cut_x = (metadata["width"] - in_offsets[0][0]) - (shapes[0][0] * size)
        cut_y = (metadata["height"] - in_offsets[0][1]) - (shapes[0][1] * size)

        if border_patches_needed_x and cut_x > 0:
            shapes[0][0] += 1

        if border_patches_needed_y and cut_y > 0:
            shapes[0][1] += 1

    # calculate the offsets
    all_rows = 0
    offset_rows = []
    for i in range(len(shapes)):
        row = 0

        for j in range(len(shapes[i])):
            if j == 0:
                row = int(shapes[i][j])
            else:
                row *= int(shapes[i][j])

        offset_rows.append(row)
        all_rows += row

    offset_rows_cumsum = np.cumsum(offset_rows)

    if generate_grid_geom is True or clip_geom is not None:

        if verbose == 1:
            print("Calculating grid cells..")

        mask = np.arange(all_rows, dtype="uint64")

        ulx, uly, _lrx, _lry = metadata["extent"]

        pixel_width = abs(metadata["pixel_width"])
        pixel_height = abs(metadata["pixel_height"])

        xres = pixel_width * size
        yres = pixel_height * size

        dx = xres / 2
        dy = yres / 2

        # Ready clip geom outside of loop.
        if clip_geom is not None:
            clip_ref = open_vector(
                internal_reproject_vector(clip_geom, metadata["projection_osr"])
            )
            clip_layer = clip_ref.GetLayerByIndex(clip_layer_index)

            meta_clip = internal_vector_to_metadata(clip_ref)
            geom_clip = meta_clip["layers"][clip_layer_index]["column_geom"]

            clip_extent = meta_clip["extent_ogr"]
            clip_adjust = [
                clip_extent[0] - clip_extent[0] % xres,  # x_min
                (clip_extent[1] - clip_extent[1] % xres) + xres,  # x_max
                clip_extent[2] - clip_extent[2] % yres,  # y_min
                (clip_extent[3] - clip_extent[3] % yres) + yres,  # y_max
            ]

        coord_grid = np.empty((all_rows, 2), dtype="float64")

        # tiled_extent = [None, None, None, None]

        row_count = 0
        for idx in range(len(in_offsets)):
            x_offset = in_offsets[idx][0]
            y_offset = in_offsets[idx][1]

            x_step = shapes[idx][0]
            y_step = shapes[idx][1]

            x_min = (ulx + dx) + (x_offset * pixel_width)
            x_max = x_min + (x_step * xres)

            y_max = (uly - dy) - (y_offset * pixel_height)
            y_min = y_max - (y_step * yres)

            # if clip_geom is not None:
            #     if clip_adjust[0] > x_min:
            #         x_min = clip_adjust[0] + (x_offset * pixel_width)
            #     if clip_adjust[1] < x_max:
            #         x_max = clip_adjust[1] + (x_offset * pixel_width)
            #     if clip_adjust[2] > y_min:
            #         y_min = clip_adjust[2] - (y_offset * pixel_height)
            #     if clip_adjust[3] < y_max:
            #         y_max = clip_adjust[3] - (y_offset * pixel_height)

            # if idx == 0:
            #     tiled_extent[0] = x_min
            #     tiled_extent[1] = x_max
            #     tiled_extent[2] = y_min
            #     tiled_extent[3] = y_max
            # else:
            #     if x_min < tiled_extent[0]:
            #         tiled_extent[0] = x_min
            #     if x_max > tiled_extent[1]:
            #         tiled_extent[1] = x_max
            #     if y_min < tiled_extent[2]:
            #         tiled_extent[2] = y_min
            #     if y_max > tiled_extent[3]:
            #         tiled_extent[3] = y_max

            # y is flipped so: xmin --> xmax, ymax -- ymin to keep same order as numpy array
            x_patches = round((x_max - x_min) / xres)
            y_patches = round((y_max - y_min) / yres)

            xr = np.arange(x_min, x_max, xres)[0:x_step]
            if xr.shape[0] < x_patches:
                xr = np.arange(x_min, x_max + epsilon, xres)[0:x_step]
            elif xr.shape[0] > x_patches:
                xr = np.arange(x_min, x_max - epsilon, xres)[0:x_step]

            yr = np.arange(y_max, y_min + epsilon, -yres)[0:y_step]
            if yr.shape[0] < y_patches:
                yr = np.arange(y_max, y_min - epsilon, -yres)[0:y_step]
            elif yr.shape[0] > y_patches:
                yr = np.arange(y_max, y_min + epsilon, -yres)[0:y_step]

            if generate_border_patches and idx == 0:

                if border_patches_needed_x:
                    xr[-1] = xr[-1] - ((xr[-1] + dx) - metadata["extent_dict"]["right"])

                if border_patches_needed_y:
                    yr[-1] = yr[-1] - (
                        (yr[-1] - dy) - metadata["extent_dict"]["bottom"]
                    )

            oxx, oyy = np.meshgrid(xr, yr)
            oxr = oxx.ravel()
            oyr = oyy.ravel()

            offset_length = oxr.shape[0]

            coord_grid[row_count : row_count + offset_length, 0] = oxr
            coord_grid[row_count : row_count + offset_length, 1] = oyr

            row_count += offset_length

            offset_rows_cumsum[idx] = offset_length

        offset_rows_cumsum = np.cumsum(offset_rows_cumsum)
        coord_grid = coord_grid[:row_count]

        # Output geometry
        driver = ogr.GetDriverByName("GPKG")
        patches_path = f"/vsimem/patches_{uuid4().int}.gpkg"
        patches_ds = driver.CreateDataSource(patches_path)
        patches_layer = patches_ds.CreateLayer(
            "patches_all", geom_type=ogr.wkbPolygon, srs=metadata["projection_osr"]
        )
        patches_fdefn = patches_layer.GetLayerDefn()

        og_fid = "og_fid"

        field_defn = ogr.FieldDefn(og_fid, ogr.OFTInteger)
        patches_layer.CreateField(field_defn)

        if clip_geom is not None:
            clip_feature_count = meta_clip["layers"][clip_layer_index]["feature_count"]
            spatial_index = rtree.index.Index(interleaved=False)
            for _ in range(clip_feature_count):
                clip_feature = clip_layer.GetNextFeature()
                clip_fid = clip_feature.GetFID()
                clip_feature_geom = clip_feature.GetGeometryRef()
                xmin, xmax, ymin, ymax = clip_feature_geom.GetEnvelope()

                spatial_index.insert(clip_fid, (xmin, xmax, ymin, ymax))

        fids = 0
        mask = []
        for tile_id in range(coord_grid.shape[0]):
            x, y = coord_grid[tile_id]

            if verbose == 1:
                progress(tile_id, coord_grid.shape[0], "Patch generation")

            x_min = x - dx
            x_max = x + dx
            y_min = y - dx
            y_max = y + dx

            tile_intersects = True

            grid_geom = None
            poly_wkt = None

            if clip_geom is not None:
                tile_intersects = False

                if not ogr_bbox_intersects([x_min, x_max, y_min, y_max], clip_extent):
                    continue

                intersections = list(
                    spatial_index.intersection((x_min, x_max, y_min, y_max))
                )
                if len(intersections) == 0:
                    continue

                poly_wkt = f"POLYGON (({x_min} {y_max}, {x_max} {y_max}, {x_max} {y_min}, {x_min} {y_min}, {x_min} {y_max}))"
                grid_geom = ogr.CreateGeometryFromWkt(poly_wkt)

                for fid1 in intersections:
                    clip_feature = clip_layer.GetFeature(fid1)
                    clip_geom = clip_feature.GetGeometryRef()

                    if grid_geom.Intersects(clip_geom):
                        tile_intersects = True
                        continue

            if tile_intersects:
                ft = ogr.Feature(patches_fdefn)

                if grid_geom is None:
                    poly_wkt = f"POLYGON (({x_min} {y_max}, {x_max} {y_max}, {x_max} {y_min}, {x_min} {y_min}, {x_min} {y_max}))"
                    grid_geom = ogr.CreateGeometryFromWkt(poly_wkt)

                ft_geom = ogr.CreateGeometryFromWkt(poly_wkt)
                ft.SetGeometry(ft_geom)

                ft.SetField(og_fid, int(fids))
                ft.SetFID(int(fids))

                patches_layer.CreateFeature(ft)
                ft = None

                mask.append(tile_id)
                fids += 1

        if verbose == 1:
            progress(coord_grid.shape[0], coord_grid.shape[0], "Patch generation")

        mask = np.array(mask, dtype=int)

        if generate_grid_geom is True:
            if out_dir is None:
                output_geom = patches_ds
            else:
                raster_basename = metadata["basename"]
                geom_name = f"{prefix}{raster_basename}_geom_{str(size)}{postfix}.gpkg"
                output_geom = os.path.join(out_dir, geom_name)

                overwrite_required(output_geom, overwrite)
                remove_if_overwrite(output_geom, overwrite)

                if verbose == 1:
                    print("Writing output geometry..")

                internal_vector_to_disk(patches_ds, output_geom, overwrite=overwrite)

    if verbose == 1:
        print("Writing numpy array..")

    output_blocks = []

    for raster in in_rasters:

        base = None
        basename = None
        output_block = None

        if out_dir is not None:
            base = os.path.basename(raster)
            basename = os.path.splitext(base)[0]
            output_block = os.path.join(out_dir + f"{prefix}{basename}{postfix}.npy")

        metadata = internal_raster_to_metadata(raster)

        if generate_grid_geom is True or clip_geom is not None:
            output_shape = (row_count, size, size, metadata["band_count"])
        else:
            output_shape = (all_rows, size, size, metadata["band_count"])

        input_datatype = metadata["datatype"]

        output_array = np.empty(output_shape, dtype=input_datatype)

        # if clip_geom is not None:
        #     ref = raster_to_array(raster, filled=True, extent=tiled_extent)
        # else:
        ref = raster_to_array(raster, filled=True)

        for k, offset in enumerate(in_offsets):
            start = 0
            if k > 0:
                start = offset_rows_cumsum[k - 1]

            blocks = None
            if (
                k == 0
                and generate_border_patches
                and (border_patches_needed_x or border_patches_needed_y)
            ):
                blocks = array_to_blocks(
                    ref,
                    (size, size),
                    offset,
                    border_patches_needed_x,
                    border_patches_needed_y,
                )
            else:
                blocks = array_to_blocks(ref, (size, size), offset)

            output_array[start : offset_rows_cumsum[k]] = blocks

        if generate_grid_geom is False and clip_geom is None:
            if out_dir is None:
                output_blocks.append(output_array)
            else:
                output_blocks.append(output_block)
                np.save(output_block, output_array)
        else:
            if out_dir is None:
                output_blocks.append(output_array[mask])
            else:
                output_blocks.append(output_block)
                np.save(output_block, output_array[mask])

    if verify_output and generate_grid_geom:
        test_extraction(
            in_rasters,
            output_blocks,
            output_geom,
            samples=verification_samples,
            grid_layer_index=0,
            verbose=verbose,
        )

    if len(output_blocks) == 1:
        output_blocks = output_blocks[0]

    return (output_blocks, output_geom)


# TODO: Create input option
def predict_raster(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    model: str,
    out_path: Optional[str] = None,
    offsets: Union[List[Tuple[int, int]], List[List[Tuple[int, int]]]] = [(0, 0)],
    region: Optional[Union[str, ogr.DataSource]] = None,
    device: str = "gpu",
    merge_method: str = "median",
    mirror: bool = False,
    rotate: bool = False,
    custom_objects: Dict[str, Any] = {},
    dtype: str = "same",
    batch_size: int = 16,
    overwrite: bool = True,
    creation_options: List[str] = [],
    verbose: int = 1,
) -> str:
    """ Runs a raster or list of rasters through a deep learning network (Tensorflow).
        Supports tiling and reconstituting the output. Offsets are allowed and will be
        bleneded with the merge_method. If the output is a different resolution
        than the input. The output will automatically be scaled to match.
    Args:
        raster (list | path | raster): The raster(s) to convert.

        model (path): A path to the tensorflow .h5 model.

    **kwargs:
        out_path (str | None): Where to save the reconstituted raster. If None
        are memory raster is returned.

        offsets (tuple, list, ndarray): The offsets used in the original. A (0 ,0)
        offset is assumed.

        border_patches (bool): Do the blocks contain border patches?

        device (str): Either CPU or GPU to use with tensorflow.

        merge_method (str): How to handle overlapping pixels. Options are:
        median, average, mode, min, max

        mirror (bool): Mirror the raster and do predictions as well.

        rotate (bool): rotate the raster and do predictions as well.

        dtype (str | None): The dtype of the output. If None: Float32, "save"
        is the same as the input raster. Otherwise overwrite dtype.

        overwrite (bool): Overwrite output files if they exists.

        creation_options: Extra creation options for the output raster.

        verbose (int): If 1 will output messages on progress.

    Returns:
        A predicted raster.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(model, [str], "model")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(offsets, [list], "offsets")
    type_check(region, [str, ogr.DataSource], allow_none=True)
    type_check(device, [str], "device")
    type_check(merge_method, [str], "merge_method")
    type_check(mirror, [bool], "mirror")
    type_check(rotate, [bool], "rotate")
    type_check(custom_objects, [dict], "custom_objects")
    type_check(dtype, [str], "rotate", allow_none=True)
    type_check(batch_size, [int], "batch_size")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")
    type_check(verbose, [int], "verbose")

    if mirror or rotate:
        raise Exception("Mirror and rotate currently disabled.")

    import tensorflow as tf

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    if verbose == 1:
        print("Loading model.")

    model_loaded = tf.keras.models.load_model(model, custom_objects=custom_objects)

    multi_input = False
    if isinstance(model_loaded.input, list) and len(model_loaded.input) > 1:
        if not isinstance(raster, list):
            raise TypeError("Multi input model must have a list as input.")

        if len(offsets) > 0:
            for offset in offsets:
                if not isinstance(offset, list):
                    raise TypeError(
                        "Offsets must be a list of tuples, same length as inputs."
                    )

                for _offset in offset:
                    if not isinstance(_offset, tuple):
                        raise TypeError("Offset must be a tuple")

                    if len(_offset) != 2:
                        raise ValueError("Offset must be length 2.")

            if len(model_loaded.input) != len(offsets):
                raise ValueError("Length of offsets must equal model inputs.")

        multi_input = True

    model_inputs = (
        model_loaded.input
        if isinstance(model_loaded.input, list)
        else [model_loaded.input]
    )
    shape_output = tuple(model_loaded.output.shape)
    dst_tile_size = shape_output[1]

    prediction_arr = []
    readied_inputs = []
    pixel_factor = 1.0
    for index, model_input in enumerate(model_inputs):
        if verbose == 1:
            print(f"Readying input: {index}")

        shape_input = tuple(model_input.shape)

        if len(shape_input) != 4 or len(shape_output) != 4:
            raise ValueError(f"Model input not 4d: {shape_input} - {shape_output}")

        if shape_input[1] != shape_input[2] or shape_output[1] != shape_output[2]:
            raise ValueError("Model does not take square images.")

        src_tile_size = shape_input[1]
        pixel_factor = src_tile_size / dst_tile_size
        scale_factor = dst_tile_size / src_tile_size

        dst_offsets = []

        in_offsets: List[Tuple[Number, Number]] = []
        if multi_input:
            if len(offsets) > 0:
                in_offsets = offsets[index]
        else:
            in_offsets = offsets

        for offset in in_offsets:
            if not isinstance(offset, tuple):
                raise ValueError(
                    f"Offset must be a tuple of two ints. Recieved: {offset}"
                )
            if len(offset) != 2:
                raise ValueError("Offsets must have two values. Both integers.")

            dst_offsets.append(
                (round(offset[0] * scale_factor), round(offset[1] * scale_factor),)
            )

        use_raster = raster[index] if isinstance(raster, list) else raster

        if region is not None:
            use_raster = clip_raster(use_raster, region)

        blocks, _ = extract_patches(
            use_raster,
            size=src_tile_size,
            offsets=in_offsets,
            generate_border_patches=True,
            generate_grid_geom=False,
            verbose=verbose,
        )

        readied_inputs.append(blocks)

    first_len = None
    for index, readied in enumerate(readied_inputs):
        if index == 0:
            first_len = readied.shape[0]
        else:
            if readied.shape[0] != first_len:
                raise ValueError(
                    "Length of inputs do not match. Have you set the offsets in the correct order?"
                )

    if verbose == 1:
        print("Predicting raster.")

    start = 0
    end = readied_inputs[0].shape[0]

    predictions = np.empty(
        (end, dst_tile_size, dst_tile_size, shape_output[3]), dtype="float32"
    )

    if multi_input is False:
        if device == "cpu":
            with tf.device("/cpu:0"):
                while start < end:
                    predictions[
                        start : start + batch_size
                    ] = model_loaded.predict_on_batch(
                        readied_inputs[0][start : start + batch_size]
                    )
                    start += batch_size
                    progress(start, end - 1, "Predicting")
        else:
            while start < end:
                predictions[start : start + batch_size] = model_loaded.predict_on_batch(
                    readied_inputs[0][start : start + batch_size]
                )
                start += batch_size
                progress(start, end - 1, "Predicting")
    else:
        if device == "cpu":
            with tf.device("/cpu:0"):
                while start < end:
                    batch = []
                    for i in range(len(readied_inputs)):
                        batch.append(readied_inputs[i][start : start + batch_size])
                    predictions[
                        start : start + batch_size
                    ] = model_loaded.predict_on_batch(batch)
                    start += batch_size
                    progress(start, end - 1, "Predicting")
        else:
            while start < end:
                batch = []
                for i in range(len(readied_inputs)):
                    batch.append(readied_inputs[i][start : start + batch_size])
                predictions[start : start + batch_size] = model_loaded.predict_on_batch(
                    batch
                )
                start += batch_size
                progress(start, end - 1, "Predicting")
    print("")
    print("Reconstituting Raster.")

    rast_meta = None
    target_size = None
    resampled = None
    if isinstance(raster, list):
        rast_meta = internal_raster_to_metadata(raster[-1])
        target_size = (
            rast_meta["pixel_width"] * pixel_factor,
            rast_meta["pixel_height"] * pixel_factor,
        )
        resampled = internal_resample_raster(
            raster[-1], target_size=target_size, dtype="float32"
        )

    else:
        rast_meta = internal_raster_to_metadata(raster)
        target_size = (
            rast_meta["pixel_width"] * pixel_factor,
            rast_meta["pixel_height"] * pixel_factor,
        )
        resampled = internal_resample_raster(
            raster, target_size=target_size, dtype="float32"
        )

    if region is not None:
        resampled = clip_raster(resampled, region)

    prediction_arr.append(
        blocks_to_raster(
            predictions,
            resampled,
            border_patches=True,
            offsets=dst_offsets,
            merge_method=merge_method,
            output_array=True,
        )
    )

    if verbose == 1:
        print("Merging rasters.")

    if merge_method == "median":
        predicted = np.median(prediction_arr, axis=0)
    elif merge_method == "mean" or merge_method == "average":
        predicted = np.mean(prediction_arr, axis=0)
    elif merge_method == "min" or merge_method == "minumum":
        predicted = np.min(prediction_arr, axis=0)
    elif merge_method == "max" or merge_method == "maximum":
        predicted = np.max(prediction_arr, axis=0)
    elif merge_method == "mode" or merge_method == "majority":
        for index, _ in enumerate(prediction_arr):
            prediction_arr[index] = np.rint(prediction_arr[index]).astype(int)

        predicted = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=prediction_arr
        )
    else:
        raise ValueError(f"Unable to parse merge_method: {merge_method}")

    if dtype == "same":
        predicted = array_to_raster(
            predicted.astype(rast_meta["datatype"]), reference=resampled,
        )
    elif dtype is not None:
        predicted = array_to_raster(
            raster_to_array(predicted).astype(dtype), reference=resampled,
        )

    if out_path is None:
        return predicted
    else:
        return internal_raster_to_disk(
            predicted,
            out_path=out_path,
            overwrite=overwrite,
            creation_options=creation_options,
        )


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"

    # raster_to_predict = folder + "Fyn_B2_20m.tif"
    vector = folder + "fjord.gpkg"
    raster = folder + "B08_10m.jp2"
    out_dir = folder + "out/"
    # tensorflow_model_path = out_dir + "model.h5"

    path_np, path_geom = extract_patches(
        raster,
        out_dir=out_dir,
        prefix="",
        postfix="_patches",
        size=32,
        offsets=[(8, 8), (16, 16), (24, 24)],
        generate_grid_geom=True,
        generate_zero_offset=True,
        generate_border_patches=True,
        clip_geom=vector,
        verify_output=True,
        verification_samples=100,
        verbose=1,
    )

    # offsets = [(64, 64), (64, 0), (0, 64)]
    # borders = True

    # path = predict_raster(
    #     raster_to_predict,
    #     tensorflow_model_path,
    #     out_path=out_dir + "predicted_raster_32-16.tif",
    #     offsets=[(32, 32), (16, 16)],
    #     mirror=True,
    #     rotate=True,
    #     device="gpu",
    # )

    # B03_10m = folder + "B03_10m.jp2"
    # B04_10m = folder + "B04_10m.jp2"
    # B04_20m = folder + "B04_20m.jp2"
    # B08_10m = folder + "B08_10m.jp2"
    # B11_20m = folder + "B11_20m.jp2"

    # model = folder + "upsampling_10epochs.h5"

    # blocks_to_raster(
    #     path_np,
    #     raster,
    #     out_path=out_dir + "fyn_close_reconstituded.tif",
    #     offsets=offsets,
    #     border_patches=borders,
    #     merge_method="median",
    # )

    # dsm = folder + "dsm_test_clip.tif"
    # dtm = folder + "dtm_test_clip.tif"
    # hot = folder + "hot_test_clip.tif"
    # model = folder + "model3_3L_rotations.h5"

    # stacked = stack_rasters([dtm, dsm, hot], folder + "dtm_dsm_hot_stacked.tif") # shape = (14400, 26112, 3)

    # from tensorflow_addons.activations import mish

    # 2m 17s
    # 0m 50s

    # path = predict_raster(
    #     [B11_20m, B08_10m],
    #     model,
    #     out_path=out_dir + "validation.tif",
    #     offsets=[[(16, 16), (8, 8)], [(32, 32), (16, 16)]],
    #     batch_size=64,
    #     mirror=False,
    #     rotate=False,
    #     device="gpu",
    #     custom_objects={"mish": mish},
    # )
