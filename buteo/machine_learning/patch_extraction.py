import sys;  sys.path.append("../../")
import numpy as np
import os
import random
from typing import Union
from osgeo import ogr, gdal
from uuid import uuid1
from buteo.raster.io import (
    raster_to_array,
    raster_to_metadata,
    raster_to_memory,
    array_to_raster,
    raster_in_memory,
)
from buteo.vector.io import (
    vector_to_memory,
    vector_to_reference,
    vector_to_metadata,
    vector_to_disk,
    vector_in_memory,
)
from buteo.raster.clip import clip_raster
from buteo.raster.align import is_aligned
from buteo.vector.reproject import reproject_vector
from buteo.utils import overwrite_required, remove_if_overwrite, progress, type_check
from buteo.gdal_utils import to_array_list, to_raster_list, ogr_bbox_intersects


def reconstitute_raster(
    blocks: Union[str, np.ndarray],
    raster_height: int,
    raster_width: int,
    size: int,
    offset: tuple,
    border_patches: bool,
    border_patches_x: bool,
    border_patches_y: bool,
) -> Union[str, gdal.Dataset]:
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
        (ref_shape[0] // size) * size,
        (ref_shape[1] // size) * size,
        blocks.shape[3],
    )

    # Order: Y, X, Z
    if border_patches and (border_patches_x or border_patches_y):

        if border_patches_x:
            x_offset = ref_shape[1] - raster_width
            x_edge = destination[:raster_height, -(size - x_offset):, :]
            destination[:raster_height, -size:-x_offset, :] = x_edge

        if border_patches_y:
            y_offset = ref_shape[0] - raster_height
            y_edge = destination[-(size - y_offset):, :raster_width, :]
            destination[-size:-y_offset, :raster_width, :] = y_edge

        if border_patches_y and border_patches_y:
            corner = destination[-(size - y_offset):, -(size - x_offset):, :]
            destination[-size:-y_offset, -size:-x_offset, :] = corner

        destination = destination[ : raster_height, : raster_width, 0 : blocks.shape[3]]
    
    return destination


# TODO: reconstitute offsets.
def blocks_to_raster(
    blocks: Union[str, np.ndarray],
    reference: Union[str, gdal.Dataset],
    out_path: Union[str, None]=None,
    offsets: Union[list, tuple, np.ndarray, None]=None,
    border_patches: bool=True,
    generate_zero_offset: bool=True,
    merge_method: str="median",
    verbose: int=1,
) -> Union[str, gdal.Dataset]:
    """ Recombines a series of blocks to a raster.
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

        verbose (int): If 1 will output messages on progress.

    Returns:
        A reconstituted raster.
    """
    type_check(blocks, [str, np.ndarray], "blocks")
    type_check(reference, [str, gdal.Dataset], "reference")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(offsets, [list, tuple, np.ndarray], "offsets", allow_none=True)
    type_check(border_patches, [bool], "border_patches")
    type_check(merge_method, [str], "merge_method")

    if isinstance(blocks, str):
        try:
            blocks = np.load(blocks)
        except:
            raise ValueError(f"Failed to parse blocks: {blocks}")

    if verbose == 1:
        print("Reconstituting blocks into target raster.")

    metadata = raster_to_metadata(reference)

    border_patches_x = False
    border_patches_y = False

    if blocks.shape[1] != blocks.shape[2]:
        raise ValueError("The input blocks must be square. Rectangles might supported in the future.")
    
    size = blocks.shape[1]

    if metadata["width"] % size != 0 and border_patches:
        border_patches_x = True
    if metadata["height"] % size != 0 and border_patches:
        border_patches_y = True

    # internal offset array. Avoid manipulating the og array.
    in_offsets = []
    if generate_zero_offset and (0, 0) not in offsets:
        in_offsets.append((0, 0))
    
    for offset in offsets:
        if offset != (0, 0):
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise ValueError(f"offset must be a list or tuple of two integers. Recieved: {offset}")
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
                np.ma.masked_all((
                    metadata["height"],
                    metadata["width"],
                    blocks.shape[3],
                ),
                dtype=metadata["dtype"],
                ),
            )
            
            if index == 0:
                x_blocks = ((metadata["width"] - offset[0]) // size) + border_patches_x
                y_blocks = ((metadata["height"] - offset[1]) // size) + border_patches_x
            else:
                x_blocks = (metadata["width"] - offset[0]) // size
                y_blocks = (metadata["height"] - offset[1]) // size

            block_size = x_blocks * y_blocks

            raster_pass = reconstitute_raster( #pylint: disable=too-many-function-args
                blocks[previous:block_size + previous, :, :, :],
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
            raster = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=passes)
        else:
            raise ValueError(f"Unable to parse merge_method: {merge_method}")

    else:
        raster = reconstitute_raster( #pylint: disable=too-many-function-args
            blocks,
            metadata["height"],
            metadata["width"],
            size,
            (0, 0),
            border_patches,
            border_patches_x,
            border_patches_y,
        )

    return array_to_raster(
        raster,
        reference,
        out_path=out_path,
    )


def shape_to_blockshape(
    shape: Union[list, tuple, np.ndarray],
    block_shape: Union[list, tuple, np.ndarray],
    offset: Union[list, tuple, np.ndarray],
) -> list:
    """ Calculates the shape of the output array.
    Args:
        shape (tuple | list): The shape if the original raster.

        block_shape (tuple | list): The size of the blocks eg. (64, 64)

        offset (tuple, list): An initial offset for the array eg. (32, 32)

    Returns:
        A list with the modified shape.
    """
    type_check(shape, [list, tuple, np.ndarray], "shape")
    type_check(block_shape, [list, tuple, np.ndarray], "block_shape")
    type_check(offset, [list, tuple, np.ndarray], "offset")

    assert len(offset) >= len(shape), "Input offsets must equal array dimensions."
    assert len(shape) <= 3, "Unable to handle more than 3 dimensions."

    base_shape = list(shape)
    for index, value in enumerate(offset):
        base_shape[index] = base_shape[index] - value

    sizes = []

    for index, value in enumerate(base_shape):
        sizes.append(value // block_shape[index])

    return sizes


def array_to_blocks(
    array: np.ndarray,
    block_shape: Union[list, tuple, np.ndarray],
    offset: Union[list, tuple, np.ndarray],
    border_patches_x: bool=False,
    border_patches_y: bool=False,
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
    type_check(block_shape, [list, tuple], "block_shape")
    type_check(offset, [list, tuple], "offset")
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
            x_shape = ((array.shape[1] // block_shape[0]) * block_shape[0]) + block_shape[0]
        
        if border_patches_y:
            y_shape = ((array.shape[0] // block_shape[1]) * block_shape[1]) + block_shape[1]

        # Expand and copy the original array
        arr_exp = np.empty((y_shape, x_shape, array.shape[2]), dtype=array.dtype)
        arr_exp[0:arr.shape[0], 0:arr.shape[1], 0:arr.shape[2]] = arr

        if border_patches_x:
            arr_exp[:array.shape[0], -block_shape[1]:, :] = array[:, -block_shape[0]:, :]
        
        if border_patches_y:
            arr_exp[-block_shape[0]:, :array.shape[1], :] = array[-block_shape[1]:, :, :]

        # The bottom right corner will still have empty values if both a True
        if border_patches_x and border_patches_y:
            border_square = array[-block_shape[0]:, -block_shape[1]:, :]
            arr_exp[-block_shape[0]:, -block_shape[1]:, :] = border_square

        arr = arr_exp

    # This only creates views into the array, so should still be fast.
    reshaped = arr.reshape(
        arr.shape[0] // block_shape[0], block_shape[0],
        arr.shape[1] // block_shape[1], block_shape[1],
        arr.shape[2], arr.shape[2],
    )
    swaped = reshaped.swapaxes(1, 2)
    merge = swaped.reshape(
        -1,
        block_shape[0],
        block_shape[1],
        array.shape[2],
    )

    return merge


def test_extraction(
    rasters: Union[list, str, gdal.Dataset],
    arrays: Union[list, np.ndarray],
    grid: Union[ogr.DataSource, str],
    samples: int=1000, # if 0, all
    grid_layer_index: int=0,
    verbose: int=1,
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
    type_check(grid, [list, str, np.ndarray], "grid")
    type_check(samples, [int], "samples")
    type_check(grid_layer_index, [int], "clip_layer_index")
    type_check(verbose, [int], "verbose")

    in_rasters = to_raster_list(rasters)
    in_arrays = to_array_list(arrays)

    if verbose == 1:
        print("Verifying integrity of output grid..")

    grid_memory = None
    if vector_in_memory(grid):
        grid_memory = grid
    else:
        grid_memory = vector_to_memory(
            grid,
            memory_path=f"/vsimem/grid_{uuid1().int}.gpkg",
            opened=True
        )

    grid_metadata = vector_to_metadata(grid, latlng_and_footprint=False)
    grid_projection = grid_metadata["projection_osr"]

    if grid_layer_index > (grid_metadata["layer_count"] - 1):
        raise ValueError(f"Requested non-existing layer index: {grid_layer_index}")

    grid_layer = grid_memory.GetLayer(grid_layer_index)

    # Select sample fids
    feature_count = grid_metadata["layers"][grid_layer_index]["feature_count"]
    test_samples = samples if samples > 0 else feature_count
    max_test = min(test_samples, feature_count) - 1
    test_fids = np.array(random.sample(range(0, feature_count), max_test), dtype="uint64")

    mem_driver = ogr.GetDriverByName("Memory")
    for index, raster in enumerate(in_rasters):
        test_rast = None
        if raster_in_memory(raster):
            test_rast = raster
        else:
            test_rast = raster_to_memory(
                raster,
                memory_path=f"/vsimem/raster_{uuid1().int}.tif",
                opened=True,
            )

        test_array = in_arrays[index]
        if isinstance(test_array, str):
            if not os.path.exists(test_array):
                raise ValueError(f"Numpy array does not exist: {test_array}")

            try:
                test_array = np.load(in_arrays[index])
            except:
                raise Exception(f"Attempted to read numpy raster from: {in_arrays[index]}")

        base = os.path.basename(raster)
        basename = os.path.splitext(base)[0]

        if verbose == 1:
            print(f"Testing: {basename}")

        tested = 0
        for test in test_fids:
            feature = grid_layer.GetFeature(test)

            if feature is None:
                raise Exception(f"Feature not found: {test}")

            test_ds = mem_driver.CreateDataSource("test_mem_grid.gpkg")
            test_ds_lyr = test_ds.CreateLayer(
                "test_mem_grid_layer",
                geom_type=ogr.wkbPolygon,
                srs=grid_projection
            )
            test_ds_lyr.CreateFeature(feature.Clone())

            clipped = clip_raster(
                test_rast,
                test_ds,
                adjust_bbox=False,
                crop_to_geom=True,
                all_touch=False,
            )
            ref_image = raster_to_array(clipped, filled=True)
            image_block = test_array[test]

            if not np.array_equal(ref_image, image_block):
                raise Exception(f"Image {basename} and grid cell did not match..")

            if verbose == 1:
                progress(tested, len(test_fids) - 1, "verifying..")

            tested += 1

    return True


def extract_patches(
    raster: Union[str, list, gdal.Dataset],
    output_folder: str,
    prefix: str="",
    postfix: str="_patches",
    size: int=32,
    offsets: list=[],
    generate_border_patches: bool=True,
    generate_zero_offset: bool=True,
    generate_grid_geom: bool=True,
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset, None]=None,
    clip_layer_index: int=0,
    verify_output=False,
    verification_samples=100,
    overwrite=True,
    epsilon: float=1e-9,
    verbose: int=1,
) -> tuple:
    """ Extracts square tiles from a raster.
    Args:
        raster (list of rasters | path | raster): The raster(s) to convert.

        output_folder (path): The path of the output folder.

    **kwargs:
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
    type_check(raster, [str, list, gdal.Dataset], "in_rasters")
    type_check(output_folder, [str], "output_folder")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(size, [int], "size")
    type_check(offsets, [list], "offsets")
    type_check(generate_grid_geom, [bool], "generate_grid_geom")
    type_check(clip_geom, [str, ogr.DataSource, gdal.Dataset], "clip_layer_index", allow_none=True)
    type_check(clip_layer_index, [int], "clip_layer_index")
    type_check(overwrite, [bool], "overwrite")
    type_check(epsilon, [float], "epsilon")
    type_check(verbose, [int], "verbose")

    in_rasters = to_raster_list(raster)

    if not os.path.isdir(output_folder):
        raise ValueError(f"Output folder does not exists: {output_folder}")

    if not is_aligned(in_rasters):
        raise ValueError("Input rasters must be aligned. Please use the align function.")

    out_path = None
    out_path_geom = None

    metadata = raster_to_metadata(in_rasters[0])

    if verbose == 1:
        print("Generating blocks..")

    # internal offset array. Avoid manipulating the og array.
    in_offsets = []
    if generate_zero_offset and (0, 0) not in offsets:
        in_offsets.append((0, 0))
    
    for offset in offsets:
        if offset != (0, 0):
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise ValueError(f"offset must be a list or tuple of two integers. Recieved: {offset}")
            in_offsets.append((offset[0], offset[1]))

    border_patches_needed_x = True
    border_patches_needed_y = True

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

        geo_fid = np.zeros(all_rows, dtype="uint64")
        mask = geo_fid

        ulx, uly, _lrx, _lry = metadata["extent"]

        pixel_width = abs(metadata["pixel_width"])
        pixel_height = abs(metadata["pixel_height"])

        xres = pixel_width * size
        yres = pixel_height * size

        dx = xres / 2
        dy = yres / 2

        coord_grid = np.empty((all_rows, 2), dtype="float64")

        for l in range(len(in_offsets)):
            x_offset = in_offsets[l][0]
            y_offset = in_offsets[l][1]

            x_step = shapes[l][0]
            y_step = shapes[l][1]

            x_min = (ulx + dx) + (x_offset * pixel_width)
            x_max = x_min + (x_step * xres)

            y_max = (uly - dy) - (y_offset * pixel_height)
            y_min = y_max - (y_step * yres)

            # y is flipped so: xmin --> xmax, ymax -- ymin to keep same order as numpy array
            xr = np.arange(x_min, x_max + epsilon, xres)[0:x_step]
            yr = np.arange(y_max, y_min + epsilon, -yres)[0:y_step]

            if generate_border_patches and l == 0:

                if border_patches_needed_x:
                    xr[-1] = xr[-1] - ((xr[-1] + dx) - metadata["extent_dict"]["right"])
                
                if border_patches_needed_y:
                    yr[-1] = yr[-1] - ((yr[-1] - dy) - metadata["extent_dict"]["bottom"])

            oxx, oyy = np.meshgrid(xr, yr)

            start = 0
            if l > 0:
                start = offset_rows_cumsum[l - 1]

            coord_grid[start:offset_rows_cumsum[l], 0] = oxx.ravel()
            coord_grid[start:offset_rows_cumsum[l], 1] = oyy.ravel()

        if clip_geom is not None:
            clip_vector = vector_to_reference(clip_geom)
            clip_vector_metadata = vector_to_metadata(clip_vector, latlng_and_footprint=False)

            if clip_layer_index > (clip_vector_metadata["layer_count"] - 1):
                raise ValueError("Requested a layer not present in the clip_vector.")

            clip_vector_memory = None

            if not metadata["projection_osr"].IsSame(clip_vector_metadata["projection_osr"]):
                clip_vector_memory = reproject_vector(
                    clip_vector, metadata["projection_osr"],
                    out_path=f"/vsimem/clip_vect_{uuid1().int}.gpkg",
                )
                clip_vector_memory = vector_to_reference(clip_vector_memory)

            if clip_vector_memory is None:
                if vector_in_memory(clip_vector):
                    clip_vector_memory = clip_vector
                else:
                    clip_vector_memory = vector_to_memory(
                        clip_vector_memory,
                        memory_path=f"clip_vect_{uuid1().int}.gpkg",
                        opened=True,
                    )
  
            clip_layer = clip_vector_memory.GetLayer(clip_layer_index)
            clip_layer_extent = clip_layer.GetExtent() # x_min, x_max, y_min, y_max
            clip_feature_count = clip_vector_metadata["layers"][clip_layer_index]["feature_count"]

        # Output geometry
        driver = ogr.GetDriverByName("GPKG")
        patches_ds = driver.CreateDataSource(f"/vsimem/patches_{uuid1().int}.gpkg")
        patches_layer = patches_ds.CreateLayer("patches", geom_type=ogr.wkbPolygon, srs=metadata["projection_osr"])
        patches_fdefn = patches_layer.GetLayerDefn()

        if verbose == 1:
            print("Creating patches..")

        valid_fid = -1
        for q in range(all_rows):
            x, y = coord_grid[q]

            tile_intersects_geom = False

            if verbose == 1:
                progress(q, all_rows, "Patches")

            if clip_geom is not None:

                clip_layer.ResetReading()

                tile_bounds = (x - dx, x + dx, y - dy, y + dy)

                if not ogr_bbox_intersects(tile_bounds, clip_layer_extent):
                    continue

                tile_ring = ogr.Geometry(ogr.wkbLinearRing)
                tile_ring.AddPoint(x - dx, y + dy) # ul
                tile_ring.AddPoint(x + dx, y + dy) # ur
                tile_ring.AddPoint(x + dx, y - dy) # lr 
                tile_ring.AddPoint(x - dx, y - dy) # ll
                tile_ring.AddPoint(x - dx, y + dy) # ul begin

                tile_poly = ogr.Geometry(ogr.wkbPolygon)
                tile_poly.AddGeometry(tile_ring)

                for _ in range(clip_feature_count):
                    clip_feature = clip_layer.GetNextFeature()
                    clip_geometry = clip_feature.GetGeometryRef()

                    if tile_poly.Intersects(clip_geometry):
                        valid_fid += 1
                        tile_intersects_geom = True
                        break

            else:
                valid_fid += 1
                
            if (generate_grid_geom is True and clip_geom is None) or (generate_grid_geom is True and clip_geom is not None and tile_intersects_geom is True):

                poly_wkt = f"POLYGON (({x - dx} {y + dy}, {x + dx} {y + dy}, {x + dx} {y - dy}, {x - dx} {y - dy}, {x - dx} {y + dy}))"

                ft = ogr.Feature(patches_fdefn)
                ft.SetGeometry(ogr.CreateGeometryFromWkt(poly_wkt))
                ft.SetFID(valid_fid)

                geo_fid[valid_fid] = q

                patches_layer.CreateFeature(ft)
                ft = None

        if verbose == 1:
            progress(all_rows, all_rows, "Patches")

        # Create mask for numpy arrays
        mask = geo_fid[0:int(valid_fid + 1)]

        if generate_grid_geom is True:
            if valid_fid == -1:
                print("WARNING: Empty geometry output")

            raster_basename = metadata["basename"]
            geom_name = f"{prefix}{raster_basename}_geom_{str(size)}{postfix}.gpkg"
            out_path_geom = os.path.join(output_folder, geom_name)

            overwrite_required(out_path_geom, overwrite)
            remove_if_overwrite(out_path_geom, overwrite)

            if verbose == 1:
                print("Writing output geometry..")

            vector_to_disk(patches_ds, out_path_geom, overwrite=overwrite)

    if verbose == 1:
        print("Writing numpy array to disc..")

    # Generate some numpy arrays
    for raster in in_rasters:
        base = os.path.basename(raster)
        basename = os.path.splitext(base)[0]
        out_path = os.path.join(output_folder + f"{prefix}{basename}{postfix}.npy")

        metadata = raster_to_metadata(raster)

        output_shape = (all_rows, size, size, metadata["bands"])
        input_datatype = metadata["dtype"]

        output_array = np.empty(output_shape, dtype=input_datatype)

        ref = raster_to_array(raster, filled=True)

        for k, offset in enumerate(in_offsets):
            start = 0
            if k > 0:
                start = offset_rows_cumsum[k - 1]
            
            blocks = None
            if k == 0 and generate_border_patches and (border_patches_needed_x or border_patches_needed_y):
                blocks = array_to_blocks(ref, (size, size), offset, border_patches_needed_x, border_patches_needed_y)
            else:
                blocks = array_to_blocks(ref, (size, size), offset)

            output_array[start:offset_rows_cumsum[k]] = blocks

        if generate_grid_geom is False and clip_geom is None:
            np.save(out_path, output_array)
        else:
            np.save(out_path, output_array[mask])

    if verify_output:
        test_extraction(
            in_rasters,
            out_path,
            out_path_geom,
            samples=verification_samples,
            grid_layer_index=0,
            verbose=verbose,
        ) 

    return (out_path, out_path_geom)



if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"
    
    raster = folder + "fyn_close.tif"
    vector = folder + "odense2.gpkg"
    out_dir = folder + "out/"

    blocks = out_dir + "fyn_close_patches.npy"

    offsets = [(32, 32), (32, 0)]
    borders = True

    path_np, path_geom = extract_patches(
        raster,
        out_dir,
        prefix="",
        postfix="_patches",
        size=64,
        offsets=offsets,
        generate_grid_geom=True,
        generate_border_patches=borders,
        # clip_geom=vector,
        verify_output=True,
        verification_samples=100,
        verbose=1,
    )

    blocks_to_raster(
        blocks,
        raster,
        out_path=out_dir + "fyn_close_reconstituded.tif",
        offsets=offsets,
        border_patches=borders,
        merge_method="median",
    )
