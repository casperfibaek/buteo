import sys
from typing import Union; sys.path.append("../../")
from osgeo import ogr, gdal
import numpy as np
import os
import random
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



def blocks_to_raster(
    blocks: Union[str, np.ndarray],
    reference: Union[str, gdal.Dataset],
    out_path: Union[str, None]=None,
    offsets: Union[list, tuple, np.ndarray, None]=None,
    merge_method: str="median",
):
    """ Recombines a series of blocks to a raster.
    Args:
        blocks (ndarray): A numpy array with the values to recombine. The shape
        should be (blocks, rows, column, channel).

        reference (str, raster): A reference raster to help coax the blocks back
        into shape.

        out_path (str | None): Where to save the reconstituted raster. If None
        are memory raster is returned.

        offsets (tuple, list, ndarray): The offsets used in the original. A (0 ,0)
        offset is assumed.

        merge_method (str): How to handle overlapping pixels. Options are:
        median, average, mode, min, max

    Returns:
        A reconstituted raster.
    """
    type_check(blocks, [str, np.ndarray], "blocks")
    type_check(reference, [str, gdal.Dataset], "reference")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(offsets, [list, tuple, np.ndarray], "offsets", allow_none=True)
    type_check(merge_method, [str], "merge_method")

    if isinstance(blocks, str):
        try:
            blocks = np.load(blocks)
        except:
            raise ValueError(f"Failed to parse blocks: {blocks}")

    metadata = raster_to_metadata(reference)
    reference_shape = (metadata["height"], metadata["width"])

    if offsets is None:
        tiles = blocks.reshape(
            reference_shape[0] // blocks.shape[1],
            reference_shape[1] // blocks.shape[2],
            blocks.shape[1],
            blocks.shape[2],
            blocks.shape[3],
            blocks.shape[3],
        ).swapaxes(1, 2).reshape(
            (reference_shape[0] // blocks.shape[1]) * blocks.shape[1],
            (reference_shape[1] // blocks.shape[2]) * blocks.shape[2],
            blocks.shape[3],
        )


        import pdb; pdb.set_trace()

        return array_to_raster(tiles, reference, out_path=out_path)

    

    internal_offsets = []
    if (0, 0) not in offsets or [0, 0] not in offsets:
        internal_offsets.append([0, 0])


def shape_to_blockshape(
    shape: Union[list, tuple, np.ndarray],
    block_shape: Union[list, tuple, np.ndarray],
    offset: Union[list, tuple, np.ndarray],
) -> tuple:
    """ Calculates the shape of the output array.
    Args:
        shape (tuple | list): The shape if the original raster.

        block_shape (tuple | list): The size of the blocks eg. (64, 64)

        offset (tuple, list): An initial offset for the array eg. (32, 32)

    Returns:
        A tuple with the modified shape.
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

    return tuple(sizes)


# Channel last!
def array_to_blocks(
    array: np.ndarray,
    block_shape: Union[list, tuple, np.ndarray],
    offset: Union[list, tuple, np.ndarray],
) -> np.ndarray:
    """ Turns an array into a series of blocks. The array can be offset.
    Args:
        array (ndarray): The array to turn to blocks.

        block_shape (tuple | list | ndarray): The size of the blocks eg. (64, 64)

        offset (tuple, list, ndarray): An initial offset for the array eg. (32, 32)

    Returns:
        A modified view into the array.
    """
    type_check(array, [np.ndarray], "array")
    type_check(block_shape, [list, tuple], "block_shape")
    type_check(offset, [list, tuple], "offset")

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


def extract_patches(
    raster: Union[str, list, gdal.Dataset],
    output_folder: str,
    prefix: str="",
    postfix: str="_patches",
    size: int=32,
    offsets: list=[],
    generate_zero_offset: bool=True,
    generate_grid_geom: bool=True,
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset, None]=None,
    clip_layer_index: int=0,
    overwrite=True,
    epsilon: float=1e-7,
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
        offsets=[(16, 16), (8, 8), (0, 16)]. Will offset the initial raster
        and extract from there.

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

    if generate_zero_offset and (0, 0) not in offsets:
        offsets.insert(0, (0, 0)) # insert a 0,0 overlap

    shapes = []
    for offset in offsets:
        shapes.append(shape_to_blockshape(metadata["shape"], (size, size), offset))

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

        for l in range(len(offsets)):
            x_offset = offsets[l][0]
            y_offset = offsets[l][1]

            x_step = shapes[l][0]
            y_step = shapes[l][1]

            x_min = (ulx + dx) + (x_offset * pixel_width)
            x_max = x_min + (x_step * xres)

            y_max = (uly - dx) - (y_offset * pixel_height)
            y_min = y_max - (y_step * yres)

            # y is flipped so: xmin --> xmax, ymax -- ymin to keep same order as numpy array
            xr = np.arange(x_min, x_max + epsilon, xres, dtype="float64")[0:x_step]
            yr = np.arange(y_min, y_max + epsilon, yres, dtype="float64")[::-1][0:y_step]

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
            progress(q, all_rows, "Patches")

        # Create mask for numpy arrays
        mask = geo_fid[0:int(valid_fid + 1)]

        if generate_grid_geom is True:
            if valid_fid == -1:
                print("WARNING: Empty geometry output")

            geom_name = f"{prefix}geom_{str(size)}{postfix}.gpkg"
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

        for k, offset in enumerate(offsets):
            start = 0
            if k > 0:
                start = offset_rows_cumsum[k - 1]

            output_array[start:offset_rows_cumsum[k]] = array_to_blocks(ref, (size, size), offset)

        if generate_grid_geom is False and clip_geom is None:
            np.save(out_path, output_array)
        else:
            np.save(out_path, output_array[mask])

        ref = None
        output_array = None

    return (out_path, out_path_geom)


def test_extraction(
    rasters: Union[list, str, gdal.Dataset],
    arrays: Union[list, np.ndarray],
    grid: Union[ogr.DataSource, str],
    samples: int=1000, # if 0, all
    clip_layer_index: int=0,
    verbose: int=1,
) -> bool:
    type_check(rasters, [list, str, gdal.Dataset], "rasters")
    type_check(arrays, [list, str, np.ndarray], "arrays")
    type_check(grid, [list, str, np.ndarray], "grid")
    type_check(samples, [int], "samples")
    type_check(clip_layer_index, [int], "clip_layer_index")
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

    if clip_layer_index > (grid_metadata["layer_count"] - 1):
        raise ValueError(f"Requested non-existing layer index: {clip_layer_index}")

    grid_layer = grid_memory.GetLayer(clip_layer_index)

    # Select sample fids
    feature_count = grid_metadata["layers"][clip_layer_index]["feature_count"]
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
            ref_img = raster_to_array(clipped, filled=True)
            image_block = test_array[test]

            if not np.array_equal(ref_img, image_block):
                raise Exception(f"Image {basename} and grid cell did not match..")

            if verbose == 1:
                progress(tested, len(test_fids) - 1, "verifying..")

            tested += 1

    return True


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"
    
    raster = folder + "fyn.tif"
    vector = folder + "odense2.gpkg"
    out_dir = folder + "out/"

    blocks = out_dir + "fyn_patches.npy"

    # extract_patches(
    #     raster,
    #     out_dir,
    #     prefix="",
    #     postfix="_patches",
    #     size=64,
    #     # offsets=[(32, 32), (32, 0), (0, 32)],
    #     generate_grid_geom=True,
    #     # clip_geom=vector,
    #     verbose=1,
    # )

    # test_extraction(
    #     raster,
    #     out_dir + "fyn_patches.npy",
    #     out_dir + "geom_64_patches.gpkg",
    #     samples=100,
    # )

    blocks_to_raster(
        blocks,
        raster,
        out_path=out_dir + "reconstituted.tif",
        # offsets=[(32, 32), (32, 0), (0, 32)],
        merge_method="median",
    )