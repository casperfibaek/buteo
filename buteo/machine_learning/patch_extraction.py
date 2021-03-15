import sys

sys.path.append("..")
from lib.raster_io import (
    raster_to_array,
    raster_to_metadata,
    raster_to_memory,
    array_to_raster,
)
from lib.vector_io import vector_to_memory
from lib.raster_clip import clip_raster
from lib.raster_align import is_aligned
from lib.utils_core import progress
import numpy as np
from osgeo import ogr, osr
import rtree
import os
import random

def blocks_to_raster(blocks, reference, output):
    metadata = raster_to_metadata(reference)
    reference_shape = (metadata["height"], metadata["width"])
    top_left = [metadata["transform"][0], metadata["transform"][3]]
    pixel_size = [metadata["transform"][1], metadata["transform"][5]]
    proj = metadata["projection"]

    tiles = blocks.reshape(
        reference_shape[0] // blocks.shape[1],
        reference_shape[1] // blocks.shape[2],
        blocks.shape[1],
        blocks.shape[2],
    ).swapaxes(1, 2).reshape(
        (reference_shape[0] // blocks.shape[1]) * blocks.shape[1],
        (reference_shape[1] // blocks.shape[2]) * blocks.shape[2],
    )

    array_to_raster(
        tiles,
        reference_raster=array_to_raster(
            tiles,
            top_left=top_left,
            pixel_size=pixel_size,
            dst_projection=proj,
        ),
        out_raster=output,
    )

def shape_to_blockshape(shape, block_shape, offset=(0, 0, 0)):
    assert len(offset) >= len(shape), "Input offsets must equal array dimensions."
    assert len(shape) <= 3, "Unable to handle more than 3 dimensions."

    base_shape = list(shape)
    for index, value in enumerate(offset):
        base_shape[index] = base_shape[index] - value
    
    sizes = []

    for index, value in enumerate(base_shape):
        sizes.append(value // block_shape[index])

    return tuple(sizes)


# Channel last format
def array_to_blocks(array, block_shape, offset=(0, 0, 0)):
    assert len(offset) >= len(array.shape), "input offsets must equal array dimensions."
    if len(array.shape) == 1:
        arr = array[
            offset[0] : int(
                array.shape[0] - ((array.shape[0] - offset[0]) % block_shape[0])
            ),
        ]

        return arr.reshape(arr.shape[0] // block_shape[0], block_shape[0]).swapaxes(1).reshape(-1, block_shape[0])
    elif len(array.shape) == 2:
        arr = array[
            offset[1] : int(
                array.shape[0] - ((array.shape[0] - offset[1]) % block_shape[0])
            ),
            offset[0] : int(
                array.shape[1] - ((array.shape[1] - offset[0]) % block_shape[1])
            ),
        ]

        return arr.reshape(arr.shape[0] // block_shape[0], block_shape[0], arr.shape[1] // block_shape[1], block_shape[1]).swapaxes(1, 2).reshape(-1, block_shape[0], block_shape[1])
    elif len(array.shape) == 3:
        arr = array[
            offset[2] : int(
                array.shape[0] - ((array.shape[0] - offset[2]) % block_shape[0])
            ),
            offset[1] : int(
                array.shape[1] - ((array.shape[1] - offset[1]) % block_shape[1])
            ),
            offset[0] : int(
                array.shape[2] - ((array.shape[2] - offset[0]) % block_shape[2])
            ),
        ]

        return arr.reshape(arr.shape[0] // block_shape[0], block_shape[0], arr.shape[1] // block_shape[1], block_shape[1], arr.shape[2] // block_shape[2], block_shape[2]).swapaxes(1, 2).reshape(-1, block_shape[0], block_shape[1], block_shape[2])
    else:
        raise Exception("Unable to handle more than 3 dimensions")


def extract_patches(
    in_rasters,
    output_folder,
    prefix="",
    postfix="_patches",
    size=32,
    offsets=[],
    output_geom=True,
    clip_to_vector=None,
    verbose=1,
):
    if len(in_rasters) == 0:
        raise ValueError("An input raster is required.")

    if not is_aligned(in_rasters):
        raise ValueError("Input rasters must be aligned. Please use the align function.")

    metadata = None

    if clip_to_vector is not None:
        metadata = raster_to_metadata(clip_raster(in_rasters[0], cutline=clip_to_vector, cutline_all_touch=True))
    else:
        metadata = raster_to_metadata(in_rasters[0])

    input_shape = metadata["shape"]

    if verbose == 1:
        print("Generating blocks..")

    offsets.insert(0, (0, 0)) # insert a 0,0 overlap

    shapes = []
    
    for offset in offsets:
        shapes.append(shape_to_blockshape(input_shape, (size, size), offset))

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

    geo_fid = np.zeros(all_rows, dtype="uint64")
    mask = geo_fid

    if output_geom is True or clip_to_vector is not None:

        if verbose == 1:
            print("Calculating grid cells..")

        ulx, uly, lrx, lry = metadata["extent"]

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
            xr = np.arange(x_min, x_max, xres, dtype="float64")[0:x_step]
            yr = np.arange(y_min, y_max + yres, yres, dtype="float64")[::-1][0:y_step]

            oxx, oyy = np.meshgrid(xr, yr)
            
            start = 0
            if l > 0:
                start = offset_rows_cumsum[l - 1]

            coord_grid[start:offset_rows_cumsum[l], 0] = oxx.ravel()
            coord_grid[start:offset_rows_cumsum[l], 1] = oyy.ravel()

        projection = osr.SpatialReference()
        projection.ImportFromWkt(metadata["projection"])

        mem_driver = ogr.GetDriverByName("MEMORY")
        gpkg_driver = ogr.GetDriverByName("GPKG")

        if clip_to_vector is not None:
            clip_vector = (
                clip_to_vector
                if isinstance(clip_to_vector, ogr.DataSource)
                else ogr.Open(clip_to_vector)
            )
            clip_layer = clip_vector.GetLayer(0)
            clip_projection = clip_layer.GetSpatialRef()
            clip_projection_osr = osr.SpatialReference()
            clip_projection_osr.ImportFromWkt(str(clip_projection))

            if not projection.IsSame(clip_projection_osr):
                raise Exception(
                    "clip vector and reference vector is not in the same reference system. Please reproject.."
                )

            # Copy ogr to memory
            clip_mem = mem_driver.CreateDataSource("memData")
            clip_mem.CopyLayer(clip_layer, "mem_clip", ["OVERWRITE=YES"])
            clip_mem_layer = clip_mem.GetLayer("mem_clip")

            if verbose == 1:
                print("Generating rTree..")

            # Generate spatial index
            clip_index = rtree.index.Index(interleaved=False)
            clip_feature_count = clip_mem_layer.GetFeatureCount()

            # Insert the clip geometries into the index
            for p in range(0, clip_feature_count):
                clip_feature = clip_mem_layer.GetNextFeature()
                clip_geometry = clip_feature.GetGeometryRef()
                xmin, xmax, ymin, ymax = clip_geometry.GetEnvelope()
                clip_index.insert(clip_feature.GetFID(), (xmin, xmax, ymin, ymax))

                if verbose == 1:
                    progress(p, clip_feature_count, "rTree generation")

        # Output geometry
        ds = mem_driver.CreateDataSource("mem_grid")
        lyr = ds.CreateLayer("mem_grid_layer", geom_type=ogr.wkbPolygon, srs=projection)
        fdefn = lyr.GetLayerDefn()

        if verbose == 1:
            print("Creating patches..")

        valid_fid = -1
        for q in range(all_rows):
            x, y = coord_grid[q]

            tile_intersects_geom = False

            if clip_to_vector is not None:
                tile_bounds = (x - dx, x + dx, y - dy, y + dy)
                intersections = list(clip_index.intersection(tile_bounds))

                if len(intersections) == 0:

                    if verbose == 1:
                        progress(q, all_rows, "Patches")

                    continue

                for intersection in intersections:
                    clip_feature = clip_layer.GetFeature(intersection)
                    clip_geometry = clip_feature.GetGeometryRef()

                    tile_ring = ogr.Geometry(ogr.wkbLinearRing)
                    tile_ring.AddPoint(x - dx, y + dy) # ul
                    tile_ring.AddPoint(x + dx, y + dy) # ur
                    tile_ring.AddPoint(x + dx, y - dy) # lr 
                    tile_ring.AddPoint(x - dx, y - dy) # ll
                    tile_ring.AddPoint(x - dx, y + dy) # ul begin

                    tile_poly = ogr.Geometry(ogr.wkbPolygon)
                    tile_poly.AddGeometry(tile_ring)

                    if tile_poly.Intersects(clip_geometry) is True:
                        tile_intersects_geom = True
                        break
                
                if tile_intersects_geom is True:
                    valid_fid += 1
            else:
                valid_fid += 1
                
            if (output_geom is True and clip_to_vector is None) or (output_geom is True and clip_to_vector is not None and tile_intersects_geom is True):

                poly_wkt = f"POLYGON (({x - dx} {y + dy}, {x + dx} {y + dy}, {x + dx} {y - dy}, {x - dx} {y - dy}, {x - dx} {y + dy}))"

                ft = ogr.Feature(fdefn)
                ft.SetGeometry(ogr.CreateGeometryFromWkt(poly_wkt))
                ft.SetFID(valid_fid)

                geo_fid[valid_fid] = q

                lyr.CreateFeature(ft)
                ft = None

            if verbose == 1:
                progress(q, all_rows, "Patches")

        # Create mask for numpy arrays
        mask = geo_fid[0:int(valid_fid + 1)]

        if output_geom is True:
            geom_name = f"{prefix}patches_{str(size)}{postfix}"
            geom_out_path = os.path.join(output_folder + f"{geom_name}.gpkg")

            if verbose == 1:
                print("Writing output geometry..")

            if os.path.exists(geom_out_path):
                gpkg_driver.DeleteDataSource(geom_out_path)

            out_grid = gpkg_driver.CreateDataSource(geom_out_path)
            out_grid.CopyLayer(lyr, geom_name, ["OVERWRITE=YES"])

            if valid_fid == -1:
                print("WARNING: Empty geometry output")

    if verbose == 1:
        print("Writing numpy array to disc..")

    # Generate some numpy arrays
    for raster in in_rasters:
        base = os.path.basename(raster)
        basename = os.path.splitext(base)[0]
        out_path = os.path.join(output_folder + f"{prefix}{basename}{postfix}.npy")

        metadata = raster_to_metadata(raster)

        output_shape = (all_rows, size, size)
        input_datatype = metadata["dtype"]

        output_array = np.empty(output_shape, dtype=input_datatype)

        ref = None
        if clip_to_vector is not None:
            ref = raster_to_array(clip_raster(raster, cutline=clip_to_vector, cutline_all_touch=True), filled=True)
        else:
            ref = raster_to_array(raster, filled=True)

        for k, offset in enumerate(offsets):

            start = 0
            if k > 0:
                start = offset_rows_cumsum[k - 1]

            output_array[start:offset_rows_cumsum[k]] = array_to_blocks(ref, (size, size), offset)

        np.save(out_path, output_array[mask])

        ref = None
        output_array = None

    return 1


def test_extraction(in_rasters, numpy_arrays, grid, test_sample=1000, verbose=1):
    if verbose == 1:
        print("Verifying integrity of output grid..")

    test_vect = vector_to_memory(grid)
    test_lyr = test_vect.GetLayer(0)
    test_projection = test_lyr.GetSpatialRef()

    feature_count = test_lyr.GetFeatureCount()
    mem_driver = ogr.GetDriverByName("MEMORY")

    max_test = min(test_sample, feature_count) - 1
    test_fids = np.array(random.sample(range(0, feature_count), max_test), dtype="uint64")

    for index, raster in enumerate(in_rasters):
        test_rast = raster_to_memory(raster)
        test_array = np.load(numpy_arrays[index])

        base = os.path.basename(raster)
        basename = os.path.splitext(base)[0]
        
        test_lyr = test_vect.GetLayer(0)

        if verbose == 1:
            print(f"Testing: {basename}")

        tested = 0
        for test in test_fids:
            feature = test_lyr.GetFeature(test)

            if feature is None:
                raise Exception(f"Feature not found: {test}")

            test_ds = mem_driver.CreateDataSource("test_mem_grid")
            test_ds_lyr = test_ds.CreateLayer(
                "test_mem_grid_layer", geom_type=ogr.wkbPolygon, srs=test_projection
            )
            test_ds_lyr.CreateFeature(feature.Clone())

            ref_img = raster_to_array(test_rast, cutline=test_ds, quiet=True, filled=True)

            image_block = test_array[test]

            if not np.array_equal(ref_img, image_block):
                raise Exception(f"Image {basename} and grid cell did not match..")

            if verbose == 1:
                progress(tested, len(test_fids) - 1, "verifying..")

            tested += 1


if __name__ == "__main__":
    from glob import glob 

    folder = "C:/Users/caspe/Desktop/align_test/"

    extract_patches(
        glob(folder + "aligned/*.tif"),
        folder + "out/",
        prefix="",
        postfix="_patches",
        size=64,
        offsets=[(32, 32), (32, 0), (0, 32)],
        output_geom=True,
        clip_to_vector=folder + "project_area3.gpkg",
        verbose=1,
    )

    aligned = glob(folder + "aligned/*.tif")
    numpy_arrays = glob(folder + "out/*.npy")
    grid = folder + "out/patches_64_patches.gpkg"

    test_extraction(aligned, numpy_arrays, grid)
