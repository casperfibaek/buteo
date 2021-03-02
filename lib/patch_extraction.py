# Extract patches to numpy arrays from a rasters extent and pixel count
# optionally output centroids
from pdb import main
import sys

sys.path.append("..")
from lib.raster_io import (
    raster_to_array,
    raster_to_metadata,
    raster_to_memory,
    array_to_raster,
)
from lib.utils_core import progress
import numpy as np
from osgeo import ogr, osr
import rtree
import os

def blocks_to_array(blocks, reference, output):
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
    reference,
    output_numpy,
    size=32,
    offsets=[],
    output_geom=None,
    clip_to_vector=None,
    fill_value=None,
    epsilon=1e-7,
    verbose=1,
    testing=False,
    testing_sample=1000,
    dtype=None,
    start_fid=0,
):
    metadata = raster_to_metadata(reference)

    input_datatype = metadata["dtype"]
    input_nodata_value = metadata["nodata_value"]
    input_shape = metadata["shape"]

    if verbose == 1:
        print("Generating blocks..")

    offsets.insert(0, (0, 0)) # insert a 0,0 overlap

    shapes = []
    
    for offset in offsets:
        shapes.append(shape_to_blockshape(input_shape, (size, size), offset))

    all_rows = 0
    offset_rows = [0]
    for i in range(len(shapes)):
        row = 0

        for j in range(len(shapes[i])):
            if j == 0:
                row = int(shapes[i][j])
            else:
                row *= int(shapes[i][j])

        offset_rows.append(row)
        all_rows += row

    output_shape = (all_rows, size, size)
    output_array = np.empty(output_shape, dtype=input_datatype)
    offset_rows_cumsum = np.cumsum(offset_rows)

    for index, offset in enumerate(offsets):
        output_array[offset_rows_cumsum[index]:offset_rows_cumsum[index + 1]] = array_to_blocks(raster_to_array(reference), (size, size), offset)

    mask = np.ones(all_rows, 'uint64')

    if output_geom is not None or clip_to_vector is not None:

        if verbose == 1:
            print("Calculating grid cells..")

        ulx, uly, lrx, lry = metadata["extent"]

        pixel_width = abs(metadata["pixel_width"])
        pixel_height = abs(metadata["pixel_height"])

        xres = pixel_width * size
        yres = pixel_height * size

        dx = xres / 2
        dy = yres / 2

        coord_grid = np.empty((all_rows, 2), dtype="uint64")

        for i in range(len(offsets)):
            x_offset = offsets[i][0]
            y_offset = offsets[i][1]

            x_step = ((input_shape[1] - x_offset) - ((input_shape[1] - x_offset) % size)) // size
            y_step = ((input_shape[0] - y_offset) - ((input_shape[0] - y_offset) % size)) // size

            x_min = (ulx + dx) + (x_offset * pixel_width)
            x_max = x_min + (x_step * xres)

            y_max = (uly - dx) - (y_offset * pixel_height)
            y_min = y_max - (y_step * yres)

            # y is flipped so: xmin --> xmax, ymax -- ymin to keep same order as numpy array
            xr = np.arange(x_min, x_max, xres, dtype="uint64")[0:x_step]
            yr = np.arange(y_min, y_max + yres, yres, dtype="uint64")[::-1][0:y_step]

            # adjust shape if needed
            if yr.shape[0] < y_step:
                yr = np.arange(y_min, y_max + yres, yres, dtype="uint64")[::-1][0:y_step]
            elif yr.shape[0] > y_step:
                yr = np.arange(y_min, y_max - yres, yres, dtype="uint64")[::-1][0:y_step]

            if xr.shape[0] < x_step:
                xr = np.arange(x_min, x_max + xres, xres, dtype="uint64")[0:x_step]
            elif xr.shape[0] > x_step:
                xr = np.arange(x_min, x_max - xres, xres, dtype="uint64")[0:x_step]

            oxx, oyy = np.meshgrid(xr, yr)

            if i > 0 and oxx.size != offset_rows[i + 1]:
                import pdb; pdb.set_trace()
                raise Exception("Error while matching grid and images.")

            coord_grid[offset_rows_cumsum[i]:offset_rows_cumsum[i + 1], 0] = oxx.ravel()
            coord_grid[offset_rows_cumsum[i]:offset_rows_cumsum[i + 1], 1] = oyy.ravel()

        if coord_grid.shape[0] != all_rows:
            raise Exception("Error while calculating. Total_images != total squares")

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
            print("")

        valid_fid = start_fid - 1
        for i in range(all_rows):
            x, y = coord_grid[i]

            if clip_to_vector is not None:
                tile_bounds = (x - dx, x + dx, y - dy, y + dy)
                intersections = list(clip_index.intersection(tile_bounds))

                if len(intersections) == 0:

                    if verbose == 1:
                        progress(i, all_rows, "Patches")

                    continue

                tile_intersects_geom = False
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

            if clip_to_vector is None or tile_intersects_geom is True:
                valid_fid += 1
                mask[valid_fid] = i

                poly_wkt = f"POLYGON (({x - dx} {y + dy}, {x + dx} {y + dy}, {x + dx} {y - dy}, {x - dx} {y - dy}, {x - dx} {y + dy}))"

                ft = ogr.Feature(fdefn)
                ft.SetGeometry(ogr.CreateGeometryFromWkt(poly_wkt))
                ft.SetFID(valid_fid)

                lyr.CreateFeature(ft)
                ft = None

            if verbose == 1:
                progress(i, all_rows, "Patches")

        grid_cells = lyr.GetFeatureCount()

        mask = mask[:valid_fid + 1]
        output_array = output_array[mask]

        assert (
            grid_cells == output_array.shape[0]
        ), "Image count and grid count does not match."

        if testing == True and output_geom is not None:
            if verbose == 1:
                print("\nVerifying integrity of output grid..")

            test_rast = raster_to_memory(reference)
            max_test = min(int(testing_sample), int(valid_fid))
            test_fids = np.random.randint(0, grid_cells - 1, max_test)
            tested = 0

            for feature in lyr:
                fid = feature.GetFID()

                if fid not in test_fids:
                    continue

                test_ds = mem_driver.CreateDataSource("test_mem_grid")
                test_lyr = test_ds.CreateLayer(
                    "test_mem_grid_layer", geom_type=ogr.wkbPolygon, srs=projection
                )
                test_lyr.CreateFeature(feature.Clone())

                ref_img = raster_to_array(test_rast, cutline=test_ds, quiet=True)
                image_block = output_array[fid]      

                assert np.array_equal(ref_img, image_block), print("Image and grid cell did not match..")

                if verbose == 1:
                    progress(tested, len(test_fids) - 1, "verifying..")

                tested += 1

        if output_geom is not None:
            if verbose == 1:
                print("")
                print("Writing output geometry..")

            if os.path.exists(output_geom):
                gpkg_driver.DeleteDataSource(output_geom)

            out_name = os.path.basename(output_geom).rsplit(".", 1)[0]
            out_grid = gpkg_driver.CreateDataSource(output_geom)
            out_grid.CopyLayer(lyr, out_name, ["OVERWRITE=YES"])

            if valid_fid == start_fid - 1:
                print("WARNING: Empty geometry output")

    if verbose == 1:
        print("Writing numpy array to disc..")
    
    if dtype is not None:
        output_array = output_array.astype(dtype)

    if isinstance(output_array, np.ma.MaskedArray):
        
        fill = fill_value
        
        if fill == None:
            fill = input_nodata_value
        
        if fill == None:
            fill = 0
        
        np.save(output_numpy, output_array.filled(fill_value=fill_value))
    else:
        np.save(output_numpy, output_array)

    return 1

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from patch_extraction import extract_patches
    from raster_io import raster_to_array, array_to_raster
    import numpy as np
    from glob import glob
    import os


    folder = "C:/Users/caspe/Desktop/patch_test/"

    extract_patches(
        folder + "red_fyn.tif",
        folder + "red_fyn.npy",
        size=64,
        clip_to_vector=folder + "buffered_wall.gpkg",
        offsets=[(0, 32), (32, 32), (32, 0)],
        fill_value=0,
        output_geom=folder + "red_fyn.gpkg",
        testing=True,
    )

