# Extract patches to numpy arrays from a rasters extent and pixel count
# optionally output centroids
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

# Channel last format
def array_to_blocks(array, block_shape, offset=(0, 0, 0)):
    assert len(offset) >= len(array.shape), "input offsets must equal array dimensions."
    if len(array.shape) == 1:
        arr = array[
            offset[0] : int(
                array.shape[0] - ((array.shape[0] - offset[0]) % block_shape[0])
            ),
        ]
        shape = (np.array(arr.shape) / np.array(block_shape)).astype("int64")
        return (
            arr.reshape(arr.shape[0] // block_shape[0], block_shape[0],)
            .swapaxes(1)
            .reshape(-1, block_shape[0]),
            shape,
        )
    elif len(array.shape) == 2:
        arr = array[
            offset[1] : int(
                array.shape[0] - ((array.shape[0] - offset[1]) % block_shape[0])
            ),
            offset[0] : int(
                array.shape[1] - ((array.shape[1] - offset[0]) % block_shape[1])
            ),
        ]
        shape = (np.array(arr.shape) / np.array(block_shape)).astype("int64")
        return (
            arr.reshape(
                arr.shape[0] // block_shape[0],
                block_shape[0],
                arr.shape[1] // block_shape[1],
                block_shape[1],
            )
            .swapaxes(1, 2)
            .reshape(-1, block_shape[0], block_shape[1]),
            shape,
        )
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
        shape = (np.array(arr.shape) / np.array(block_shape)).astype("int64")
        return (
            arr.reshape(
                arr.shape[0] // block_shape[0],
                block_shape[0],
                arr.shape[1] // block_shape[1],
                block_shape[1],
                arr.shape[2] // block_shape[2],
                block_shape[2],
            )
            .swapaxes(1, 2)
            .reshape(-1, block_shape[0], block_shape[1], block_shape[2]),
            shape,
        )
    else:
        raise Exception("Unable to handle more than 3 dimensions")


def extract_patches(
    reference,
    output_numpy,
    size=32,
    overlaps=[],
    output_geom=None,
    clip_to_vector=None,
    fill_value=None,
    epsilon=1e-7,
    verbose=1,
    testing=False,
    testing_sample=1000,
):
    metadata = raster_to_metadata(reference)
    reference_array = raster_to_array(reference)

    if verbose == 1:
        print("Generating blocks..")
    blocks, shape = array_to_blocks(reference_array, (size, size))
    images_per_block = [blocks.shape[0]]
    shapes = [shape.tolist()]

    for overlap in overlaps:
        block, shape = array_to_blocks(reference_array, (size, size), offset=overlap)
        blocks = np.concatenate([blocks, block])
        images_per_block.append(block.shape[0])
        shapes.append(shape.tolist())

    images = blocks.shape[0]
    mask = np.ones(images, dtype=bool)

    if output_geom is not None or clip_to_vector is not None:
        ulx, uly, lrx, lry = metadata["extent"]

        width = abs(metadata["width"])
        height = abs(metadata["height"])
        pixel_width = abs(metadata["pixel_width"])
        pixel_height = abs(metadata["pixel_height"])

        xres = pixel_width * size
        yres = pixel_height * size

        dx = xres / 2
        dy = yres / 2

        x_max = lrx + dx - ((width % size) * pixel_width)
        x_min = ulx + dx
        y_max = uly - dy
        y_min = lry - dy + ((height % size) * pixel_height)

        # y is flipped so: xmin --> xmax, ymax -- ymin to keep same order as numpy array
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, xres), np.arange(y_max, y_min, -yres)
        )

        coord_grid = np.array([xx.ravel(), yy.ravel()])

        for i in range(len(overlaps)):
            overlap_x_size = overlaps[i][0] * pixel_width
            overlap_y_size = overlaps[i][1] * pixel_height

            xt = shapes[i + 1][1]
            yt = shapes[i + 1][0]

            # y is flipped so: xmin --> xmax, ymax -- ymin to keep same order as numpy array
            xr = np.arange(x_min + overlap_x_size, x_max - overlap_x_size, xres)
            yr = np.arange(y_max - overlap_y_size, y_min + overlap_y_size, -yres)

            xfix = 0
            yfix = 0
            if xr.shape[0] < xt:
                xfix = xres
            elif xr.shape[0] > xt:
                xfix = -xres

            if yr.shape[0] < yt:
                yfix = -yres
            elif yr.shape[0] > yt:
                yfix = yres

            xr = np.arange(x_min + overlap_x_size, x_max - overlap_x_size + xfix, xres)
            yr = np.arange(y_max - overlap_y_size, y_min + overlap_y_size + yfix, -yres)

            oxx, oyy = np.meshgrid(xr, yr)

            if oxx.size != images_per_block[i + 1]:
                raise Exception("Error while matching grid and images.")

            coord_grid = np.append(
                coord_grid, np.array([oxx.ravel(), oyy.ravel()]), axis=1
            )

        coord_grid = np.swapaxes(coord_grid, 0, 1)

        if coord_grid.shape[0] != images:
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
            for p in range(0, clip_feature_count):
                clip_feature = clip_mem_layer.GetNextFeature()
                clip_geometry = clip_feature.GetGeometryRef()
                xmin, xmax, ymin, ymax = clip_geometry.GetEnvelope()
                clip_index.insert(clip_feature.GetFID(), (xmin, xmax, ymin, ymax))
                if verbose == 1:
                    progress(p, clip_feature_count, "rTree generation")

        ds = mem_driver.CreateDataSource("mem_grid")
        lyr = ds.CreateLayer("mem_grid_layer", geom_type=ogr.wkbPolygon, srs=projection)
        fdefn = lyr.GetLayerDefn()

        if verbose == 1:
            print("Creating patches..")

        valid_fid = -1
        for i in range(images):
            x, y = coord_grid[i]

            if x + dx > lrx or x - dx < ulx or y + dy > uly or y - dy < lry:
                mask[i] = False
                continue

            if clip_to_vector is not None:
                intersections = list(
                    clip_index.intersection((x - dx, x + dx, y - dy, y + dy))
                )

                if len(intersections) < 4:
                    mask[i] = False
                    continue

                if len(intersections) < 9:
                    is_within = [False, False, False, False]
                    for intersection in intersections:
                        clip_feature = clip_layer.GetFeature(intersection)
                        clip_geometry = clip_feature.GetGeometryRef()

                        ul = ogr.Geometry(ogr.wkbPoint)
                        ul.AddPoint(x - dx, y + dy)
                        ur = ogr.Geometry(ogr.wkbPoint)
                        ur.AddPoint(x + dx, y + dy)
                        ll = ogr.Geometry(ogr.wkbPoint)
                        ll.AddPoint(x - dx, y - dy)
                        lr = ogr.Geometry(ogr.wkbPoint)
                        lr.AddPoint(x + dx, y - dy)

                        if clip_geometry.Intersects(ul.Buffer(epsilon)) is True:
                            is_within[0] = True
                        if clip_geometry.Intersects(ur.Buffer(epsilon)) is True:
                            is_within[1] = True
                        if clip_geometry.Intersects(ll.Buffer(epsilon)) is True:
                            is_within[2] = True
                        if clip_geometry.Intersects(lr.Buffer(epsilon)) is True:
                            is_within[3] = True

                        clip_feature = None
                        clip_geometry = None

                        if False not in is_within:
                            break

                    if False in is_within:
                        mask[i] = False
                        continue

            poly_wkt = f"POLYGON (({x - dx} {y + dy}, {x + dx} {y + dy}, {x + dx} {y - dy}, {x - dx} {y - dy}, {x - dx} {y + dy}))"

            valid_fid += 1

            ft = ogr.Feature(fdefn)
            ft.SetGeometry(ogr.CreateGeometryFromWkt(poly_wkt))
            ft.SetFID(valid_fid)

            lyr.CreateFeature(ft)
            ft = None

            if verbose == 1:
                progress(i, images, "Patches")

        grid_cells = lyr.GetFeatureCount()
        assert (
            grid_cells == blocks[mask].shape[0]
        ), "Image count and grid count does not match."

        if testing == True:
            if verbose == 1:
                print("\nVerifying integrity of output grid..")
            test_rast = raster_to_memory(reference)
            img = blocks[mask]
            test_fids = np.random.randint(0, grid_cells, testing_sample)
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
                image_block = img[fid]

                if not np.array_equal(ref_img, image_block):
                    import pdb

                    pdb.set_trace()

                assert np.array_equal(
                    ref_img, image_block
                ), "Image and grid cell did not match.."

                if verbose == 1:
                    progress(tested, testing_sample - 1, "verifying..")
                tested += 1

        if output_geom is not None:
            if verbose == 1:
                print("Writing output geometry..")

            if os.path.exists(output_geom):
                gpkg_driver.DeleteDataSource(output_geom)

            out_name = os.path.basename(output_geom).rsplit(".", 1)[0]
            out_grid = gpkg_driver.CreateDataSource(output_geom)
            out_grid.CopyLayer(lyr, out_name, ["OVERWRITE=YES"])

    lyr = None
    ds = None
    clip_vector = None
    clip_layer = None
    out_grid = None

    if verbose == 1:
        print("Writing numpy array to disc..")
    
    output = blocks[mask]

    if isinstance(output, np.ma.MaskedArray):
        np.save(output_numpy, output.filled(fill_value=fill_value))
    else:
        np.save(output_numpy, output)

    return 1
