# Extract patches to numpy arrays from a rasters extent and pixel count
# optionally output centroids
import sys; sys.path.append('..');
from lib.raster_io import raster_to_array, array_to_raster, raster_to_metadata
from lib.utils_core import progress
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from osgeo import ogr, osr
import rtree
import os


# Channel last format
def array_to_blocks(array, block_shape, offset=(0, 0, 0)):
    assert len(offset) >= len(array.shape), "input offsets must equal array dimensions."

    if len(array.shape) == 1:
        arr = array[
            offset[0]:int(array.shape[0] - ((array.shape[0] - offset[0]) % block_shape[0])),
        ]
        return arr.reshape(
            arr.shape[0] // block_shape[0],
            block_shape[0],
        ).swapaxes(1).reshape(-1, block_shape[0])
    elif len(array.shape) == 2:
        arr = array[
            offset[0]:int(array.shape[0] - ((array.shape[0] - offset[0]) % block_shape[0])),
            offset[1]:int(array.shape[1] - ((array.shape[1] - offset[1]) % block_shape[1])),
        ]
        return arr.reshape(
            arr.shape[0] // block_shape[0],
            block_shape[0],
            arr.shape[1] // block_shape[1],
            block_shape[1]
        ).swapaxes(1, 2).reshape(-1, block_shape[0], block_shape[1])
    elif len(array.shape) == 3:
        arr = array[
            offset[0]:int(array.shape[0] - ((array.shape[0] - offset[0]) % block_shape[0])),
            offset[1]:int(array.shape[1] - ((array.shape[1] - offset[1]) % block_shape[1])),
            offset[2]:int(array.shape[2] - ((array.shape[2] - offset[2]) % block_shape[2])),
        ]
        return arr.reshape(
            arr.shape[0] // block_shape[0],
            block_shape[0],
            arr.shape[1] // block_shape[1],
            block_shape[1],
            arr.shape[2] // block_shape[2],
            block_shape[2]
        ).swapaxes(1, 2).reshape(-1, block_shape[0], block_shape[1], block_shape[2])
    else:
        raise Exception("Unable to handle more than 3 dimensions")


def to_8bit(arr, min_target, max_target):
    return np.interp(arr, (min_target, max_target), (0, 255)).astype('uint8')


# TODO: Enable offset overlap (i.e. overlaps=[(8,0), (8,8), (0,8)])
# TODO: Handle 3d arrays

def extract_patches(reference, output_numpy, size=32, overlaps=[], output_geom=None, clip_to_vector=None, epsilon=1e-7, verbose=1):
    metadata = raster_to_metadata(reference)
    reference = to_8bit(raster_to_array(reference), 0, 2000)

    if verbose == 1: print("Generating blocks..")
    blocks = array_to_blocks(reference, (size, size))
    images_per_block = [blocks.shape[0]]

    for overlap in overlaps:
        block = array_to_blocks(reference, (size, size), offset=overlap)
        blocks = np.concatenate([blocks, block])
        images_per_block.append(block.shape[0])

    images = blocks.shape[0]
    mask = np.ones(images, dtype=bool)

    if output_geom is not None:
        ulx, uly, lrx, lry = metadata["extent"]

        width = abs(metadata["width"])
        height = abs(metadata["height"])
        pixel_width = abs(metadata["pixel_width"])
        pixel_height = abs(metadata["pixel_height"])

        # Resolution
        xres = pixel_width * size
        yres = pixel_height * size

        dx = xres / 2
        dy = yres / 2

        x_max = lrx + dx - ((width % size) * pixel_width)
        x_min = ulx + dx
        y_max = uly + dy
        y_min = lry + dy + ((height % size) * pixel_height)

        xx, yy = np.meshgrid(np.arange(x_min, x_max, xres), np.arange(y_min, y_max, yres))

        coord_grid = np.array([xx.ravel(), yy.ravel()])

        for i in range(len(overlaps)):
            overlap_x_size = overlaps[i][0] * pixel_width
            overlap_y_size = overlaps[i][1] * pixel_height

            oxx, oyy = np.meshgrid(
                np.arange(x_min + overlap_x_size, x_max - overlap_x_size, xres),
                np.arange(y_min + overlap_y_size, y_max - overlap_y_size, yres),
            )

            import pdb; pdb.set_trace()

            coord_grid = np.append(coord_grid, np.array([oxx.ravel(), oyy.ravel()]), axis=1)

        coord_grid = np.swapaxes(coord_grid, 0, 1)

        if coord_grid.shape[0] != images:
            raise Exception("Error while calculating. Total_images != total squares")

        projection = osr.SpatialReference()
        projection.ImportFromWkt(metadata["projection"])

        mem_driver = ogr.GetDriverByName('MEMORY')
        shp_driver = ogr.GetDriverByName('ESRI Shapefile')

        if clip_to_vector is not None:
            clip_vector = clip_to_vector if isinstance(clip_to_vector, ogr.DataSource) else ogr.Open(clip_to_vector)
            clip_layer = clip_vector.GetLayer(0)
            clip_projection = clip_layer.GetSpatialRef()
            clip_projection_osr = osr.SpatialReference()
            clip_projection_osr.ImportFromWkt(str(clip_projection))

            if not projection.IsSame(clip_projection_osr):
                raise Exception("clip vector and reference vector is not in the same reference system. Please reproject..")

            # Copy ogr to memory
            clip_mem = mem_driver.CreateDataSource('memData')
            clip_mem.CopyLayer(clip_layer, 'mem_clip', ['OVERWRITE=YES'])
            clip_mem_layer = clip_mem.GetLayer('mem_clip')

            if verbose == 1: print("Generating rTree..")
            # Generate spatial index
            clip_index = rtree.index.Index(interleaved=False)
            clip_feature_count = clip_mem_layer.GetFeatureCount()
            for clip_fid in range(0, clip_feature_count):
                clip_feature = clip_mem_layer.GetNextFeature()
                clip_geometry = clip_feature.GetGeometryRef()
                xmin, xmax, ymin, ymax = clip_geometry.GetEnvelope()
                clip_index.insert(clip_fid, (xmin, xmax, ymin, ymax))
                if verbose == 1: progress(clip_fid, clip_feature_count, 'rTree generation')


        ds = mem_driver.CreateDataSource("mem_grid")
        lyr = ds.CreateLayer("mem_grid_layer", geom_type=ogr.wkbPolygon, srs=projection)
        fdefn = lyr.GetLayerDefn()

        if verbose == 1: print("Creating patches..")

        for i in range(images):
            x, y = coord_grid[i]

            if (x + dx) > lrx or (y - dy) < lry:
                mask[i] = False
                continue

            poly_wkt = f'POLYGON (({x - dx} {y + dy}, {x + dx} {y + dy}, {x + dx} {y - dy}, {x - dx} {y - dy}, {x - dx} {y + dy}))'

            ft = ogr.Feature(fdefn)
            ft.SetGeometry(ogr.CreateGeometryFromWkt(poly_wkt))

            if clip_to_vector is not None:
                intersections = list(clip_index.intersection((x - dx, x + dx, y - dy, y + dy)))

                if len(intersections) < 4:
                    mask[i] = False
                    continue

                if len(intersections) < 9:
                    is_within = [False, False, False, False]
                    for intersection in intersections:
                        clip_feature = clip_layer.GetFeature(intersection)
                        clip_geometry = clip_feature.GetGeometryRef()
                        ft_geom = ft.GetGeometryRef()

                        ul = ogr.Geometry(ogr.wkbPoint); ul.AddPoint(x - dx, y + dy)
                        ur = ogr.Geometry(ogr.wkbPoint); ur.AddPoint(x + dx, y + dy)
                        ll = ogr.Geometry(ogr.wkbPoint); ll.AddPoint(x - dx, y - dy)
                        lr = ogr.Geometry(ogr.wkbPoint); lr.AddPoint(x + dx, y - dy)

                        if clip_geometry.Intersects(ul.Buffer(epsilon)) is True: is_within[0] = True
                        if clip_geometry.Intersects(ur.Buffer(epsilon)) is True: is_within[1] = True
                        if clip_geometry.Intersects(ll.Buffer(epsilon)) is True: is_within[2] = True
                        if clip_geometry.Intersects(lr.Buffer(epsilon)) is True: is_within[3] = True

                        clip_feature = None
                        clip_geometry = None

                        if False not in is_within:
                            break

                    if False in is_within:
                        mask[i] = False
                        continue

            lyr.CreateFeature(ft)
            ft = None
            if verbose == 1: progress(i, images, 'Patches')
    
    grid_cells = lyr.GetFeatureCount()
    if grid_cells != blocks[mask].shape[0]:
        print("Count does not match..")
        import pdb; pdb.set_trace()

    if os.path.exists(output_geom):
        shp_driver.DeleteDataSource(output_geom)

    out_name = os.path.basename(output_geom).rsplit('.', 1)[0]
    out_grid = shp_driver.CreateDataSource(output_geom)
    out_grid.CopyLayer(lyr, out_name, ['OVERWRITE=YES'])
    
    lyr = None
    ds = None
    clip_vector = None
    clip_layer = None
    out_grid = None
    out_grid_layer = None

    np.save(output_numpy, blocks[mask])

    import pdb; pdb.set_trace()

    return 1


if __name__ == "__main__":
    folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\patch_extraction\\"
    ref = folder + "b04_test.tif"
    grid = folder + "grid_test.shp"
    geom = folder + "processed\\b4_160m_geom.shp"
    numpy_arr = folder + "processed\\b4_160m.npy"

    extract_patches(
        ref,
        numpy_arr,
        size=16,
        overlaps=[(8, 8)],
        output_geom=geom,
        clip_to_vector=grid,
        epsilon=1e-7,
    )
