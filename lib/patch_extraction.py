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
def array_to_blocks(array, block_shape, offset=0):
    if offset == 0:
        arr = array
    else:
        if len(array.shape) == 1:
            arr = array[
                offset:array.shape[0] - ((array.shape[0] - offset) % block_shape[0]),
            ]
        if len(array.shape) == 2:
            arr = array[
                offset:array.shape[0] - ((array.shape[0] - offset) % block_shape[0]),
                offset:array.shape[1] - ((array.shape[1] - offset) % block_shape[1]),
            ]
        if len(array.shape) == 3:
            arr = array[
                offset:array.shape[0] - ((array.shape[0] - offset) % block_shape[0]),
                offset:array.shape[1] - ((array.shape[1] - offset) % block_shape[1]),
                offset:array.shape[2] - ((array.shape[2] - offset) % block_shape[2]),
            ]

    if len(arr.shape) == 1:
        return arr[
            0:arr.shape[0] - ceil(arr.shape[0] % block_shape[0]),
        ].reshape(
            arr.shape[0] // block_shape[0],
            block_shape[0],
        ).swapaxes(1).reshape(-1, block_shape[0])

    elif len(arr.shape) == 2:
        return arr[
            0:arr.shape[0] - ceil(arr.shape[0] % block_shape[0]),
            0:arr.shape[1] - ceil(arr.shape[1] % block_shape[1]),
        ].reshape(
            arr.shape[0] // block_shape[0],
            block_shape[0],
            arr.shape[1] // block_shape[1],
            block_shape[1]
        ).swapaxes(1, 2).reshape(-1, block_shape[0], block_shape[1])

    elif len(arr.shape) == 3:
        return arr[
            0:arr.shape[0] - ceil(arr.shape[0] % block_shape[0]),
            0:arr.shape[1] - ceil(arr.shape[1] % block_shape[1]),
            0:arr.shape[2] - ceil(arr.shape[2] % block_shape[2])
        ].reshape(
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
# TODO: Match processed and image count

def extract_patches(reference, output_numpy, width=32, height=32, overlaps=[], output_geom=None, clip_to_vector=None, epsilon=1e-5):
    assert width == height, print("height must equal width.")

    metadata = raster_to_metadata(reference)
    reference = to_8bit(raster_to_array(reference), 0, 2000)

    img_width = metadata["width"]
    img_height = metadata["height"]

    patches_x = int(img_width / width)
    patches_y = int(img_height / height)

    blocks = [array_to_blocks(reference, (width, height))]
    images = blocks[0].shape[0]

    for overlap in overlaps:
        block = array_to_blocks(reference, (width, height), offset=overlap)
        images += block.shape[0]
        blocks.append(block)

    print("block1: ", blocks[0].shape)

    if len(overlaps) > 0:
        blocks = np.concatenate(blocks)
    else:
        blocks = blocks[0]
    
    mask = np.ones(images, dtype=bool)

    total_length = 0
    
    # imgplot = plt.imshow(blocks[0]); plt.show()
    if output_geom is not None:
        ulx, uly, lrx, lry = metadata["extent"]

        # Resolution
        xres = metadata["pixel_width"] * width
        yres = metadata["pixel_height"] * height

        dx = xres / 2
        dy = yres / 2

        xx, yy = np.meshgrid(
            np.arange(ulx + dx, lrx + dx, xres), 
            np.arange(uly + dy, lry + dy, yres),
        )

        coord_grid_x = [xx]
        coord_grid_y = [yy]

        total_length += xx.shape[0] * yy.shape[1]

        for o in overlaps:
            odx = ((metadata["pixel_width"] * width) - (o * metadata["pixel_width"])) - dx
            ody = ((metadata["pixel_height"] * height) - (o * metadata["pixel_height"])) - dy

            # Verify that this is correct
            oxx, oyy = np.meshgrid(
                np.arange(ulx + odx + dx, lrx + odx - dx, xres), 
                np.arange(uly + ody + dy, lry + ody - dy, yres),
            )

            coord_grid_x.append(oxx)
            coord_grid_y.append(oyy)

            total_length += oxx.shape[0] * oyy.shape[1]

        projection = osr.SpatialReference()
        projection.ImportFromWkt(metadata["projection"])

        if clip_to_vector is not None:
            clip_vector = clip_to_vector if isinstance(clip_to_vector, ogr.DataSource) else ogr.Open(clip_to_vector)
            clip_layer = clip_vector.GetLayer(0)
            clip_projection = clip_layer.GetSpatialRef()
            clip_projection_osr = osr.SpatialReference()
            clip_projection_osr.ImportFromWkt(str(clip_projection))

            if not projection.IsSame(clip_projection_osr):
                raise Exception("clip vector and reference vector is not in the same reference system. Please reproject..")

            # Generate spatial index
            clip_index = rtree.index.Index(interleaved=False)
            for clip_fid in range(0, clip_layer.GetFeatureCount()):
                clip_feature = clip_layer.GetNextFeature()
                clip_geometry = clip_feature.GetGeometryRef()
                xmin, xmax, ymin, ymax = clip_geometry.GetEnvelope()
                clip_index.insert(clip_fid, (xmin, xmax, ymin, ymax))

        driver = ogr.GetDriverByName('GPKG')

        if os.path.exists(output_geom):
            driver.DeleteDataSource(output_geom)

        ds = driver.CreateDataSource(output_geom)
        lyr = ds.CreateLayer(os.path.basename(output_geom).rsplit('.', 1)[0], geom_type=ogr.wkbPolygon, srs=projection)
        fdefn = lyr.GetLayerDefn()

        processed = 0
        valids = 0
        progress(processed, total_length, 'Creating patches')

        for coord_grid_id in range(len(coord_grid_x)):
            for x, y in zip(coord_grid_x[coord_grid_id].ravel(), coord_grid_y[coord_grid_id].ravel()):
                if (x + dx) > lrx or (y + dy) < lry:
                    mask[processed] = False
                    processed += 1
                    progress(processed, total_length, 'Patches')
                    continue

                poly_wkt = f'POLYGON (({x - dx} {y - dy}, {x + dx} {y - dy}, {x + dx} {y + dy}, {x - dx} {y + dy}, {x - dx} {y - dy}))'

                ft = ogr.Feature(fdefn)
                ft.SetGeometry(ogr.CreateGeometryFromWkt(poly_wkt))

                if clip_to_vector is not None:
                    intersections = list(clip_index.intersection((x - dx, x + dx, y + dy, y - dy)))

                    if len(intersections) == 0:
                        mask[processed] = False
                        processed += 1
                        progress(processed, total_length, 'Patches')
                        continue

                    #             ul,    ur,    ll,    lr
                    is_within = [False, False, False, False]
                    for i in intersections:
                        clip_feature = clip_layer.GetFeature(i)
                        clip_geometry = clip_feature.GetGeometryRef()
                        ft_geom = ft.GetGeometryRef()

                        ul = ogr.Geometry(ogr.wkbPoint); ul.AddPoint(x - dx, y - dy)
                        ur = ogr.Geometry(ogr.wkbPoint); ur.AddPoint(x + dx, y - dy)
                        ll = ogr.Geometry(ogr.wkbPoint); ll.AddPoint(x - dx, y + dy)
                        lr = ogr.Geometry(ogr.wkbPoint); lr.AddPoint(x + dx, y + dy)

                        if clip_geometry.Intersects(ul) is True: is_within[0] = True
                        if clip_geometry.Intersects(ur) is True: is_within[1] = True
                        if clip_geometry.Intersects(ll) is True: is_within[2] = True
                        if clip_geometry.Intersects(lr) is True: is_within[3] = True

                    if False in is_within:
                        mask[processed] = False
                        processed += 1
                        progress(processed, total_length, 'Patches')
                        continue

                lyr.CreateFeature(ft)
                ft = None

                valids += 1
                processed += 1
                progress(processed, total_length, 'Patches')

        lyr = None
        ds = None
        clip_vector = None
        clip_layer = None

    np.save(output_numpy, blocks[mask])
    
    import pdb; pdb.set_trace()

    return 1


if __name__ == "__main__":
    folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\patch_extraction\\"
    ref = folder + "b04_test.tif"
    grid = folder + "test_grid.shp"
    geom = folder + "b4_160m_geom.gpkg"
    numpy_arr = folder + "b4_160m.npy"

    extract_patches(
        ref,
        numpy_arr,
        width=16,
        height=16,
        overlaps=[8],
        output_geom=geom,
        clip_to_vector=grid,
    )
