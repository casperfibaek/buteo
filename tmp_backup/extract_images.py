import sys

sys.path.append("..")
import numpy as np
from math import ceil, floor
from osgeo import gdal, ogr, osr
from lib.raster_io import raster_to_array, array_to_raster, raster_to_metadata
from lib.utils_core import progress


def extract_tiles(image, grid, out_folder, prefix="", height=32, width=32):
    grid = grid if isinstance(grid, ogr.DataSource) else ogr.Open(grid)
    image = image if isinstance(image, gdal.Dataset) else gdal.Open(image)
    grid_layer = grid.GetLayer(0)

    # Check that projections match
    grid_projection = grid_layer.GetSpatialRef()
    grid_projection_osr = osr.SpatialReference()
    grid_projection_osr.ImportFromWkt(str(grid_projection))

    image_projection = image.GetProjection()
    image_projection_osr = osr.SpatialReference(image_projection)

    if not grid_projection_osr.IsSame(image_projection_osr):
        print("Vector projection: ", grid_projection_osr)
        print("Raster projection: ", image_projection_osr)
        raise Exception("Projections do not match!")

    grid_feature_count = grid_layer.GetFeatureCount()

    driver = gdal.GetDriverByName("GTiff")  # 'GTiff'
    inputBand = image.GetRasterBand(1)
    creationOptions = []
    dtype = inputBand.DataType

    rgeo = image.GetGeoTransform()

    progress(0, grid_feature_count, name="extracting")
    for n in range(grid_feature_count):
        grid_feature = grid_layer.GetNextFeature()
        fid = grid_feature.GetFID()
        vector_geom = grid_feature.GetGeometryRef()
        ext = vector_geom.GetEnvelope()

        out_name = f"{out_folder}{prefix}{fid}.tif"

        destinationDataframe = driver.Create(
            out_name,  # Location of the saved raster, ignored if driver is memory.
            width,  # Dataframe width in pixels (e.g. 1920px).
            height,  # Dataframe height in pixels (e.g. 1280px).
            image.RasterCount,  # The number of bands required.
            dtype,  # Datatype of the destination
            creationOptions,  # Compressions options for non-memory output.
        )

        destination_transform = (ext[0], rgeo[1], 0, ext[3], 0, rgeo[5])

        destinationDataframe.SetGeoTransform(destination_transform)
        destinationDataframe.SetProjection(image_projection)

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        gdal.Warp(
            destinationDataframe,
            image,
            xRes=width,
            yRes=height,
            format="GTiff",
            targetAlignedPixels=True,
            multithread=True,
        )
        gdal.PopErrorHandler()

        progress(n, grid_feature_count, name="extracting")

    return 1


def extract_tiles_numpy(image, grid, out_folder, name="", height=32, width=32):
    grid = grid if isinstance(grid, ogr.DataSource) else ogr.Open(grid)
    image = image if isinstance(image, gdal.Dataset) else gdal.Open(image)

    grid_layer = grid.GetLayer(0)

    # Check that projections match
    grid_projection = grid_layer.GetSpatialRef()
    grid_projection_osr = osr.SpatialReference()
    grid_projection_osr.ImportFromWkt(str(grid_projection))

    image_projection = image.GetProjection()
    image_projection_osr = osr.SpatialReference(image_projection)

    if not grid_projection_osr.IsSame(image_projection_osr):
        print("Vector projection: ", grid_projection_osr)
        print("Raster projection: ", image_projection_osr)
        raise Exception("Projections do not match!")

    grid_feature_count = grid_layer.GetFeatureCount()

    inputBand = image.GetRasterBand(1)
    creationOptions = []
    dtype = inputBand.DataType

    rgeo = image.GetGeoTransform()

    # Copy stuff to memory for faster access
    gdal.Translate(f"/vsimem/{name}.tif", image)

    raster_driver = gdal.GetDriverByName("MEM")
    vector_driver = ogr.GetDriverByName("Memory")
    vector_source = vector_driver.CreateDataSource("memData")
    vector_source.CopyLayer(grid.GetLayer(0), "grid_features")

    grid_layer = vector_source.GetLayer(0)

    images = []
    fids = []

    progress(0, grid_feature_count, name="extracting")
    for n in range(grid_feature_count):
        grid_feature = grid_layer.GetNextFeature()
        fid = grid_feature.GetFID()
        vector_geom = grid_feature.GetGeometryRef()
        ext = vector_geom.GetEnvelope()

        out_name = f"/vsimem/{name}_{n}.tif"

        destinationDataframe = raster_driver.Create(
            out_name,  # Location of the saved raster, ignored if driver is memory.
            width,  # Dataframe width in pixels (e.g. 1920px).
            height,  # Dataframe height in pixels (e.g. 1280px).
            image.RasterCount,  # The number of bands required.
            dtype,  # Datatype of the destination
            creationOptions,  # Compressions options for non-memory output.
        )

        destination_transform = (ext[0], rgeo[1], 0, ext[3], 0, rgeo[5])

        destinationDataframe.SetGeoTransform(destination_transform)
        destinationDataframe.SetProjection(image_projection)

        gdal.PushErrorHandler("CPLQuietErrorHandler")
        gdal.Warp(
            destinationDataframe,
            f"/vsimem/{name}.tif",
            xRes=width,
            yRes=height,
            format="MEM",
            targetAlignedPixels=True,
            multithread=True,
        )
        gdal.PopErrorHandler()

        images.append(raster_to_array(destinationDataframe))
        fids.append(fid)

        progress(n, grid_feature_count, name=f"{name} extracting")

    stacked = np.stack(images)
    fids = np.array(fids, dtype="int64")

    np.save(out_folder + name + ".npy", stacked)
    np.save(out_folder + name + "_fids.npy", fids)

    return stacked


# Channel last format
def array_to_blocks(arr, block_shape):
    if len(arr.shape) == 1:
        return (
            arr[0 : arr.shape[0] - ceil(arr.shape[0] % block_shape[0]),]
            .reshape(arr.shape[0] // block_shape[0], block_shape[0],)
            .swapaxes(1)
            .reshape(-1, block_shape[0])
        )

    elif len(arr.shape) == 2:
        return (
            arr[
                0 : arr.shape[0] - ceil(arr.shape[0] % block_shape[0]),
                0 : arr.shape[1] - ceil(arr.shape[1] % block_shape[1]),
            ]
            .reshape(
                arr.shape[0] // block_shape[0],
                block_shape[0],
                arr.shape[1] // block_shape[1],
                block_shape[1],
            )
            .swapaxes(1, 2)
            .reshape(-1, block_shape[0], block_shape[1])
        )

    elif len(arr.shape) == 3:
        return (
            arr[
                0 : arr.shape[0] - ceil(arr.shape[0] % block_shape[0]),
                0 : arr.shape[1] - ceil(arr.shape[1] % block_shape[1]),
                0 : arr.shape[2] - ceil(arr.shape[2] % block_shape[2]),
            ]
            .reshape(
                arr.shape[0] // block_shape[0],
                block_shape[0],
                arr.shape[1] // block_shape[1],
                block_shape[1],
                arr.shape[2] // block_shape[2],
                block_shape[2],
            )
            .swapaxes(1, 2)
            .reshape(-1, block_shape[0], block_shape[1], block_shape[2])
        )

    else:
        raise Exception("Unable to handle more than 3 dimensions")
