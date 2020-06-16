import sys; sys.path.append('..');
import numpy as np
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
        print('Vector projection: ', grid_projection_osr)
        print('Raster projection: ', image_projection_osr)
        raise Exception('Projections do not match!')

    grid_feature_count = grid_layer.GetFeatureCount()
    
    driver = gdal.GetDriverByName("GTiff") # 'GTiff'
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
            out_name,                           # Location of the saved raster, ignored if driver is memory.
            width,                              # Dataframe width in pixels (e.g. 1920px).
            height,                             # Dataframe height in pixels (e.g. 1280px).
            image.RasterCount,                  # The number of bands required.
            dtype,                              # Datatype of the destination
            creationOptions,                    # Compressions options for non-memory output.
        )

        destination_transform = (ext[0], rgeo[1], 0, ext[3], 0, rgeo[5])
        
        destinationDataframe.SetGeoTransform(destination_transform)
        destinationDataframe.SetProjection(image_projection)

        gdal.PushErrorHandler('CPLQuietErrorHandler')
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


if __name__ == "__main__":
    folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\"
    in_raster = folder + "scaled.tif"
    in_grid = folder + "grid_80m.gpkg"
    out_folder = folder + "tiles_80m\\"


    extract_tiles(in_raster, in_grid, out_folder, "80m_", 8, 8)