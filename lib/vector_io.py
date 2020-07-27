import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_metadata
from osgeo import gdal, ogr, osr


def intersection_rasters(raster_1, raster_2):
    raster_1 = raster_1 if isinstance(raster_1, gdal.Dataset) else gdal.Open(raster_1)
    raster_2 = raster_2 if isinstance(raster_2, gdal.Dataset) else gdal.Open(raster_2)

    img_1 = raster_to_metadata(raster_1)
    img_2 = raster_to_metadata(raster_2)

    driver = ogr.GetDriverByName('Memory')
    dst_source = driver.CreateDataSource('clipped_rasters')
    dst_srs = ogr.osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    dst_layer = dst_source.CreateLayer('unused', dst_srs, geom_type=ogr.wkbPolygon)

    geom1 = gdal.OpenEx(img_1['footprint'])
    layer1 = geom1.GetLayer()
    feature1 = layer1.GetFeature(0)
    feature1_geom = feature1.GetGeometryRef()

    geom2 = gdal.OpenEx(img_2['footprint'])
    layer2 = geom2.GetLayer()
    feature2 = layer2.GetFeature(0)
    feature2_geom = feature2.GetGeometryRef()

    if feature2_geom.Intersects(feature1_geom):
        intersection = feature2_geom.Intersection(feature1_geom)
        dstfeature = ogr.Feature(dst_layer.GetLayerDefn())
        dstfeature.SetGeometry(intersection)
        dst_layer.CreateFeature(dstfeature)
        dstfeature.Destroy()
        
        return dst_source
    else:
        return False


def vector_mask(vector, raster):
    raster = raster if isinstance(raster, gdal.Dataset) else gdal.Open(raster)
    vector = vector if isinstance(vector, ogr.DataSource) else ogr.Open(vector)

    # Create destination dataframe
    driver = gdal.GetDriverByName('MEM')

    destination = driver.Create(
        'in_memory_raster',     # Location of the saved raster, ignored if driver is memory.
        raster.RasterXSize,     # Dataframe width in pixels (e.g. 1920px).
        raster.RasterYSize,     # Dataframe height in pixels (e.g. 1280px).
        1,                      # The number of bands required.
        gdal.GDT_Byte,          # Datatype of the destination.
    )

    destination.SetGeoTransform(raster.GetGeoTransform())
    destination.SetProjection(raster.GetProjection())

    # Rasterize and retrieve data
    destination_band = destination.GetRasterBand(1)
    destination_band.Fill(1)

    gdal.RasterizeLayer(destination, [1], vector.GetLayer(), burn_values=[0], options=['ALL_TOUCHED=TRUE'])

    return destination
    # return destination_band.ReadAsArray()


def vector_to_memory(in_vector, layer=0):
    try:
        if isinstance(in_vector, ogr.DataSource):
            src = in_vector
        else:
            src = ogr.Open(in_vector)
    except:
        raise Exception("Could not read input raster")

    driver = ogr.GetDriverByName("MEMORY")
    copy = driver.CreateDataSource("copy")
    copy.CopyLayer(src.GetLayer(layer), "copy", ["OVERWRITE=YES"])

    return copy

if __name__ == "__main__":
    bob = vector_to_memory("../geometry/denmark_10km_grid.gpkg")
    import pdb; pdb.set_trace()