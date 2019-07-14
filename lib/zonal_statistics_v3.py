import multiprocessing
import sys
import numpy as np
import numpy.ma as ma
from time import time
from glob import glob
from osgeo import ogr, gdal, osr

sys.path.append('../lib')
from raster_stats_v2 import calc_stats
from utils import progress


def _align_extent(raster_transform, vector_extent, raster_size):
    pixel_width = abs(raster_transform[1])
    pixel_height = abs(raster_transform[5])

    raster_min_x = raster_transform[0]
    raster_max_x = raster_min_x + (raster_size[0] * pixel_width)
    raster_max_y = raster_transform[3]
    raster_min_y = raster_max_y + (raster_size[1] * -pixel_width)

    vector_min_x = vector_extent[0]
    vector_max_x = vector_extent[1]
    vector_min_y = vector_extent[2]
    vector_max_y = vector_extent[3]

    # Align the two extents
    vector_min_x = vector_min_x - (vector_min_x - raster_min_x) % pixel_width
    vector_max_x = vector_max_x + (vector_max_x - raster_min_x) % pixel_width
    vector_min_y = vector_min_y - (vector_min_y - raster_max_y) % pixel_height
    vector_max_y = vector_max_y + (vector_max_y - raster_max_y) % pixel_height

    rasterized_x_size = int((vector_max_x - vector_min_x) * pixel_width)
    rasterized_y_size = int((vector_max_y - vector_min_y) * pixel_height)

    rasterized_x_offset = int((vector_min_x - raster_min_x) / pixel_width)
    rasterized_y_offset = raster_size[1] - int((vector_min_y - raster_min_y) / pixel_height)

    if rasterized_x_offset < 0:
        rasterized_x_offset = 0

    if rasterized_y_offset < 0:
        rasterized_y_offset = 0

    if (rasterized_x_offset + rasterized_x_size) > raster_size[0]:
        rasterized_x_offset = rasterized_x_offset - ((rasterized_x_offset + rasterized_x_size) - raster_size[0])

    if (rasterized_y_offset + rasterized_y_size) > raster_size[1]:
        rasterized_y_offset = rasterized_y_offset - ((rasterized_y_offset + rasterized_y_size) - raster_size[1])

    return np.array([
        vector_min_x, vector_max_x, vector_min_y, vector_max_y,
        rasterized_x_size, rasterized_y_size, pixel_width, pixel_height,
        rasterized_x_offset, rasterized_y_offset
    ], dtype=float)


align_extent = np.vectorize(_align_extent, otypes=[float])


def _get_intersection(extent1, extent2):
    one_bottomLeftX = extent1[0]
    one_topRightX = extent1[1]
    one_bottomLeftY = extent1[2]
    one_topRightY = extent1[3]

    two_bottomLeftX = extent2[0]
    two_topRightX = extent2[1]
    two_bottomLeftY = extent2[2]
    two_topRightY = extent2[3]

    if two_bottomLeftX > one_topRightX:     # Too far east
        return np.array([], dtype=bool)
    elif two_bottomLeftY > one_topRightY:   # Too far north
        return np.array([], dtype=bool)
    elif two_topRightX < one_bottomLeftX:   # Too far west
        return np.array([], dtype=bool)
    elif two_topRightY < one_bottomLeftY:   # Too far south
        return np.array([], dtype=bool)
    else:
        return np.array([
            max(one_bottomLeftX, two_bottomLeftX),    # minX of intersection
            min(one_topRightX, two_topRightX),        # maxX of intersection
            max(one_bottomLeftY, two_bottomLeftY),    # minY of intersection
            min(one_topRightY, two_topRightY),        # maxY of intersection
        ], dtype=float)


get_intersection = np.vectorize(_get_intersection, otypes=[float])


def _get_extent(transform, raster_size):
    bottomRightX = transform[0] + (raster_size[0] * transform[1])
    bottomRightY = transform[3] + (raster_size[1] * transform[5])

    # minX, maxX, minY, maxY
    return np.array([transform[0], bottomRightX, bottomRightY, transform[3]], dtype=float)


get_extent = np.vectorize(_get_extent, otypes=[float])


# vect_extent = (minX, maxX, minY, maxY)
def rasterize_vector(vector, extent, projection):

    # Create destination dataframe
    driver = gdal.GetDriverByName('MEM')
    destination = driver.Create(
        'in_memory_raster',                         # Location of the saved raster, ignored if driver is memory.
        extent[4],                                  # Dataframe width in pixels (e.g. 1920px).
        extent[5],                                  # Dataframe height in pixels (e.g. 1280px).
        1,                                          # The number of bands required.
        gdal.GDT_Byte,                              # Datatype of the destination.
    )

    destination.SetGeoTransform((extent[0], extent[6], 0, extent[3], 0, -extent[7]))
    destination.SetProjection(projection)

    # Rasterize and retrieve data
    destination_band = destination.GetRasterBand(1).Fill(1)
    gdal.RasterizeLayer(destination, [1], vector, burn_values=[0])
    data = destination_band.ReadAsArray()

    return data


def crop_raster(raster_band, aligned_extent):
    return raster_band.ReadAsArray(aligned_extent[8], aligned_extent[9], aligned_extent[4], aligned_extent[5])


def zonal_rasterize(vector, rast, prefix='', stats=['mean', 'med', 'std']):

    # Read the raster:
    raster = gdal.Open(rast)
    raster_band = raster.GetRasterBand(1)

    # Read the vector
    vector = ogr.Open(vect, 1)
    vector_layer = vector.GetLayer(0)

    # Check that projections match
    vector_projection = vector_layer.GetSpatialRef()
    raster_projection = raster.GetProjection()
    raster_projection_osr = osr.SpatialReference(raster_projection)
    vector_projection_osr = osr.SpatialReference()
    vector_projection_osr.ImportFromWkt(str(vector_projection))

    if not vector_projection_osr.IsSame(raster_projection_osr):
        print('Projections do not match!')
        print('Vector projection: ', vector_projection_osr)
        print('Raster projection: ', raster_projection_osr)
        exit()

    # Read raster data in overlap
    raster_transform = np.array(raster.GetGeoTransform(), dtype=float)
    raster_size = np.array([raster.RasterXSize, raster.RasterYSize], dtype=int)

    raster_extent = get_extent(raster_transform, raster_size)
    vector_extent = np.array(vector_layer.GetExtent(), dtype=float)
    overlap_extent = get_intersection(raster_extent, vector_extent)

    if len(overlap_extent) is 0:
        print('Vector and raster do not overlap!')
        print('raster_extent: ', raster_extent)
        print('vector_extent: ', vector_extent)
        exit()

    # overlap_extent_aligned = align_extent(raster_transform, overlap_extent, raster_x_size, raster_y_size)
    overlap_extent_aligned = align_extent(raster_transform, overlap_extent, raster_size)
    overlap_transform = np.array([overlap_extent_aligned[0], raster_transform[1], 0, overlap_extent_aligned[3], 0, raster_transform[5]], dtype=float)
    overlap_size = np.array([
        (overlap_extent_aligned[1] - overlap_extent_aligned[0]) / raster_transform[1],
        (overlap_extent_aligned[3] - overlap_extent_aligned[2]) / abs(raster_transform[5]),
    ], dtype=int)

    raster_data = crop_raster(raster_band, overlap_extent_aligned)

    # Create fields
    vector_layer_defn = vector_layer.GetLayerDefn()
    for stat in stats:
        if vector_layer_defn.GetFieldIndex(f'{prefix}{stat}') is -1:
            field_defn = ogr.FieldDefn(f'{prefix}{stat}', ogr.OFTReal)
            vector_layer.CreateField(field_defn)

    # Loop the features
    vector_driver = ogr.GetDriverByName('Memory')
    vector_feature_count = vector_layer.GetFeatureCount()

    for n in range(vector_feature_count):
        vector_feature = vector_layer.GetNextFeature()
        feature_extent = vector_feature.GetGeometryRef().GetEnvelope()

        # Create temp layer
        temp_vector_datasource = vector_driver.CreateDataSource(f'vector_{n}')
        temp_vector_layer = temp_vector_datasource.CreateLayer('temp_polygon', vector_projection, ogr.wkbPolygon)
        temp_vector_layer.CreateFeature(vector_feature.Clone())
        temp_vector_layer.SyncToDisk()

        # feature_aligned_extent = align_extent(raster_transform, feature_extent, raster_x_size, raster_y_size)
        feature_aligned_extent = align_extent(overlap_transform, feature_extent, overlap_size)

        feature_rasterized = rasterize_vector(temp_vector_layer, feature_aligned_extent, raster_projection)

        cropped_raster = raster_data[
            feature_aligned_extent[8]:feature_aligned_extent[4],
            feature_aligned_extent[9]:feature_aligned_extent[5],
        ]

        if feature_rasterized is None:
            for key in stats:
                vector_feature.SetField(f'{prefix}{key}', None)
            print(f'feature_rasterized was None. FID: {vector_feature.GetFID()}')
        elif cropped_raster is None:
            for key in stats:
                vector_feature.SetField(f'{prefix}{key}', None)
            print(f'cropped_raster was None. FID: {vector_feature.GetFID()}')
        elif feature_rasterized.size != cropped_raster.size:
            for key in stats:
                vector_feature.SetField(f'{prefix}{key}', None)
            print(f'feature and raster did not match for: {vector_feature.GetFID()}')
        else:
            raster_data_masked = ma.masked_array(cropped_raster, mask=feature_rasterized).compressed()
            zonal_stats = calc_stats(raster_data_masked, stats)

            for key in zonal_stats.keys():
                vector_feature.SetField(f'{prefix}{key}', float(zonal_stats[key]))

        vector_layer.SetFeature(vector_feature)

        progress(n, vector_feature_count - 1)

    vector_layer.SyncToDisk()


if __name__ == '__main__':
    # vect = 'C:\\Users\\CFI\\Desktop\\scratch_pad\\test_zones.shp'
    # rast = 'C:\\Users\\CFI\\Desktop\\scratch_pad\\dry_b04.tif'
    vect = 'E:\\SATF\\phase_IV_urban-classification\\urban_classification_zonal.shp'
    rast = glob('E:\\SATF\\data\\s2\\*tif')

    before = time()

    for i, r in enumerate(rast):
        zonal_rasterize(vect, r, prefix=f'{i}_', stats=['mean', 'med', 'mad', 'std', 'kurt', 'skew', 'iqr'])

    after = time()
    dif = after - before
    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = "{0:.2f}".format(dif % 60)
    print(f"Zonal_stats took: {hours}h {minutes}m {seconds}s")

    # zonal_rasterize(vect, rast, stats=['mean', 'med', 'mad', 'std', 'kurt', 'skew', 'iqr'])
    # zonal_rasterize(vect, rast)
