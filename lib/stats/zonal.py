import numpy as np
import sys; sys.path.append('/mnt/c/users/caspe/desktop/yellow/lib/')
from osgeo import ogr, gdal, osr
from lib.raster_io import raster_to_array
from lib.vector_io import vector_to_memory
from stats_global import enumerate_stats, global_statistics
from lib.utils_core import progress


def calc_ipq(area, perimeter):
    if perimeter == 0:
        return 0
    else:
        return (4 * np.pi * area) / perimeter ** 2


def overlap_size_calc(extent, raster_transform):
    return np.array([
        (extent[1] - extent[0]) / raster_transform[1],
        (extent[3] - extent[2]) / abs(raster_transform[5]),
    ], dtype=np.int32)


def align_extent(raster_transform, vector_extent, raster_size):
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

    rasterized_x_size = int((vector_max_x - vector_min_x) / pixel_width)
    rasterized_y_size = int((vector_max_y - vector_min_y) / pixel_height)

    rasterized_x_offset = int((vector_min_x - raster_min_x) / pixel_width)
    rasterized_y_offset = int(raster_size[1] - int((vector_min_y - raster_min_y) / pixel_height)) - rasterized_y_size

    if rasterized_x_offset < 0:
        rasterized_x_offset = 0

    if rasterized_y_offset < 0:
        rasterized_y_offset = 0

    if (rasterized_x_offset + rasterized_x_size) > raster_size[0]:
        rasterized_x_offset = rasterized_x_offset - ((rasterized_x_offset + rasterized_x_size) - raster_size[0])

    if (rasterized_y_offset + rasterized_y_size) > raster_size[1]:
        rasterized_y_offset = rasterized_y_offset - ((rasterized_y_offset + rasterized_y_size) - raster_size[1])

    new_vector_extent = np.array([vector_min_x, vector_max_x, vector_min_y, vector_max_y], dtype=np.float32)
    rasterized_size = np.array([rasterized_x_size, rasterized_y_size, pixel_width, pixel_height], dtype=np.float32)
    offset = np.array([rasterized_x_offset, rasterized_y_offset], dtype=np.int32)

    return new_vector_extent, rasterized_size, offset


def get_intersection(extent1, extent2):
    one_bottomLeftX = extent1[0]
    one_topRightX = extent1[1]
    one_bottomLeftY = extent1[2]
    one_topRightY = extent1[3]

    two_bottomLeftX = extent2[0]
    two_topRightX = extent2[1]
    two_bottomLeftY = extent2[2]
    two_topRightY = extent2[3]

    if two_bottomLeftX > one_topRightX:     # Too far east
        return np.array([0, 0, 0, 0], dtype=np.float32)
    elif two_bottomLeftY > one_topRightY:   # Too far north
        return np.array([0, 0, 0, 0], dtype=np.float32)
    elif two_topRightX < one_bottomLeftX:   # Too far west
        return np.array([0, 0, 0, 0], dtype=np.float32)
    elif two_topRightY < one_bottomLeftY:   # Too far south
        return np.array([0, 0, 0, 0], dtype=np.float32)
    else:

        x_min = one_bottomLeftX if one_bottomLeftX > two_bottomLeftX else two_bottomLeftX
        x_max = one_topRightX if one_topRightX < two_topRightX else two_topRightX
        y_min = one_bottomLeftY if one_bottomLeftY > two_bottomLeftY else two_bottomLeftY
        y_max = one_topRightY if one_topRightY < two_topRightY else two_topRightY

        return np.array([x_min, x_max, y_min, y_max], dtype=np.float32)


def get_extent(raster_transform, raster_size):
    bottomRightX = raster_transform[0] + (raster_size[0] * raster_transform[1])
    bottomRightY = raster_transform[3] + (raster_size[1] * raster_transform[5])

    return np.array([raster_transform[0], bottomRightX, bottomRightY, raster_transform[3]], dtype=np.float32)


def rasterize_vector(vector, extent, raster_size, projection, all_touch=False):

    # Create destination dataframe
    driver = gdal.GetDriverByName('MEM')

    destination = driver.Create(
        'in_memory_raster',     # Location of the saved raster, ignored if driver is memory.
        int(raster_size[0]),    # Dataframe width in pixels (e.g. 1920px).
        int(raster_size[1]),    # Dataframe height in pixels (e.g. 1280px).
        1,                      # The number of bands required.
        gdal.GDT_Byte,          # Datatype of the destination.
    )

    destination.SetGeoTransform((extent[0], raster_size[2], 0, extent[3], 0, -raster_size[3]))
    destination.SetProjection(projection)

    # Rasterize and retrieve data
    destination_band = destination.GetRasterBand(1)
    destination_band.Fill(1)

    gdal.RasterizeLayer(destination, [1], vector, burn_values=[0])

    return destination_band.ReadAsArray()


def crop_raster(raster_band, rasterized_size, offset):
    return raster_band.ReadAsArray(int(offset[0]), int(offset[1]), int(rasterized_size[0]), int(rasterized_size[1]))


def calc_shapes(in_vector):
    vector = ogr.Open(in_vector, 1)
    vector_layer = vector.GetLayer(0)
    vector_layer_defn = vector_layer.GetLayerDefn()
    vector_field_counts = vector_layer_defn.GetFieldCount()
    vector_current_fields = []
    
    # Get current fields
    for i in range(vector_field_counts):
        vector_current_fields.append(vector_layer_defn.GetFieldDefn(i).GetName())

    vector_layer.StartTransaction()
    
    # Add missing fields
    for attribute in ['area', 'perimeter', 'ipq']:
        if attribute not in vector_current_fields:
            field_defn = ogr.FieldDefn(attribute, ogr.OFTReal)
            vector_layer.CreateField(field_defn)

    vector_feature_count = vector_layer.GetFeatureCount()
    for i in range(vector_feature_count):
        vector_feature = vector_layer.GetNextFeature()
    
        try:
            vector_geom = vector_feature.GetGeometryRef()
        except:
            vector_geom.Buffer(0)
            Warning('Invalid geometry at : ', i)

        if vector_geom is None:
            raise Exception('Invalid geometry. Could not fix.')

        vector_area = vector_geom.GetArea()
        vector_perimeter = vector_geom.Boundary().Length()
        vector_ipq = calc_ipq(vector_area, vector_perimeter)

        vector_feature.SetField('area', vector_area)
        vector_feature.SetField('perimeter', vector_perimeter)
        vector_feature.SetField('ipq', vector_ipq)

        vector_layer.SetFeature(vector_feature)
        
        progress(i, vector_feature_count, name='shape')
        
    vector_layer.CommitTransaction()


def calc_zonal(in_vector, in_rasters=[], prefixes=[], stats=['mean', 'med', 'std']):

    # Translate stats to integers
    stats_translated = enumerate_stats(stats)

    # Read the raster meta:
    raster_origin = gdal.Open(in_rasters[0])

    # Read the vector
    vector = vector_to_memory(in_vector)
    vector_layer = vector.GetLayer()

    # Check that projections match
    vector_projection = vector_layer.GetSpatialRef()
    raster_projection = raster_origin.GetProjection()
    raster_projection_osr = osr.SpatialReference(raster_projection)
    vector_projection_osr = osr.SpatialReference()
    vector_projection_osr.ImportFromWkt(str(vector_projection))

    if not vector_projection_osr.IsSame(raster_projection_osr):
        print('Vector projection: ', vector_projection_osr)
        print('Raster projection: ', raster_projection_osr)
        raise Exception('Projections do not match!')

    # Read raster data in overlap
    raster_transform = np.array(raster_origin.GetGeoTransform(), dtype=np.float32)
    raster_size = np.array([raster_origin.RasterXSize, raster_origin.RasterYSize], dtype=np.int32)

    raster_extent = get_extent(raster_transform, raster_size)

    vector_extent = np.array(vector_layer.GetExtent(), dtype=np.float32)
    overlap_extent = get_intersection(raster_extent, vector_extent)

    if overlap_extent is False:
        print('raster_extent: ', raster_extent)
        print('vector_extent: ', vector_extent)
        raise Exception('Vector and raster do not overlap!')

    overlap_aligned_extent, overlap_aligned_rasterized_size, overlap_aligned_offset = align_extent(raster_transform, overlap_extent, raster_size)
    overlap_transform = np.array([
        overlap_aligned_extent[0],
        raster_transform[1],
        0,
        overlap_aligned_extent[3],
        0,
        raster_transform[5],
    ], dtype=np.float32)
    overlap_size = overlap_size_calc(overlap_aligned_extent, raster_transform)

    # Loop the features
    vector_driver = ogr.GetDriverByName('Memory')
    vector_feature_count = vector_layer.GetFeatureCount()
    vector_layer.StartTransaction()

    # Create fields
    vector_layer_defn = vector_layer.GetLayerDefn()
    vector_field_counts = vector_layer_defn.GetFieldCount()
    vector_current_fields = []
    
    # Get current fields
    for i in range(vector_field_counts):
        vector_current_fields.append(vector_layer_defn.GetFieldDefn(i).GetName())
    
    # Add fields where missing
    for stat in stats:
        for i in range(len(in_rasters)):
            field_name = f'{prefixes[i]}{stat}'
            if field_name not in vector_current_fields:
                field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
                vector_layer.CreateField(field_defn)

    rasterized_features = []
    offsets = []
    sizes = []
    for raster_index, raster_value in enumerate(in_rasters):

        columns = {}
        for stat in stats:
            columns[prefixes[raster_index] + stat] = []

        fits_in_memory = True
        try:
            raster_data = raster_to_array(raster_value, crop=[
                overlap_aligned_offset[0],
                overlap_aligned_offset[1],
                overlap_aligned_rasterized_size[0],
                overlap_aligned_rasterized_size[1],
            ])
        except:
            fits_in_memory = False
            print("Raster does not fit in memory.. Doing IO for each feature.")

        raster_data = None

        for n in range(vector_feature_count):
            vector_feature = vector_layer.GetNextFeature()

            if raster_index == 0:

                try:
                    vector_geom = vector_feature.GetGeometryRef()
                except:
                    vector_geom.Buffer(0)
                    Warning('Invalid geometry at : ', n)

                if vector_geom is None:
                    raise Exception('Invalid geometry. Could not fix.')

                feature_extent = vector_geom.GetEnvelope()

                # Create temp layer
                temp_vector_datasource = vector_driver.CreateDataSource(f'vector_{n}')
                temp_vector_layer = temp_vector_datasource.CreateLayer('temp_polygon', vector_projection, ogr.wkbPolygon)
                temp_vector_layer.CreateFeature(vector_feature.Clone())

                feature_aligned_extent, feature_aligned_rasterized_size, feature_aligned_offset = align_extent(overlap_transform, feature_extent, overlap_size)
                rasterized_features.append(rasterize_vector(temp_vector_layer, feature_aligned_extent, feature_aligned_rasterized_size, raster_projection))

                offsets.append(feature_aligned_offset)
                sizes.append(feature_aligned_rasterized_size)

            if fits_in_memory is True:
                cropped_raster = raster_data[
                    offsets[n][1]:offsets[n][1] + int(sizes[n][1]), # X
                    offsets[n][0]:offsets[n][0] + int(sizes[n][0]), # Y
                ]
            else:
                cropped_raster = raster_to_array(raster_value, crop=[
                    overlap_aligned_offset[0] + offsets[n][0],
                    overlap_aligned_offset[1] + offsets[n][1],
                    int(sizes[n][0]),
                    int(sizes[n][1]),
                ])

            if rasterized_features[n] is None:
                for stat in stats:
                    field_name = f'{prefixes[raster_index]}{stat}'
                    vector_feature.SetField(field_name, None)
            elif cropped_raster is None:
                for stat in stats:
                    field_name = f'{prefixes[raster_index]}{stat}'
                    vector_feature.SetField(field_name, None)
            else:
                raster_data_masked = np.ma.masked_array(cropped_raster, mask=rasterized_features[n], dtype='float32').compressed()
                zonal_stats = global_statistics(raster_data_masked, translated_stats=stats_translated)
                nodata = np.isnan(zonal_stats).any()

                for index, stat in enumerate(stats):
                    field_name = f'{prefixes[raster_index]}{stat}'
                    if nodata is True:
                        vector_feature.SetField(field_name, None)
                    else:
                        vector_feature.SetField(f'{prefixes[raster_index]}{stat}', float(zonal_stats[index]))

            vector_layer.SetFeature(vector_feature)

            progress(n, vector_feature_count, name=prefixes[raster_index])
        
        vector_layer.ResetReading()

    vector_layer.CommitTransaction()
