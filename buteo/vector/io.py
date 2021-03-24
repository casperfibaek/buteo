import sys; sys.path.append('../../')
import os, json
import numpy as np
import pandas as pd
import osgeo
from osgeo import gdal, ogr, osr

from buteo.raster.io import raster_to_metadata
from buteo.gdal_utils import progress, parse_projection

# TODO:
#   - rasterize - with antialiasing/weights
#   - join by attribute + summary
#   - join by location + summary
#   - intersection, buffer, union, clip, erase
#   - sanity checks: vectors_intersect, is_not_empty, does_vectors_match, match_vectors
#   - repair vector
#   - multithreaded processing


def vector_to_reference(vector, writeable=False):
    try:
        if isinstance(vector, ogr.DataSource):  # Dataset already OGR dataframe.
            return vector
        else:
            opened = ogr.Open(vector, 1) if writeable else ogr.Open(vector, 0)
            
            if opened is None:
                raise Exception("Could not read input raster")

            return opened
    except:
        raise Exception("Could not read input raster")


def vector_to_memory(vector):
    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref, process_layers="all")

    driver = ogr.GetDriverByName("Memory")

    basename = metadata["basename"] if metadata["basename"] is not None else "mem_vector"
    copy = driver.CreateDataSource(basename)

    for layer_idx in range(metadata["layer_count"]):
        layername = metadata["layers"][layer_idx]["layer_name"]
        copy.CopyLayer(ref.GetLayer(layer_idx), layername, ["OVERWRITE=YES"])

    return copy


def vector_to_metadata(vector, process_layers="first"):
    try:
        vector = vector if isinstance(vector, ogr.DataSource) else ogr.Open(vector)
    except:
        raise Exception("Could not read input vector")

    abs_path = os.path.abspath(vector.GetName())

    metadata = {
        "path": abs_path,
        "basename": os.path.basename(abs_path),
        "filetype": os.path.splitext(os.path.basename(abs_path))[1],
        "name": os.path.splitext(os.path.basename(abs_path))[0],
        "layer_count": vector.GetLayerCount(),
        "layers": []
    }

    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326) # WGS84, latlng

    for layer_index in range(metadata["layer_count"]):
        layer = vector.GetLayerByIndex(layer_index)

        min_x, max_x, min_y, max_y = layer.GetExtent()

        layer_dict = {
            "layer_name": layer.GetName(),
            "minx": min_x,
            "maxx": max_x,
            "miny": min_y,
            "maxy": max_y,
            "extent": [min_x, max_y, max_x, min_y],
            "extent_ogr": layer.GetExtent(),
            "fid_column": layer.GetFIDColumn(),
            "feature_count": layer.GetFeatureCount(),
        }

        projection = layer.GetSpatialRef().ExportToWkt()
        projection_osr = osr.SpatialReference()
        projection_osr.ImportFromWkt(projection)

        if layer_index == 0:
            metadata["projection"] = projection
            metadata["projection_osr"] = projection_osr

        layer_dict["projection"] = projection
        layer_dict["projection_osr"] = projection_osr
    
        bottom_left = ogr.Geometry(ogr.wkbPoint)
        top_left = ogr.Geometry(ogr.wkbPoint)
        top_right = ogr.Geometry(ogr.wkbPoint)
        bottom_right = ogr.Geometry(ogr.wkbPoint)

        bottom_left.AddPoint(min_x, min_y)
        top_left.AddPoint(min_x, max_y)
        top_right.AddPoint(max_x, max_y)
        bottom_right.AddPoint(max_x, min_y)

        if not projection_osr.IsSame(wgs84):
            tx = osr.CoordinateTransformation(projection_osr, wgs84)

            bottom_left.Transform(tx)
            top_left.Transform(tx)
            top_right.Transform(tx)
            bottom_right.Transform(tx)

            layer_dict["extent_wgs84"] = [
                top_left.GetX(),
                top_left.GetY(),
                bottom_right.GetX(),
                bottom_right.GetY(),
            ]

        else:
            layer_dict["extent_wgs84"] = layer_dict["extent"]
        
        # WKT has latitude first, geojson has longitude first
        coord_array = [
            [bottom_left.GetY(), bottom_left.GetX()],
            [top_left.GetY(), top_left.GetX()],
            [top_right.GetY(), top_right.GetX()],
            [bottom_right.GetY(), bottom_right.GetX()],
            [bottom_left.GetY(), bottom_left.GetX()],
        ]

        wkt_coords = ""
        for coord in coord_array:
            wkt_coords += f"{coord[1]} {coord[0]}, "
        wkt_coords = wkt_coords[:-2] # Remove the last ", "

        layer_dict["extent_wkt"] = f"POLYGON (({wkt_coords}))"

        layer_dict["extent_geojson_dict"] = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [coord_array],
            },
        }
        layer_dict["extent_geojson"] = json.dumps(layer_dict["extent_geojson_dict"])

        layer_defn = layer.GetLayerDefn()

        layer_dict["field_count"] = layer_defn.GetFieldCount()
        layer_dict["geom_type"] = ogr.GeometryTypeToName(layer_defn.GetGeomType())
        layer_dict["geom_type_ogr"] = layer_defn.GetGeomType()

        layer_dict["field_names"] = []
        layer_dict["field_types"] = []

        for field_index in range(layer_dict["field_count"]):
            field_defn = layer_defn.GetFieldDefn(field_index)
            layer_dict["field_names"].append(field_defn.GetName())
            layer_dict["field_types"].append(field_defn.GetFieldTypeName(field_defn.GetType()))

        if process_layers == "first":

            for key, value in layer_dict.items():
                metadata[key] = value
            
            del metadata["layers"]
            break

        metadata["layers"].append(layer_dict)

    return metadata


def vector_to_disc(vector, output_path, driver="GPKG"):
    assert isinstance(vector, ogr.DataSource), "Input not a vector datasource."

    driver = ogr.GetDriverByName(driver)
    assert driver != None, "Unable to parse driver."

    metadata = vector_to_metadata(vector, process_layers="all")

    copy = driver.CreateDataSource(output_path)

    for layer_idx in range(metadata["layer_count"]):
        layer_name = metadata["layers"][layer_idx]["layer_name"]
        copy.CopyLayer(vector.GetLayer(layer_idx), str(layer_name), ["OVERWRITE=YES"])

    copy = None
    return None


def is_vector(vector):
    if isinstance(vector, ogr.DataSource):
        return True
    if isinstance(vector, str):
        ref = ogr.Open(vector, 0)
        if isinstance(ref, ogr.DataSource):
            ref = None
            return True
    
    return False


def reproject_vector(vector, target, output=None):
    origin = vector_to_reference(vector)
    metadata = vector_to_metadata(origin, process_layers="all")

    origin_projection = metadata["projection_osr"]
    target_projection = parse_projection(target)

    # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
    if int(osgeo.__version__[0]) >= 3:
        origin_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    coord_trans = osr.CoordinateTransformation(origin_projection, target_projection)

    driver = ogr.GetDriverByName('Memory')

    destination = driver.CreateDataSource(metadata["name"])

    for layer_idx in range(len(metadata["layers"])):
        origin_layer = origin.GetLayerByIndex(layer_idx)
        origin_layer_defn = origin_layer.GetLayerDefn()

        layer_dict = metadata["layers"][layer_idx]
        layer_name = layer_dict["layer_name"]
        layer_geom_type = layer_dict["geom_type_ogr"]

        destination_layer = destination.CreateLayer(layer_name, target_projection, layer_geom_type)
        destination_layer_defn = destination_layer.GetLayerDefn()

        # Copy field definitions
        origin_layer_defn = origin_layer.GetLayerDefn()
        for i in range(0, origin_layer_defn.GetFieldCount()):
            field_defn = origin_layer_defn.GetFieldDefn(i)
            destination_layer.CreateField(field_defn)

        # Loop through the input features
        for _ in range(origin_layer.GetFeatureCount()):
            feature = origin_layer.GetNextFeature()
            geom = feature.GetGeometryRef()
            geom.Transform(coord_trans)

            new_feature = ogr.Feature(destination_layer_defn)
            new_feature.SetGeometry(geom)

            # Copy field values
            for i in range(0, destination_layer_defn.GetFieldCount()):
                new_feature.SetField(destination_layer_defn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))

            destination_layer.CreateFeature(new_feature)
            
        destination_layer.ResetReading()
        destination_layer.CommitTransaction()

    if output is not None:
        vector_to_disc(destination, output_path=output)
        return output
    else:
        return destination
    



def vector_get_attribute_table(vector, process_layers="first", geom=False):
    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref, process_layers=process_layers)

    dataframes = []

    for vector_layer in range(metadata["layer_count"]):
        attribute_table_header = None
        feature_count = None

        if process_layers != "first":
            attribute_table_header = metadata["layers"][vector_layer]["field_names"]
            feature_count = metadata["layers"][vector_layer]["feature_count"]
        else:
            attribute_table_header = metadata["field_names"]
            feature_count = metadata["feature_count"]

        attribute_table = []

        layer = ref.GetLayer(vector_layer)

        for _ in range(feature_count):
            feature = layer.GetNextFeature()
            attributes = [feature.GetFID()]

            for field_name in attribute_table_header:
                attributes.append(feature.GetField(field_name))

            if geom:
                geom_defn = feature.GetGeometryRef()
                attributes.append(geom_defn.ExportToIsoWkt())
            
            attribute_table.append(attributes)

        attribute_table_header.insert(0, "fid")

        if geom:
            attribute_table_header.append("geom")
        
        df = pd.DataFrame(attribute_table, columns=attribute_table_header)

        if process_layers == "first": return df
        
        dataframes.append(df)

    return dataframes
    

def intersection_rasters(raster_1, raster_2):
    raster_1 = raster_1 if isinstance(raster_1, gdal.Dataset) else gdal.Open(raster_1)
    raster_2 = raster_2 if isinstance(raster_2, gdal.Dataset) else gdal.Open(raster_2)

    img_1 = raster_to_metadata(raster_1)
    img_2 = raster_to_metadata(raster_2)

    driver = ogr.GetDriverByName('Memory')
    dst_source = driver.CreateDataSource('clipped_rasters')
    dst_srs = osr.SpatialReference()
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

    destination:gdal.Dataset = driver.Create(
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


def rasterize_vector(vector, extent, raster_size, projection, all_touch=False, optim="raster", band=1, antialias=False):

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
    destination_band = destination.GetRasterBand(band)
    destination_band.Fill(1)

    if antialias is False:
        options = []
        if all_touch == True:
            options.append("ALL_TOUCHED=TRUE")
        
        if optim == "raster":
            options.append("OPTIM=RASTER")
        elif optim == "vector":
            options.append("OPTIM=VECTOR")
        else:
            options.append("OPTIM=AUTO")

        gdal.RasterizeLayer(destination, [1], vector, burn_values=[0], options=options)

    return destination_band.ReadAsArray()


def calc_ipq(area, perimeter):
    if perimeter == 0:
        return 0
    else:
        return (4 * np.pi * area) / perimeter ** 2


def calc_shapes(in_vector, shapes=["area", "perimeter", "ipq", "hull", "compactness"]):
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

        if "ipq" or "compact" in shapes:
            vector_ipq = calc_ipq(vector_area, vector_perimeter)

        if "hull" in shapes or "compact" in shapes:
            vector_hull = vector_geom.ConvexHull()
            hull_area = vector_hull.GetArea()
            hull_peri = vector_hull.Boundary().Length()
            hull_ratio = float(vector_area) / float(hull_area)
            compactness = float(hull_ratio) * float(vector_ipq)

        if "area" in shapes:
            vector_feature.SetField('area', vector_area)
        if "perimeter" in shapes:
            vector_feature.SetField('perimeter', vector_perimeter)
        if "ipq" in shapes:
            vector_feature.SetField('ipq', vector_ipq)
        if "hull" in shapes:
            vector_feature.SetField('hull_area', hull_area)
            vector_feature.SetField('hull_peri', hull_peri)
            vector_feature.SetField('hull_ratio', hull_ratio)
        if "compact" in shapes:
            vector_feature.SetField('compact', compactness)

        vector_layer.SetFeature(vector_feature)
        
        progress(i, vector_feature_count, name='shape')
        
    vector_layer.CommitTransaction()


if __name__ == "__main__":
    yellow_follow = 'C:/Users/caspe/Desktop/yellow/'
    import sys; sys.path.append(yellow_follow)
    np.set_printoptions(suppress=True)
    from buteo.raster.io import raster_to_array, array_to_raster
    from glob import glob
    
    folder = "C:/Users/caspe/Desktop/vector_test/"
    vector_utm = glob(folder + "*utm*.gpkg")[0]
    vector_wgs = glob(folder + "*wgs*.gpkg")[0]
    
    # vector_utm = "bobthegreat"

    reproject_vector(vector_wgs, vector_utm, folder + "wgs_to_utm.gpkg")
