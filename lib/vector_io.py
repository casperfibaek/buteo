import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_metadata
from osgeo import gdal, ogr, osr
import pandas as pd
import os, json


def vector_to_reference(vector):
    try:
        if isinstance(vector, ogr.DataSource):  # Dataset already OGR dataframe.
            return vector
        else:
            opened = ogr.Open(vector)
            
            if opened is None:
                raise Exception("Could not read input raster")

            return opened
    except:
        raise Exception("Could not read input raster")


def vector_to_memory(vector):
    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref)

    driver = ogr.GetDriverByName("Memory")

    copy = driver.CreateDataSource("mem_vector")

    for layer in range(metadata["layer_count"]):
        copy.CopyLayer(ref.GetLayer(layer), f"mem_vector_{layer}", ["OVERWRITE=YES"])

    return copy


def vector_to_metadata(vector, process="first"):
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
            "name": layer.GetName(),
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

        layer_dict["field_names"] = []
        layer_dict["field_types"] = []

        for field_index in range(layer_dict["field_count"]):
            field_defn = layer_defn.GetFieldDefn(field_index)
            layer_dict["field_names"].append(field_defn.GetName())
            layer_dict["field_types"].append(field_defn.GetFieldTypeName(field_defn.GetType()))

        if process == "first":

            for key, value in layer_dict.items():
                metadata[key] = value
            
            del metadata["layers"]
            break

        metadata["layers"].append(layer_dict)
   

    return metadata


def vector_get_attribute_table(vector, process="first", geom=False):
    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref, process=process)

    dataframes = []

    for vector_layer in range(metadata["layer_count"]):
        attribute_table_header = None
        feature_count = None

        if process != "first":
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

        if process == "first": return df
        
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


def vector_mask(vector:ogr.DataSource or str, raster:gdal.Dataset or str) -> gdal.Dataset:
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
