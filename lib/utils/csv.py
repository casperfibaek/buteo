import csv
import sys
import pandas as pd
import numpy as np
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def to_type(string):
    if isinstance(string, str):
        if string.isdigit():
            return int(string)
        if isfloat(string):
            return float(string)
    return string


def get_type(string):
    if isinstance(string, int):
        return 'int'
    elif isinstance(string, float):
        return 'float'
    elif isinstance(string, bool):
        return 'int'
    elif string is None or string is 'None':
        return 'int'
    elif isinstance(string, str):
        if string.isdigit():
            return 'int'
        if isfloat(string):
            return 'float'

    return 'str'


def shp_to_csv(in_shp, out_csv, with_geom=False):
    csvfile = open(out_csv, 'wb')
    ds = ogr.Open(in_shp)
    lyr = ds.GetLayer()
    dfn = lyr.GetLayerDefn()
    nfields = dfn.GetFieldCount()

    fields = []

    for i in range(nfields):
        fields.append(dfn.GetFieldDefn(i).GetName())

    if with_geom:
        fields.append('kmlgeometry')

    csvwriter = csv.DictWriter(csvfile, fields)

    try:
        csvwriter.writeheader()
    except:
        csvfile.write(','.join(fields) + '\n')

    if with_geom:
        for feat in lyr:
            attributes = feat.items()
            geom = feat.GetGeometryRef()
            attributes['kmlgeometry'] = geom.ExportToKML()
            csvwriter.writerow(attributes)
    else:
        for feat in lyr:
            attributes = feat.items()
            csvwriter.writerow(attributes)

    del csvwriter, lyr, ds
    csvfile.close()


def get_csv_types(csv, max_rows=None):
    csv = open(csv, 'r').readlines()
    csv_header = csv[0].strip().split(',')

    if max_rows is None:
        max_rows = len(csv)

    holder = {}
    for row in csv_header:
        holder[row] = 'int'

    for i, line in enumerate(csv[1:]):
        if i > max_rows:
            return holder

        prepared = line.strip().split(',')
        for i, val in enumerate(prepared):
            value = to_type(val)
            if holder[csv_header[i]] is 'str':
                continue
            elif holder[csv_header[i]] is 'float':
                if get_type(value) is 'str':
                    holder[csv_header[i]] = 'str'
                    continue
            elif holder[csv_header[i]] is 'int':
                _type = get_type(value)
                if _type is not 'int':
                    holder[csv_header[i]] = _type

    return holder


def csv_to_keydict(in_csv, csv_key):
    csv = open(in_csv, 'r').readlines()
    csv_header = csv[0].strip().split(',')

    if csv_key not in csv_header:
        raise RuntimeError('Key not in csv file.') from None

    keydict = {}
    key_index = csv_header.index(csv_key)

    for line in csv[1:]:
        prepared = line.strip().split(',')
        key = to_type(prepared[key_index])
        keydict[key] = {}

        for i, val in enumerate(prepared):
            if i is key_index:
                continue
            keydict[key][csv_header[i]] = to_type(val)

    return keydict


def join_csv_to_shp(in_shp, shp_key, in_csv, csv_key, out_shp, skip_nulls=True):
    vector_datasource = ogr.Open(in_shp, 0)
    vector_layer = vector_datasource.GetLayer(0)
    vector_driver = ogr.GetDriverByName('ESRI Shapefile')
    vector_projection = osr.SpatialReference()
    vector_projection.ImportFromWkt(str(vector_layer.GetSpatialRef()))
    vector_layer_defn = vector_layer.GetLayerDefn()

    vector_features_count = vector_layer.GetFeatureCount()
    vector_field_names = [field.name for field in vector_layer.schema]

    csv = open(in_csv, 'r').readlines()
    csv_header = csv[0].strip().split(',')

    csv_keydict = csv_to_keydict(in_csv, csv_key)
    csv_types = get_csv_types(in_csv, max_rows=5)

    if shp_key not in vector_field_names:
        raise RuntimeError('Key not in shapefile.') from None

    out_vector_datasource = vector_driver.CreateDataSource(out_shp)
    out_vector_layer = out_vector_datasource.CreateLayer('join_shp', vector_projection, ogr.wkbPolygon)

    for i in range(vector_layer_defn.GetFieldCount()):
        out_vector_layer.CreateField(vector_layer_defn.GetFieldDefn(i))

    for num in range(len(csv_header)):
        if csv_header[num] is csv_key:
            continue
        if csv_header[num] in vector_field_names:
            continue
        
        field_name = csv_header[num]

        if csv_types[field_name] is 'int':
            out_vector_layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger))
        elif csv_types[field_name] is 'float':
            out_vector_layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTReal))
        else:
            out_vector_layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTString))

    for fid in range(vector_features_count):
        vector_feature = vector_layer.GetFeature(fid)
        feature_key_value = vector_feature.GetField(shp_key)

        # Insert copy functionality
        if feature_key_value in csv_keydict:
            csv_values = csv_keydict[feature_key_value]
            vector_feature_geom = vector_feature.GetGeometryRef()

            outFeature = ogr.Feature(out_vector_layer.GetLayerDefn())

            skip = False
            for key, val in csv_values.items():
                if val is 'None' or val is 'NaN' or val is '' or val is 'nan':
                    if skip_nulls:
                        skip = True
                        break
                    outFeature.SetField(key, 'NULL')
                else:
                    outFeature.SetField(key, val)

            if not skip:
                outFeature.SetGeometry(vector_feature_geom)

                for i in range(vector_feature.GetFieldCount()):
                    outFeature.SetField(vector_feature.GetFieldDefnRef(i).GetName(), vector_feature.GetField(i))

                out_vector_layer.CreateFeature(outFeature)

    return

def find_feature_outliers(csv, out_csv, class_column, exclude=['fid', 'id', 'DN']):
    # Read csv
    df = pd.read_csv(csv)

    # Create z-score column
    df = df.assign(z_scr=pd.Series(np.zeros(len(df[class_column]))).values)

    # Get unique attributes in class column
    unique = df[class_column].unique().tolist()

    # Create exclude lust
    exclude.append(class_column)
    exclude.append('z_scr')
    exclude = list(set(exclude).intersection(df.columns.tolist()))

    z_scores = []

    # Calculate class values
    for csv_class in unique:
        sub_dataset = df.loc[df[class_column] == csv_class].drop(exclude, axis=1)

        # Calculate the medians
        medians = sub_dataset.median()

        # Calculate the median absolute deviation
        deviations = sub_dataset.subtract(medians)
        abs_deviations = deviations.abs()
        mad = abs_deviations.median()
        mad_std = mad.apply(lambda x: x * 1.4826)

        z_scores += sub_dataset.divide(mad_std).abs().mean(axis=1).tolist()

    df['z_scr'] = z_scores
    df.to_csv(out_csv)

    return out_csv
