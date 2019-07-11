import ogr
import csv
import sys


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
