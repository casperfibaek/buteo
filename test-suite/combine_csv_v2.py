import fiona
import sys
import pandas as pd
import numpy as np

INVALID_CHRS = " !@#$%^&*()`~<>,'\"\\=/+?\n\t\r"


def print_usage():
    """"""
    print("Performs a LEFT join of a csv to a shapefile on a key")
    print("Usage:\tfiona_join.py <in.shp> <shpkey> <in.csv> <csvkey> <out.shp>")


def validcolumnnames(cols):
    """"""
    cols = [c.strip(INVALID_CHRS) for c in cols]
    return ["_%s" % c if c[0].isdigit() else c for c in cols] 


def getfionadtype(dt):
    """"""
    if np.issubdtype(dt, 'i'):
        return 'int'
    if np.issubdtype(dt, 'f'):
        return 'float'
    return 'str'


def joincsvtoshp(shp, field1, in_csv, field2, out_shp):
    """

    """
    with fiona.open(shp) as source:
        df = pd.read_csv(in_csv)
        df.set_index(field2, inplace=True, verify_integrity=True)

        schema = source.schema
        shpkeys = schema['properties'].keys()
        if field1 not in shpkeys:
            print('Warning: %s not found, attempting to join by ID' % field1)
        csvkeys = validcolumnnames(df.columns)
        csvkeys = ["_%s" % c if c in shpkeys else c for c in csvkeys]
        ncols = len(df.columns)

        for i in range(ncols):
            if csvkeys[i] != df.columns[i]:
                print('Warning: invalid column name: %s' % df.columns[i])
            assert csvkeys[i] != '', 'Invalid column name: %s' % df.columns[i]
            assert csvkeys[i] not in schema['properties'], 'Duplicate column name: %s' % df.columns[i]
            schema['properties'][csvkeys[i]] = getfionadtype(df.dtypes[df.columns[i]])

        join_count = 0
        features = len(source)

        with fiona.open(out_shp, 'w',
                        driver=source.driver,
                        crs=source.crs,
                        schema=schema) as outfile:
            for item in source:
                p = item['properties']
                if field1 in p:
                    key = p[field1]
                else:
                    key = int(item['id'])
                if key in df.index:
                    join_count += 1
                    for i in range(ncols):
                        p[csvkeys[i]] = df[df.columns[i]][key]
                else:
                    for i in range(ncols):
                        p[csvkeys[i]] = None
                outfile.write(
                    {'id': -1,
                     'properties': p,
                     'geometry': item['geometry']})

        print("joined %s of %s" % (join_count, features))

if __name__ == "__main__":
    csv = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\csv\\pca_rural.csv'
    in_shp = base = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\segmentation\\pca_rural.shp'
    out_shp = base = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\segmentation\\pca_rural_meta.shp'

    joincsvtoshp(in_shp, 'DN', csv, 'DN', out_shp)
