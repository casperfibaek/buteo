from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
from collections import ChainMap
from glob import glob
import numpy as np
import multiprocessing
import tqdm
import sys
import time

sys.path.append('../lib')
from zonal_statistics_worker import _get_stats


def zonal_stats(vector_path, raster_path, prefix='_', statistics=['mean', 'std', 'skew', 'kurt', 'med', 'iqr'],
                items_per_chunk=32, maxtasksperchild=None, quiet=False, cutline_all_touch=False, threads='cpu'):
    vector_datasource = ogr.Open(vector_path, 1)
    vector_layer = vector_datasource.GetLayer(0)
    total_features = vector_layer.GetFeatureCount()

    vector_field_names = [field.name for field in vector_layer.schema]

    # Create temp csv-file
    raster_folder = raster_path.rsplit('\\', 1)[0]
    raster_name = raster_path.rsplit('\\', 1)[1].rsplit('.', 1)[0]
    csv_path = raster_folder + '\\' + raster_name + '_' + str(uuid.uuid4()) + '.csv'
    csv = open(csv_path, 'a')
    csv.write(f'fid,{prefix}{f",{prefix}".join(statistics)}\n')  # Write the header
    csv.close()

    all_features = np.array_split(
        list(range(total_features)),
        int(total_features / items_per_chunk)
    )

    if threads is 'cpu':
        cpus = int(multiprocessing.cpu_count())
    else:
        cpus = int(threads)

    ranges = []
    for num in range(len(all_features)):
        ranges.append({
            'vector_path': vector_path,
            'raster_path': raster_path,
            'csv_path': csv_path,
            'statistics': statistics,
            'fids': all_features[num],
            'cutline_all_touch': cutline_all_touch,
        })

    # print(ranges)

    if maxtasksperchild is None:
        pool = multiprocessing.Pool(cpus, maxtasksperchild=int(len(all_features) / 32))
    else:
        pool = multiprocessing.Pool(cpus, maxtasksperchild=maxtasksperchild)

    # if quiet:
    composite_parts = pool.map(_get_stats, ranges, chunksize=1)
    # else:
    #     composite_parts = tqdm.tqdm(pool.imap_unordered(_get_stats, ranges, chunksize=1), total=len(all_features))

    # all_stats = dict(ChainMap(*composite_parts))
    # print('Gathered all stats.')

    pool.close()
    print('Closed pool.')

    pool.join()
    print('Joined pool.')

    # exit()

    # # Create field if they do not exist.
    # for statistic in statistics:
    #     if f"{prefix}{statistic}" in vector_field_names:
    #         continue

    #     statField = ogr.FieldDefn(f"{prefix}{statistic}", ogr.OFTReal)
    #     vector_layer.CreateField(statField)
    # print('Added new fields.')

    # # Updates feature fields.
    # for feature in vector_layer:
    #     for statistic in statistics:
    #         fid = feature.GetFID()
    #         feature.SetField(f'{prefix}{statistic}', float(all_stats[feature.GetFID()][statistic]))
    #     vector_layer.SetFeature(feature)
    # print('Added attributes to new fields')

    # vector_layer.SyncToDisk()
    # print('Synced to disc.')

    # del vector_datasource, vector_layer, total_features, cpus, ranges, pool, composite_parts, all_stats
    return None


if __name__ == '__main__':
    base = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\'
    vect = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\segmentation\\S2_MSI_ALL_Rural.shp'
    sar_vv_coh = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\SAR\\VV_coh.tif'
    sar_vv_db = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\SAR\\VV_backscatter.tif'
    sar_vh_coh = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\SAR\\VH_coh.tif'
    sar_vh_db = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\SAR\\VH_backscatter.tif'
    # bands = glob(f"{base}*_B*.tif")

    before = time.time()
    # for band in bands:
    #     name = band.rsplit('_', 1)[1].split('.')[0]
    #     zonal_stats(vect, band, prefix=f'{name}_')

    zonal_stats(vect, sar_vv_coh, prefix='vvc_')
    # zonal_stats(vect, sar_vv_db, prefix='vvb_')
    # zonal_stats(vect, sar_vh_coh, prefix='vhc_')
    # zonal_stats(vect, sar_vh_db, prefix='vhb_')

    after = time.time()
    dif = after - before
    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = "{0:.2f}".format(dif % 60)
    print(f"Zonal_stats took: {hours}h {minutes}m {seconds}s")
