from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *
from collections import ChainMap
from glob import glob
import uuid
import numpy as np
import multiprocessing
import sys
import time

sys.path.append('../lib')
from zonal_statistics_worker import _get_stats


def zonal_stats(vector_path, raster_path, prefix='_', statistics=['mean', 'med', 'mad', 'std', 'skew', 'kurt', 'iqr'],
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

    if maxtasksperchild is None:
        pool = multiprocessing.Pool(cpus, maxtasksperchild=int(len(all_features) / 32))
    else:
        pool = multiprocessing.Pool(cpus, maxtasksperchild=maxtasksperchild)

    composite_parts = pool.map(_get_stats, ranges, chunksize=1)

    pool.close()

    pool.join()

    return None


if __name__ == '__main__':
    vect = 'E:\\SATF\\phase_II_urban-seperation\\segmentation_zonal.gpkg'
    rasters = glob('E:\\SATF\\data\\*tif')

    before = time.time()

    for index, raster in enumerate(rasters):
        zonal_stats(vect, raster, prefix=f'{index}_')

    after = time.time()
    dif = after - before
    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = "{0:.2f}".format(dif % 60)
    print(f"Zonal_stats took: {hours}h {minutes}m {seconds}s")
