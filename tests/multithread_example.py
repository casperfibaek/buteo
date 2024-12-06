# pylint: skip-file
# type: ignore

from osgeo import ogr, gdal
import multiprocessing
from osgeo import ogr, osr
from math import ceil
from multiprocessing import Pool
import os
import time
import statistics



def create_vector(input_path, feature_count, driver_name="GPKG"):
    """
    Create an in-memory vector dataset with a specified number of features.

    Args:
        input_path (str): The path to create the vector dataset.
        feature_count (int): The number of features to create.
        driver_name (str): The name of the OGR driver to use. Default is "GPKG".

    Returns:
        ogr.DataSource: The created vector dataset.
    """
    driver = ogr.GetDriverByName(driver_name)

    # For demonstration, let's create a simple in-memory shapefile
    datasource = driver.CreateDataSource(input_path)

    # Set Spatial Reference System to Pseudo Mercator
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)  # EPSG:3857 is the code for Pseudo Mercator

    layer = datasource.CreateLayer('layer', srs, ogr.wkbPolygon)

    # Add a field (optional)
    field_defn = ogr.FieldDefn('id', ogr.OFTInteger)
    layer.CreateField(field_defn)

    # Create some sample features (replace this with actual data loading)
    for i in range(feature_count):
        feature_defn = layer.GetLayerDefn()
        feature = ogr.Feature(feature_defn)
        feature.SetField('id', i)
        # Create a simple square polygon using WKT
        wkt = f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))'
        polygon = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(polygon)
        layer.CreateFeature(feature)
        feature = None  # Clean up

    layer.SyncToDisk()

    return datasource


def process_feature_batch(args):
    features, buffer_size = args
    results = []
    for fid, wkb_geom in features:
        geom = ogr.CreateGeometryFromWkb(wkb_geom)
        processed_geom = geom.Buffer(buffer_size)
        results.append((fid, processed_geom.ExportToWkb()))
    return results


def multithread_fgb_mem(num_features, processes=14):
    input_path = '/vsimem/input_multi.mem'
    datasource = create_vector(input_path, num_features, "Memory")
    layer = datasource.GetLayer()

    # Handle processes=0 case
    if processes == 0:
        processes = multiprocessing.cpu_count()

    # If only 1 process available/requested, process sequentially
    if processes == 1 or multiprocessing.cpu_count() == 1:

        for _ in range(num_features):
            feature = layer.GetNextFeature()
            if feature is None:
                continue
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            processed_geom = geom.Buffer(0.1)
            feature.SetGeometry(processed_geom)
            layer.SetFeature(feature)
            feature = None
    else:
        # Read all features in one pass
        features = []
        for _ in range(num_features):
            feature = layer.GetNextFeature()
            geom = feature.GetGeometryRef()
            if geom:
                features.append((feature.GetFID(), geom.ExportToWkb()))

        # Split features into batches
        batch_size = ceil(len(features) / processes)
        feature_batches = [features[i:i + batch_size] for i in range(0, len(features), batch_size)]

        # Process batches in parallel
        with Pool(processes=processes) as pool:
            results = pool.map(process_feature_batch, [(batch, 0.1) for batch in feature_batches])

        # Flatten results and update features
        for batch_results in results:
            for fid, wkb_geom in batch_results:
                feature = layer.GetFeature(fid)
                if feature:
                    feature.SetGeometry(ogr.CreateGeometryFromWkb(wkb_geom))
                    layer.SetFeature(feature)
                    feature = None

    driver = ogr.GetDriverByName('FlatGeobuf')
    output_path = './output_multithreaded_fgb.fgb'
    if driver.Open(output_path):
        driver.DeleteDataSource(output_path)
    driver.CopyDataSource(datasource, output_path)

    datasource = None


def naive_fgb_mem(num_features):
    input_path = "input_naive_fgb.mem"
    datasource = create_vector(input_path, num_features, "Memory")

    layer = datasource.GetLayer()

    for _ in range(num_features):
        feature = layer.GetNextFeature()
        if feature is None:
            continue
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        processed_geom = geom.Buffer(0.1)
        feature.SetGeometry(processed_geom)
        layer.SetFeature(feature)
        feature = None  # Clean up

    driver = ogr.GetDriverByName('FlatGeobuf')
    output_path = './output_sequential_fbg.fgb'
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    driver.CopyDataSource(datasource, output_path)

    datasource = None


def benchmark_naive_functions(functions, num_features, runs=5):
    for func in functions:
        execution_times = []
        for _ in range(runs):
            start_time = time.time()
            func(num_features)
            execution_times.append(time.time() - start_time)
        average_time = sum(execution_times) / runs
        std_dev = statistics.stdev(execution_times)
        print(f"{func.__name__} - Average Time: {average_time:.2f}s, Standard Deviation: {std_dev:.2f}s")


if __name__ == '__main__':
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    ogr.UseExceptions()

    num_features = 100000
    # processes = 14

    print("")
    print("Benchmarking 100 features")
    benchmark_naive_functions([naive_fgb_mem, multithread_fgb_mem], 100, 10)
    print("")
    print("Benchmarking 1000 features")
    benchmark_naive_functions([naive_fgb_mem, multithread_fgb_mem], 1000, 10)
    print("")
    print("Benchmarking 10000 features")
    benchmark_naive_functions([naive_fgb_mem, multithread_fgb_mem], 10000, 10)
    print("")
    print("Benchmarking 100000 features")
    benchmark_naive_functions([multithread_fgb_mem, naive_fgb_mem], 100000, 5)
