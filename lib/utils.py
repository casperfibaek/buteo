import os
import sys
import math
import time
import numpy as np
from osgeo import gdal
from scipy.stats import norm


def progress(count, total, name='Processing'):
    bar_len = os.get_terminal_size().columns - 24
    filled_len = int(round(bar_len * count / float(total)))
    display_name = name[:10] + '..: '

    percents = round(100.0 * count / float(total), 1)
    if percents == 100:
        sys.stdout.flush()
        return None

    bar = '█' * filled_len + '.' * (bar_len - filled_len)

    sys.stdout.write(f"{display_name}[{bar}] {percents} %\r")
    sys.stdout.flush()

    return None


def timing(before):
    after = time.time()
    dif = after - before
    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = "{0:.2f}".format(dif % 60)
    print(f"Processing took: {hours}h {minutes}m {seconds}s")


def create_progress_callback(total, name):
    def _progress_callback(complete, message, unknown):
        return progress(complete, total, name)
    return _progress_callback


def progress_callback_quiet(complete, message, unknown):
    return None


def get_extent(dataframe):
    transform = dataframe.GetGeoTransform()

    bottomRightX = transform[0] + (dataframe.RasterXSize * transform[1])
    bottomRightY = transform[3] + (dataframe.RasterYSize * transform[5])

    #      (   minX,         minY,         maxX,         maxY     )
    return (transform[0], bottomRightY, bottomRightX, transform[3])


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
        return False
    elif two_bottomLeftY > one_topRightY:   # Too far north
        return False
    elif two_topRightX < one_bottomLeftX:   # Too far west
        return False
    elif two_topRightY < one_bottomLeftY:   # Too far south
        return False
    else:
        return (
            max(one_bottomLeftX, two_bottomLeftX),    # minX of intersection
            min(one_topRightX, two_topRightX),        # maxX of intersection
            max(one_bottomLeftY, two_bottomLeftY),    # minY of intersection
            min(one_topRightY, two_topRightY),        # maxY of intersection
        )


def create_geotransform(geo_transform, extent):
    RasterXSize = round((extent[2] - extent[0]) / geo_transform[1])  # (maxX - minX) / pixelWidth
    RasterYSize = round((extent[3] - extent[1]) / geo_transform[5])  # (maxY - minY) / pixelHeight

    return {
        'Transform': [extent[0], geo_transform[1], 0, extent[3], 0, geo_transform[5]],
        'RasterXSize': abs(RasterXSize),
        'RasterYSize': abs(RasterYSize),
    }


def get_size(start_path='.', rough=True):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    if rough is True:
        return total_size >> 20
    else:
        return total_size


def divide_steps(total, step):
    steps = []
    remainder = total % step
    divided = int(total / step)
    for cpu in range(step):
        if remainder > 0:
            steps.append(divided + 1)
            remainder -= 1
        else:
            steps.append(divided)

    return steps


def divide_into_steps(arr, steps_length):
    steps = divide_steps(len(arr), steps_length)

    ret_arr = []
    last = 0
    count = 0
    for x in steps:
        count += 1
        if count > len(arr):
            continue
        ret_arr.append(arr[last:x + last])
        last += x

    return ret_arr


def step_ranges(steps):
    start_stop = []
    last = 0
    for num in range(0, len(steps)):
        step_size = steps[num]
        id = num + 1

        start_stop.append({
            "id": id,
            "start": last,
            "stop": last + step_size,
        })

        last += step_size

    return start_stop


def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)


def create_subset_dataframe(dataframe, band=1):
        # Create a GDAL driver to create dataframes in the right output_format
        driver = gdal.GetDriverByName('MEM')

        inputBand = dataframe.GetRasterBand(band)
        inputTransform = dataframe.GetGeoTransform()
        inputProjection = dataframe.GetProjection()
        inputDataType = inputBand.DataType

        subsetDataframe = driver.Create(
            'ignored',                 # Unused as destination is memory.
            dataframe.RasterXSize,     # Dataframe width in pixels (e.g. 1920px).
            dataframe.RasterYSize,     # Dataframe height in pixels (e.g. 1280px).
            1,                         # The number of bands required.
            inputDataType,             # Datatype of the destination
        )
        subsetDataframe.SetGeoTransform(inputTransform)
        subsetDataframe.SetProjection(inputProjection)

        # Write the requested inputBand to the subset
        subsetDataframe.WriteArray(inputBand.ReadAsArray())

        # Free memory
        inputBand = None

        return subsetDataframe


def copy_dataframe(dataframe, name='ignored', output_format='MEM'):
    driver = gdal.GetDriverByName(output_format)

    inputTransform = dataframe.GetGeoTransform()
    inputProjection = dataframe.GetProjection()
    inputBand = dataframe.GetRasterBand(1)
    inputDataType = inputBand.DataType
    inputBandCount = dataframe.RasterCount

    if output_format == 'MEM':
        options = []
    else:
        if datatype_is_float(inputDataType) is True:
            predictor = 3
        else:
            predictor = 2
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    destination = driver.Create(name, dataframe.RasterXSize, dataframe.RasterYSize, inputBandCount, inputDataType, options)
    destination.SetProjection(inputProjection)
    destination.SetGeoTransform(inputTransform)

    for i in range(inputBandCount):
        _inputBand = dataframe.GetRasterBand(i + 1)
        _inputBandNoDataValue = _inputBand.GetNoDataValue()
        _inputBandData = _inputBand.ReadAsArray()
        _destinationBand = destination.GetRasterBand(i + 1)
        _destinationBand.WriteArray(_inputBandData)

        if _inputBandNoDataValue is not None:
            destination.SetNoDataValue(_inputBandNoDataValue)

    return destination


def translate_resample_method(method):
    methods = {
        'nearest': 0,
        'bilinear': 1,
        'cubic': 2,
        'cubicSpline': 3,
        'lanczos': 4,
        'average': 5,
        'mode': 6,
        'max': 8,
        'min': 9,
        'median': 10,
        'q1': 11,
        'q3': 12,
    }

    if method in methods:
        return methods[method]
    else:
        return 0


def numpy_fill_values(datatype):
    datatypes = {
        'int8': 127,
        'int16': 32767,
        'int32': 2147483647,
        'int64': 9223372036854775807,
        'uint8': 255,
        'uint16': 65535,
        'uint32': 4294967295,
        'uint64': 18446744073709551615,
        'float16': -9999,
        'float32': -9999,
        'float64': -9999,
    }

    if datatype in datatypes:
        return datatypes[datatype]
    else:
        return 0


def translate_max_values(datatype):
    datatypes = {
        1: 255,             # GDT_Byte
        2: 65535,           # GDT_Uint16
        3: 32767,           # GDT_Int16
        4: 2147483647,      # GDT_Uint32
        5: 4294967295,      # GDT_Int32
        6: -9999,           # GDT_Float32
        7: -9999,           # GDT_Float64
        8: 32767,           # GDT_CInt16
        9: 4294967295,      # GDT_CInt32
        10: -9999,          # GDT_CFloat32
        11: -9999,          # GDT_CFloat64
    }

    if datatype in datatypes.keys():
        return datatypes[datatype]
    else:
        return 0


def translate_datatypes(datatype):
    datatypes = {
        'byte': 1,
        'uint16': 2,
        'int16': 3,
        'uint32': 4,
        'int32': 5,
        'float32': 6,
        'float64': 7,
        'cint16': 8,
        'cint32': 9,
        'cfloat32': 10,
        'cfloat64': 11,
    }

    if datatype in datatypes.keys():
        return datatypes[datatype]
    else:
        return 6


def numpy_to_gdal_datatype(datatype):
    datatypes = {
        'int8': gdal.GDT_Int16,
        'int16': gdal.GDT_Int16,
        'int32': gdal.GDT_Int32,
        'int64': gdal.GDT_Int32,
        'uint8': gdal.GDT_Byte,
        'uint16': gdal.GDT_UInt16,
        'uint32': gdal.GDT_UInt32,
        'uint64': gdal.GDT_UInt32,
        'float16': gdal.GDT_Float32,
        'float32': gdal.GDT_Float32,
        'float64': gdal.GDT_Float64,
    }

    if datatype.name in datatypes.keys():
        return datatypes[datatype.name]
    else:
        return gdal.GDT_Float32


def gdal_to_numpy_datatype(datatype):
    datatypes = {
        3: 'int8',
        3: 'int16',
        5: 'int32',
        5: 'int64',
        1: 'uint8',
        2: 'uint16',
        4: 'uint32',
        4: 'uint64',
        6: 'float16',
        6: 'float32',
        7: 'float64',
    }

    if datatype in datatypes.keys():
        return datatypes[datatype]
    else:
        return 'float64'


def datatype_is_float(datatype):
    floats = [6, 7, 10, 11]
    if datatype in floats:
        return True
    else:
        return False


# Turns zscores
def __scale_zscores(zscore, sqrt=False, root=math.pi):
    cdf = 1 - abs((norm.cdf(zscore) - 0.5) / 0.5)
    if sqrt is True:
        return math.pow(cdf, 1 / root)
    else:
        return cdf


_scale_zscores = np.vectorize(__scale_zscores)


def scale_zscores(arr_of_zscores, sqrt=False, root=math.pi):
    return _scale_zscores(arr_of_zscores, sqrt=sqrt, root=math.pi)
