from osgeo import gdal


def getExtent(dataframe):
    transform = dataframe.GetGeoTransform()

    bottomRightX = transform[0] + (dataframe.RasterXSize * transform[1])
    bottomRightY = transform[3] + (dataframe.RasterYSize * transform[5])

    #      (   minX,         minY,         maxX,         maxY     )
    return (transform[0], bottomRightY, bottomRightX, transform[3])


def getIntersection(extent1, extent2):
    one_bottomLeftX = extent1[0]
    one_bottomLeftY = extent1[1]
    one_topRightX = extent1[2]
    one_topRightY = extent1[3]

    two_bottomLeftX = extent2[0]
    two_bottomLeftY = extent2[1]
    two_topRightX = extent2[2]
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
            max(one_bottomLeftY, two_bottomLeftY),    # minY of intersection
            min(one_topRightX, two_topRightX),        # maxX of intersection
            min(one_topRightY, two_topRightY),        # maxY of intersection
        )


def createClipGeoTransform(geoTransform, extent):
    RasterXSize = round((extent[2] - extent[0]) / geoTransform[1])  # (maxX - minX) / pixelWidth
    RasterYSize = round((extent[3] - extent[1]) / geoTransform[5])  # (maxY - minY) / pixelHeight

    return {
        'Transform': [extent[0], geoTransform[1], 0, extent[3], 0, geoTransform[5]],
        'RasterXSize': abs(RasterXSize),
        'RasterYSize': abs(RasterYSize),
    }


def createSubsetDataframe(dataframe, band=1, noDataValue=None):
        # Create a GDAL driver to create dataframes in the right outputFormat
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


def translateResampleMethod(method):
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


def numpyFillValues(dtype):
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

    if dtype in datatypes:
        return datatypes[dtype]
    else:
        return 0


def translateMaxValues(datatype):
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

    if datatype in datatypes:
        return datatypes[datatype]
    else:
        return 0


def translateDataTypes(datatype):
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

    if datatype in datatypes:
        return datatypes[datatype]
    else:
        return 6


def datatypeIsFloat(datatype):
    floats = [6, 7, 10, 11]
    if datatype in floats:
        return True
    else:
        return False
