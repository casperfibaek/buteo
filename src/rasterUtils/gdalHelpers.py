def getExtent(dataframe):
    cols = dataframe.RasterXSize
    rows = dataframe.RasterYSize
    transform = dataframe.GetGeoTransform()

    bottomRightX = transform[0] + (cols * transform[1])
    bottomRightY = transform[3] + (rows * transform[5])

    #      (minX,         minY,         maxX,         maxY)
    return (transform[0], bottomRightY, bottomRightX, transform[3])


def getIntersection(extent1, extent2):
    # # Too east
    # if extent2[0] > extent1[2]:
    #     return False
    # # Too north
    # elif extent2[1] > extent1[3]:
    #     return False
    # # Too west
    # elif extent2[2] < extent1[0]:
    #     return False
    # # Too south
    # elif extent2[3] < extent1[1]:
    #     return False
    # else:
    return (
        max(extent1[0], extent2[0]),    # minX
        max(extent1[1], extent2[1]),    # minY
        min(extent1[2], extent2[2]),    # maxX
        min(extent1[3], extent2[3]),    # maxY
    )


def createClipGeoTransform(geoTransform, extent):
    pixelWidth = geoTransform[1]
    pixelHeight = geoTransform[5]
    newCols = round((extent[2] - extent[0]) / pixelWidth)
    newRows = round((extent[3] - extent[1]) / pixelHeight)

    return {
        'Transform': [extent[0], pixelWidth, 0, extent[3], 0, pixelHeight],
        'RasterXSize': abs(newCols),
        'RasterYSize': abs(newRows),
    }


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


def translateMaxValues(datatype):
    datatypes = {
        1: 255,
        2: 65535,
        3: 32767,
        4: 2147483647,
        5: 4294967295,
        6: 3.39999999999999996e+38,
        7: 1.7976931348623157e+308,
    }

    return datatypes[datatype]


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


def datatypeIsInteger(datatype):
    integers = [1, 2, 3, 4, 5, 8, 9]
    if datatype in integers:
        return True
    else:
        return False
