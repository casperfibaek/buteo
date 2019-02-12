def getExtent(dataframe):
    cols = dataframe.RasterXSize
    rows = dataframe.RasterYSize
    gt = dataframe.GetGeoTransform()
    ext = []
    xarr = [0, cols]
    yarr = [0, rows]

    for px in xarr:
        for py in yarr:
            x = gt[0] + (px * gt[1]) + (py * gt[2])
            y = gt[3] + (px * gt[4]) + (py * gt[5])
            ext.append([x, y])
        yarr.reverse()
    return ext


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
