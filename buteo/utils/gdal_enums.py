"""
### GDAL Enum Functions ###

Functions to translate between GDAL and Numpy datatypes.
"""

# External
from osgeo import gdal


valid_raster_driver_extensions = ["tif", "tiff", "img", "vrt", "jp2", "ecw"]
valid_vector_driver_extensions = ["shp", "geojson", "json", "gpkg", "kml", "kmz", "fgb", "gml", "gpx"]
valid_driver_extensions = valid_raster_driver_extensions + valid_vector_driver_extensions


def convert_extension_to_driver(ext):
    assert ext in valid_driver_extensions, f"Invalid extension: {ext}"

    if ext == "tif" or ext == "tiff":
        return "GTiff"
    elif ext == "img":
        return "HFA"
    elif ext == "vrt":
        return "VRT"
    elif ext == "jp2":
        return "JP2ECW"
    elif ext == "ecw":
        return "ECW"

    raise Exception("Unknown extension: " + ext)


def translate_resample_method(method):
    methods = {
        "nearest": 0,
        "bilinear": 1,
        "cubic": 2,
        "cubicSpline": 3,
        "lanczos": 4,
        "average": 5,
        "mode": 6,
        "max": 8,
        "min": 9,
        "median": 10,
        "q1": 11,
        "q3": 12,
        "sum": 13,
        "rms": 14,
    }

    if method == "sum":
        return "sum"
    elif method == "max" or method == "maximum":
        return "max"
    elif method == "min" or method == "minimum":
        return "min"

    if method in methods:
        return methods[method]
    else:
        return 0


def numpy_fill_values(datatype):
    datatypes = {
        "int8": -127,
        "int16": -32767,
        "int32": -2147483647,
        "int64": -9223372036854775807,
        "uint8": 255,
        "uint16": 65535,
        "uint32": 4294967295,
        "uint64": 18446744073709551615,
        "float16": -9999.0,
        "float32": -9999.0,
        "float64": -9999.0,
    }

    test_type = datatype
    if isinstance(test_type, np.dtype):
        test_type = test_type.name

    if test_type in datatypes:
        return datatypes[test_type]
    else:
        return -9999.0


def gdal_nodata_value_from_type(gdal_datatype_raw):
    if gdal_datatype_raw == 0:
        return 0
    elif gdal_datatype_raw == 1:
        return 255
    elif gdal_datatype_raw == 2:
        return 65535
    elif gdal_datatype_raw == 3:
        return -32767
    elif gdal_datatype_raw == 4:
        return 4294967295
    elif gdal_datatype_raw == 5:
        return -2147483647
    elif gdal_datatype_raw == 6:
        return -9999.0
    elif gdal_datatype_raw == 7:
        return -9999.0
    else:
        return 0


def translate_datatypes(datatype):
    datatypes = {
        "byte": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "cint16": 8,
        "cint32": 9,
        "cfloat32": 10,
        "cfloat64": 11,
    }

    if datatype in datatypes.keys():
        return datatypes[datatype]
    else:
        return 6


def gdal_to_numpy_datatype(gdal_int):
    datatypes = {
        "1": "uint8",
        "2": "uint16",
        "3": "int16",
        "4": "uint32",
        "5": "int32",
        "6": "float32",
        "7": "float64",
        "8": "cint16",
        "9": "cint32",
        "10": "cfloat32",
        "11": "cfloat64",
    }
    return datatypes[str(gdal_int)]


def numpy_to_gdal_datatype(datatype):
    datatypes = {
        "int8": gdal.GDT_Int16,
        "int16": gdal.GDT_Int16,
        "int32": gdal.GDT_Int32,
        "int64": gdal.GDT_Int32,
        "uint8": gdal.GDT_Byte,
        "uint16": gdal.GDT_UInt16,
        "uint32": gdal.GDT_UInt32,
        "uint64": gdal.GDT_UInt32,
        "float16": gdal.GDT_Float32,
        "float32": gdal.GDT_Float32,
        "float64": gdal.GDT_Float64,
    }

    # print("WARNING: numpy to gdal datatype is deprecated.")

    if datatype.name in datatypes.keys():
        return datatypes[datatype.name]
    else:
        return gdal.GDT_Float32


def numpy_to_gdal_datatype2(datatype):
    datatypes = {
        "int8": gdal.GDT_Int16,
        "int16": gdal.GDT_Int16,
        "int32": gdal.GDT_Int32,
        "int64": gdal.GDT_Int32,
        "uint8": gdal.GDT_Byte,
        "uint16": gdal.GDT_UInt16,
        "uint32": gdal.GDT_UInt32,
        "uint64": gdal.GDT_UInt32,
        "float16": gdal.GDT_Float32,
        "float32": gdal.GDT_Float32,
        "float64": gdal.GDT_Float64,
    }

    if datatype in datatypes.keys():
        return datatypes[datatype]
    else:
        print("warning: Unknown datatype. Defaulting to float32.")
        return gdal.GDT_Float32


def gdal_datatype_is_float(datatype):
    floats = [6, 7, 10, 11]
    if datatype in floats:
        return True
    else:
        return False
