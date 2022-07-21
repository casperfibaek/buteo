"""
### GDAL Enum Functions ###

Functions to translate between **GDAL** and **NumPy** datatypes.
"""

# External
import numpy as np
from osgeo import gdal



def get_available_drivers():
    """
    Returns a list of **all** available drivers.

    ## Returns:
    (_list_): List of dicts available drivers. Each dict has the following keys: </br>
    `short_name`: **String** of driver short name (e.g. GTiff). </br>
    `long_name`: **String** of driver long name (e.g. GeoTiff). </br>
    `extension`: **String** of driver extension (e.g. tif). **OBS**: Can be an empty string. </br>
    """
    raster_drivers = []
    vector_drivers = []

    for i in range(0, gdal.GetDriverCount()):
        driver = gdal.GetDriver(i)
        metadata = driver.GetMetadata_Dict()

        extensions = driver.GetMetadataItem(gdal.DMD_EXTENSIONS)
        if extensions is not None:
            extensions = extensions.split(" ")
        else:
            extensions = [None]

        for ext in extensions:
            info = {
                "short_name": driver.ShortName,
                "long_name": driver.LongName,
                "extension": ext,
            }

            if 'DCAP_RASTER' in metadata:
                raster_drivers.append(info)

            if 'DCAP_VECTOR' in metadata:
                vector_drivers.append(info)

    return (raster_drivers, vector_drivers)



def get_valid_raster_driver_extensions():
    """
    Returns a list of valid raster driver extensions.

    ## Returns:
    (_list_): List of valid raster driver extensions.
    """
    available_raster_drivers, _available_vector_drivers = get_available_drivers()

    valid_raster_driver_extensions = []

    for driver in available_raster_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:
            valid_raster_driver_extensions.append(driver["extension"])

    return valid_raster_driver_extensions


def get_valid_vector_driver_extensions():
    """
    Returns a list of valid vector driver extensions.

    ## Returns:
    (_list_): List of valid vector driver extensions.
    """
    _available_raster_drivers, available_vector_drivers = get_available_drivers()

    valid_vector_driver_extensions = []

    for driver in available_vector_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:
            valid_vector_driver_extensions.append(driver["extension"])

    return valid_vector_driver_extensions


def get_valid_driver_extensions():
    """
    Returns a list of all valid driver extensions (**GDAL** + **OGR**).

    ## Returns:
    (_list_): List of all valid driver extensions.
    """
    available_raster_drivers, available_vector_drivers = get_available_drivers()

    valid_driver_extensions = []

    for driver in available_raster_drivers + available_vector_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:
            valid_driver_extensions.append(driver["extension"])

    return valid_driver_extensions


def is_valid_driver_extension(ext):
    """
    Checks if a file extension is a valid **GDAL** or **OGR** driver extension.

    ## Args:
    `ext` (_str_): The file extension. </br>

    ## Returns:
    (_bool_): **True** if valid, **False** otherwise.
    """
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

    return ext in get_valid_driver_extensions()


def is_valid_raster_driver_extension(ext):
    """
    Checks if a raster file extension is a valid **GDAL** driver extension.

    ## Args:
    `ext` (_str_): The file extension. </br>

    ## Returns:
    (_bool_): **True** if valid, **False** otherwise.
    """
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

    return ext in get_valid_raster_driver_extensions()


def is_valid_vector_driver_extension(ext):
    """
    Checks if a vector file extension is a valid **OGR** driver extension.

    ## Args:
    `ext` (_str_): The file extension. </br>

    ## Returns:
    (_bool_): **True** if valid, **False** otherwise.
    """
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

    return ext in get_valid_vector_driver_extensions()


def convert_raster_extension_to_driver_shortname(ext):
    """
    Converts a raster file extension to a **GDAL** driver short_name name.

    ## Args:
    `ext` (_str_): The file extension. </br>

    ## Returns:
    (_str_): The driver short_name (e.g. GTiff).
    """
    assert is_valid_raster_driver_extension(ext), f"Invalid extension: {ext}"

    raster_drivers, _vector_drivers = get_available_drivers()

    for driver in raster_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


def convert_extension_to_driver_shortname(ext):
    """
    Converts a file extension to an **OGR** or **GDAL** driver short_name name.

    ## Args:
    `ext` (_str_): The file extension. </br>

    ## Returns:
    (_str_): The driver short_name (e.g. GPKG or GTiff).
    """
    assert is_valid_vector_driver_extension(ext) or is_valid_raster_driver_extension(ext), f"Invalid extension: {ext}"

    raster_drivers, vector_drivers = get_available_drivers()

    for driver in raster_drivers + vector_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


def convert_vector_extension_to_driver_shortname(ext):
    """
    Converts a vector file extension to an **OGR** driver short_name name.

    ## Args:
    `ext` (_str_): The file extension. </br>

    ## Returns:
    (_str_): The driver short_name (e.g. GPKG).
    """
    assert is_valid_vector_driver_extension(ext), f"Invalid extension: {ext}"

    _raster_drivers, vector_drivers = get_available_drivers()

    for driver in vector_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


def translate_resample_method(method):
    """
    Translate a string of a resampling method to a **GDAL** integer (e.g. `gdal.GRA_NearestNeighbour`).

    ## Args:
    `method` (_str_): The resampling method. </br>

    ## Returns:
    (_int_): The **GDAL** resampling integer (e.g. `"nearest"=1`).
    """
    assert isinstance(method, str), "method must be a string."
    assert len(method) > 0, "method must be a non-empty string."

    methods = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubic_spline": gdal.GRA_CubicSpline,
        "cubicspline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        "average": gdal.GRA_Average,
        "mean": gdal.GRA_Average,
        "mode": gdal.GRA_Mode,
        "max": gdal.GRA_Max,
        "maximum": gdal.GRA_Max,
        "min": gdal.GRA_Min,
        "minimum": gdal.GRA_Min,
        "median": gdal.GRA_Med,
        "med": gdal.GRA_Med,
        "q1": gdal.GRA_Q1,
        "Q1": gdal.GRA_Q1,
        "q3": gdal.GRA_Q3,
        "Q3": gdal.GRA_Q3,
        "sum": gdal.GRA_RMS,
        "rms": gdal.GRA_RMS,
        "RMS": gdal.GRA_RMS,
    }

    if method in methods:
        return methods[method]

    raise ValueError("Unknown resample method: " + method)


def translate_gdal_dtype_to_str(gdal_datatype_int):
    """
    Translates the **GDAL** datatype integer into a string. Can be used by **NumPy**.

    ## Args:
    `gdal_datatype_int` (_gdal_datatype_int_): The **GDAL** datatype integer. </br>

    ## Returns:
    (_str_): The translated string (e.g. `0="uint8"`)
    """
    assert isinstance(gdal_datatype_int, int), f"gdal_datatype must be an integer. Received: {gdal_datatype_int}"

    datatypes = {
        str(gdal.GDT_Byte): "uint8",
        str(gdal.GDT_Int16): "int16",
        str(gdal.GDT_Int32): "int32",
        str(gdal.GDT_UInt16): "uint16",
        str(gdal.GDT_UInt32): "uint32",
        str(gdal.GDT_Float32): "float32",
        str(gdal.GDT_Float64): "float64",
        str(gdal.GDT_CFloat32): "cfloat32",
        str(gdal.GDT_CFloat64): "cfloat64",
        str(gdal.GDT_CInt16): "cint16",
        str(gdal.GDT_CInt32): "cint32",
    }

    dtype_str = str(gdal_datatype_int)
    for key in datatypes:
        if dtype_str == key:
            return datatypes[dtype_str]

    raise ValueError(f"Could not translate the datatype: {gdal_datatype_int}")


def translate_str_to_gdal_dtype(dtype_str):
    """
    Translates the datatype string into a **GDAL** datatype integer. Can be used by **GDAL**.

    ## Args:
    `dtype_str` (_dtype_str_): The datatype string (e.g. "float32"). </br>

    ## Returns:
    (_int_): The translated integer (e.g. `"uint8"=0`)
    """
    assert isinstance(dtype_str, (np.dtype, str)), "dtype_str must be a string or numpy dtype."

    if isinstance(dtype_str, np.dtype):
        dtype_str = dtype_str.name

    assert len(dtype_str) > 0, "dtype_str must be a non-empty string."

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
        "cfloat32": gdal.GDT_CFloat32,
        "cfloat64": gdal.GDT_CFloat64,
        "cint16": gdal.GDT_CInt16,
        "cint32": gdal.GDT_CInt32,
    }

    for key in datatypes:
        if dtype_str == key:
            return datatypes[dtype_str]

    raise ValueError(f"Could not translate the datatype: {dtype_str}")


def get_default_nodata_value(dtype):
    """
    Returns the default fill value for masked numpy arrays.

    ## Args:
    `dtype` (_numpy.dtype_/_str_/_int_): The dtype. </br>

    ## Returns:
    (_float_/_int_): The default fill value.
    """
    assert isinstance(dtype, (np.dtype, str, int)), "numpy_dtype must be a numpy.dtype or string."

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
        "cfloat32": -9999.0,
        "cfloat64": -9999.0,
    }

    test_type = dtype
    if isinstance(dtype, np.dtype):
        test_type = test_type.name
    elif isinstance(dtype, int):
        test_type = translate_gdal_dtype_to_str(dtype)

    if test_type in datatypes:
        return datatypes[test_type]

    raise ValueError("Unknown numpy datatype: " + test_type)


def is_gdal_datatype_a_float(gdal_dtype):
    """
    Checks if a **GDAL** datatype integer is a floating datatype: </br>
    `(Float32, Float64, cFloat32, cFloat64)`

    ## Args:
    `gdal_dtype` (_int_): The **GDAL** datatype integer. </br>

    ## Returns:
    (_bool_): **True** if datatype is a float, otherwise **False**.
    """
    assert isinstance(gdal_dtype, int), f"gdal_dtype must be an integer. Received: {gdal_dtype}"

    floats = [gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CFloat32, gdal.GDT_CFloat64]

    if gdal_dtype in floats:
        return True

    return False
