"""
### GDAL Enum Functions ###

Functions to translate between **GDAL** and **NumPy** datatypes.
"""
# Standard Library
from typing import List, Tuple, Union, Dict

# External
import numpy as np
from osgeo import gdal



def get_available_drivers() -> List[Dict[str, str]]:
    """
    Returns a list of all available drivers.

    Returns:
        List[Dict[str, str]]: List of dicts containing available drivers. Each dict has the following keys:
            - 'short_name' (str): Driver short name (e.g. GTiff).
            - 'long_name' (str): Driver long name (e.g. GeoTiff).
            - 'extension' (str): Driver file extension (e.g. tif). Note: Can be an empty string.
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


def get_valid_raster_driver_extensions() -> List[str]:
    """
    Returns a list of valid raster driver extensions.

    Returns:
        list: List of valid raster driver extensions.
    """
    available_raster_drivers, _available_vector_drivers = get_available_drivers()

    valid_raster_driver_extensions = []

    for driver in available_raster_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:

            if driver["extension"] == "gpkg": continue

            valid_raster_driver_extensions.append(driver["extension"])

    return valid_raster_driver_extensions


def get_valid_vector_driver_extensions() -> List[str]:
    """
    Returns a list of valid vector driver extensions.

    Returns:
        list: List of valid vector driver extensions.
    """
    _available_raster_drivers, available_vector_drivers = get_available_drivers()

    valid_vector_driver_extensions = []

    for driver in available_vector_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:
            valid_vector_driver_extensions.append(driver["extension"])

    return valid_vector_driver_extensions


def get_valid_driver_extensions() -> List[str]:
    """
    Returns a list of all valid driver extensions (**GDAL** + **OGR**).

    Returns:
        list: List of all valid driver extensions.
    """
    available_raster_drivers, available_vector_drivers = get_available_drivers()

    valid_driver_extensions = []

    for driver in available_raster_drivers + available_vector_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:
            valid_driver_extensions.append(driver["extension"])

    return valid_driver_extensions


def is_valid_driver_extension(ext: str) -> bool:
    """
    Checks if a file extension is a valid GDAL or OGR driver extension.

    Args:
        ext (str): The file extension.

    Returns:
        bool: True if valid, False otherwise.
    """
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

    return ext in get_valid_driver_extensions()


def is_valid_raster_driver_extension(ext: str) -> bool:
    """
    Checks if a raster file extension is a valid GDAL driver extension.

    Args:
        ext (str): The file extension.

    Returns:
        bool: True if valid, False otherwise.
    """
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

    return ext in get_valid_raster_driver_extensions()


def is_valid_vector_driver_extension(ext: str) -> bool:
    """
    Checks if a vector file extension is a valid **OGR** driver extension.

    Args:
        ext (str): The file extension.

    Returns:
        bool: True if valid, False otherwise.
    """
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

    return ext in get_valid_vector_driver_extensions()


def convert_raster_ext_to_driver_shortname(ext: str) -> str:
    """
    Converts a raster file extension to a GDAL driver short name.

    Args:
        ext (str): The file extension.

    Returns:
        str: The GDAL driver short name (e.g. "GTiff").
    """
    assert is_valid_raster_driver_extension(ext), f"Invalid extension: {ext}"

    raster_drivers, _vector_drivers = get_available_drivers()

    for driver in raster_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


def convert_extension_to_driver_shortname(ext: str) -> str:
    """
    Converts a file extension to a driver short name for either OGR or GDAL.

    Args:
        ext (str): The file extension.

    Returns:
        str: The driver short name (e.g. GPKG or GTiff).
    """
    assert is_valid_vector_driver_extension(ext) or is_valid_raster_driver_extension(ext), f"Invalid extension: {ext}"

    raster_drivers, vector_drivers = get_available_drivers()

    for driver in raster_drivers + vector_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


def convert_vector_ext_to_driver_shortname(ext: str) -> str:
    """
    Converts a vector file extension to an **OGR** driver short_name name.

    Args:
        ext (str): The file extension.

    Returns:
        str: The driver short_name (e.g. GPKG).
    """
    assert is_valid_vector_driver_extension(ext), f"Invalid extension: {ext}"

    _raster_drivers, vector_drivers = get_available_drivers()

    for driver in vector_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


def translate_resample_method(method: str) -> int:
    """
    Translate a string of a resampling method to a GDAL integer (e.g. gdal.GRA_NearestNeighbour).

    Args:
        method (str): The resampling method.

    Returns:
        int: The GDAL resampling integer (e.g. "nearest"=1).
    """
    assert isinstance(method, str), "method must be a string."
    assert len(method) > 0, "method must be a non-empty string."
    try:
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
    except: # pylint: disable=bare-except
        print("Warning: Old version of GDAL running, only a subset of resampling methods are supported.")
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
        }

    if method in methods:
        return methods[method]

    raise ValueError("Unknown resample method: " + method)


def translate_gdal_dtype_to_str(gdal_datatype_int: int) -> str:
    """
    Translates the **GDAL** datatype integer into a string. Can be used by **NumPy**.

    Args:
        gdal_datatype_int (int): The **GDAL** datatype integer.

    Returns:
        str: The translated string (e.g. `0="uint8"`)
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


def translate_str_to_gdal_dtype(dtype_str: str) -> int:
    """
    Translates the datatype string into a **GDAL** datatype integer. Can be used by **GDAL**.

    Args:
    - dtype_str (str): The datatype string (e.g. "float32").

    Returns:
    - int: The **GDAL** datatype integer corresponding to the input datatype string (e.g. `"uint8"=1`).
    """
    assert isinstance(dtype_str, (np.dtype, str)), "dtype_str must be a string or numpy dtype."

    if isinstance(dtype_str, np.dtype):
        dtype_str = dtype_str.name

    dtype_str = dtype_str.lower()

    assert len(dtype_str) > 0, "dtype_str must be a non-empty string."

    datatypes = {
        "int8": gdal.GDT_Int16,
        "int16": gdal.GDT_Int16,
        "int32": gdal.GDT_Int32,
        "int64": gdal.GDT_Int32,
        "uint8": gdal.GDT_Byte,
        'bool': gdal.GDT_Byte,
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


def get_default_nodata_value(dtype: Union[np.dtype, str, int]) -> Union[float, int]:
    """
    Returns the default fill value for masked numpy arrays.

    Args:
        dtype: The data type of the array, can be either a numpy dtype object, a string representing a
            data type (e.g. 'float32') or an integer representing a numpy data type (e.g. 5 for 'float32').

    Returns:
        The default fill value for the given data type.
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

    test_type = test_type.lower()

    if test_type in datatypes:
        return datatypes[test_type]

    raise ValueError("Unknown numpy datatype: " + test_type)


def get_range_for_numpy_datatype(numpy_dtype: Union[str, np.dtype]) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Returns the range of values that can be represented by a given numpy dtype.

    Args:
        numpy_dtype (str/np.dtype): The numpy dtype.

    Returns:
        tuple: (min_value, max_value) that can be represented by the numpy dtype.
    """

    datatypes = {
        "int8": (-128, 127),
        "int16": (-32768, 32767),
        "int32": (-2147483648, 2147483647),
        "int64": (-9223372036854775808, 9223372036854775807),
        "uint8": (0, 255),
        "uint16": (0, 65535),
        "uint32": (0, 4294967295),
        "uint64": (0, 18446744073709551615),
        "float16": (-6.1e+4, 6.1e+4),
        "bfloat16": (-3.4e+38, 3.4e+38),
        "float32": (-3.4e+38, 3.4e+38),
        "float64": (-1.8e+308, 1.8e+308),
        "cfloat32": (-3.4e+38, 3.4e+38),
        "cfloat64": (-1.8e+308, 1.8e+308),
    }

    test_type = numpy_dtype
    if isinstance(numpy_dtype, np.dtype):
        test_type = test_type.name
    elif isinstance(numpy_dtype, int):
        test_type = translate_gdal_dtype_to_str(numpy_dtype)

    if test_type in datatypes:
        return datatypes[test_type]

    raise ValueError("Unknown numpy datatype: " + test_type)


def value_is_within_datatype_range(
    value: Union[int, float],
    numpy_dtype: Union[str, np.dtype],
) -> bool:
    """
    Checks if a value is within the range of a numpy datatype.

    Args:
        value (int/float): The value to check.
        numpy_dtype (str/np.dtype): The numpy dtype.

    Returns:
        bool: True if the value is within the range of the numpy dtype, False otherwise.
    """
    if value is np.nan:
        return True

    min_val, max_val = get_range_for_numpy_datatype(numpy_dtype)

    return min_val <= value <= max_val


def is_gdal_datatype_a_float(gdal_dtype: int) -> bool:
    """
    Checks if a GDAL datatype integer is a floating-point datatype:
    (Float32, Float64, cFloat32, cFloat64)

    Args:
        gdal_dtype (int): The GDAL datatype integer.

    Returns:
        bool: True if the datatype is a float, otherwise False.
    """
    assert isinstance(gdal_dtype, int), f"gdal_dtype must be an integer. Received: {gdal_dtype}"

    floats = [gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CFloat32, gdal.GDT_CFloat64]

    if gdal_dtype in floats:
        return True

    return False
