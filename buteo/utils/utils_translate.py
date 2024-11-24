"""### GDAL Enum-like Functions. ###

Functions to translate between **GDAL** and **NumPy** datatypes.
"""

# Standard Library
from typing import List, Tuple, Union, Dict, Type

# External
import numpy as np
from osgeo import gdal, gdal_array



def _get_available_drivers() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Returns lists of available GDAL raster and vector drivers.

    Returns
    -------
    Tuple[List[Dict[str, str]], List[Dict[str, str]]]
        Two lists of dicts containing available drivers. Each dict has:
        - 'short_name': Driver short name (e.g. GTiff)
        - 'long_name': Driver long name (e.g. GeoTiff)
        - 'extension': Driver file extension (e.g. tif) or empty string
    """
    raster_drivers = []
    vector_drivers = []

    for idx in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(idx)
        metadata = driver.GetMetadata_Dict()

        extensions = driver.GetMetadataItem(gdal.DMD_EXTENSIONS)
        extensions = extensions.split(" ") if extensions else [""]

        driver_info = {
            "short_name": str(driver.ShortName),
            "long_name": str(driver.LongName),
        }

        if 'DCAP_RASTER' in metadata:
            for ext in extensions:
                raster_drivers.append({**driver_info, "extension": ext or ""})

        if 'DCAP_VECTOR' in metadata:
            for ext in extensions:
                vector_drivers.append({**driver_info, "extension": ext or ""})

    return raster_drivers, vector_drivers


def _get_valid_raster_driver_extensions() -> List[str]:
    """Returns a list of valid raster driver extensions.

    Returns
    -------
    List[str]
        List of valid raster driver extensions.
    """
    available_raster_drivers, _ = _get_available_drivers()

    valid_raster_driver_extensions = [
        driver["extension"]
        for driver in available_raster_drivers
        if driver["extension"] and driver["extension"] != "gpkg"
    ]

    return valid_raster_driver_extensions


def _get_valid_vector_driver_extensions() -> List[str]:
    """Returns a list of valid vector driver extensions.

    Returns
    -------
    List[str]
        List of valid vector driver extensions.
    """
    _, available_vector_drivers = _get_available_drivers()

    valid_vector_driver_extensions = [
        driver["extension"]
        for driver in available_vector_drivers
        if driver["extension"]
    ]

    return valid_vector_driver_extensions


def _get_valid_driver_extensions() -> List[str]:
    """Returns a list of all valid driver extensions (GDAL + OGR).

    Returns
    -------
    List[str]
        List of all valid driver extensions.
    """
    available_raster_drivers, available_vector_drivers = _get_available_drivers()

    valid_driver_extensions = [
        driver["extension"]
        for driver in available_raster_drivers + available_vector_drivers
        if driver["extension"]
    ]

    return valid_driver_extensions


def _check_is_valid_driver_extension(ext: str) -> bool:
    """Checks if a file extension is a valid GDAL or OGR driver extension.

    Parameters
    ----------
    ext : str
        The file extension.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if ext is None:
        return False

    if not isinstance(ext, str):
        return False

    if not ext:
        return False

    return ext in _get_valid_driver_extensions()


def _check_is_valid_raster_driver_extension(ext: str) -> bool:
    """Checks if a raster file extension is a valid GDAL driver extension.

    Parameters
    ----------
    ext : str
        The file extension.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if ext is None:
        return False

    if not isinstance(ext, str):
        return False

    if not ext:
        return False

    return ext in _get_valid_raster_driver_extensions()


def _check_is_valid_vector_driver_extension(ext: str) -> bool:
    """Checks if a vector file extension is a valid **OGR** driver extension.

    Parameters
    ----------
    ext : str
        The file extension.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if ext is None:
        return False

    if not isinstance(ext, str):
        return False

    if not ext:
        return False

    return ext in _get_valid_vector_driver_extensions()


def _get_raster_shortname_from_ext(ext: str) -> str:
    """Converts a raster file extension to a GDAL driver short name.

    Parameters
    ----------
    ext : str
        The file extension.

    Returns
    -------
    str
        The driver short name (e.g. GTiff).

    Raises
    ------
    AssertionError
        If ext is not a valid raster driver extension.
    RuntimeError
        If no matching driver is found for the extension.
    """
    if not ext:
        raise AssertionError("Extension cannot be empty")

    if not isinstance(ext, str):
        raise AssertionError("Extension must be a string")

    assert _check_is_valid_raster_driver_extension(ext), f"Invalid extension: {ext}"

    raster_drivers, _ = _get_available_drivers()

    for driver in raster_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"No driver found for extension: {ext}")


def _get_vector_shortname_from_ext(ext: str) -> str:
    """Converts a vector file extension to an OGR driver short name.

    Parameters
    ----------
    ext : str
        The file extension.

    Returns
    -------
    str
        The driver short name (e.g. GPKG).

    Raises
    ------
    AssertionError
        If ext is not a valid vector driver extension.
    RuntimeError
        If no matching driver is found for the extension.
    """
    if not ext:
        raise AssertionError("Extension cannot be empty")

    if not isinstance(ext, str):
        raise AssertionError("Extension must be a string")

    assert _check_is_valid_vector_driver_extension(ext), f"Invalid extension: {ext}"

    _, vector_drivers = _get_available_drivers()

    for driver in vector_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"No driver found for extension: {ext}")


def _get_driver_shortname_from_ext(ext: str) -> str:
    """Converts a file extension to a driver short name for either OGR or GDAL.

    Parameters
    ----------
    ext : str
        The file extension.

    Returns
    -------
    str
        The driver short name (e.g. GPKG).

    Raises
    ------
    AssertionError
        If ext is None, empty or not a string.
    ValueError
        If extension is not a valid GDAL/OGR driver extension.
    """
    if not ext:
        raise AssertionError("Extension cannot be empty")

    if not isinstance(ext, str):
        raise AssertionError("Extension must be a string")

    if _check_is_valid_raster_driver_extension(ext):
        return _get_raster_shortname_from_ext(ext)

    if _check_is_valid_vector_driver_extension(ext):
        return _get_vector_shortname_from_ext(ext)

    raise ValueError(f"Invalid extension: {ext}")


def _translate_resample_method(method: str) -> int:
    """Translate a string of a resampling method to a GDAL integer.

    Parameters
    ----------
    method : str
        The resampling method (e.g. 'nearest', 'bilinear').

    Returns
    -------
    int
        The GDAL resampling method integer (e.g. gdal.GRA_NearestNeighbour).

    Raises
    ------
    ValueError
        If method is None, not a string, or not a valid resampling method.
    """
    if not isinstance(method, str):
        raise ValueError(f"Method must be a string, got: {type(method)}")

    if not method:
        raise ValueError("Method cannot be empty")

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

    if gdal.VersionInfo().startswith("3"):
        methods.update({
            "max": gdal.GRA_Max,
            "minimum": gdal.GRA_Min,
            "median": gdal.GRA_Med,
            "q1": gdal.GRA_Q1,
            "q3": gdal.GRA_Q3,
            "sum": gdal.GRA_Sum,
            "rms": gdal.GRA_RMS,
        })

    method_lower = method.lower()
    if method_lower in methods:
        return methods[method_lower]

    raise ValueError(f"Unknown resampling method: {method}")


def _translate_dtype_gdal_to_numpy(gdal_datatype_int: int) -> np.dtype:
    """Translates the GDAL datatype integer into a NumPy datatype.

    Parameters
    ----------
    gdal_datatype_int : int
        The GDAL datatype integer.

    Returns
    -------
    np.dtype
        The NumPy datatype.

    Raises
    ------
    TypeError
        If gdal_datatype_int is not an integer.
    ValueError
        If the GDAL datatype cannot be converted to a NumPy dtype.
    """
    if not isinstance(gdal_datatype_int, int):
        raise TypeError(f"gdal_datatype must be an integer, got: {type(gdal_datatype_int)}")

    try:
        numeric_type = gdal_array.GDALTypeCodeToNumericTypeCode(gdal_datatype_int)
        if numeric_type is None:
            raise ValueError(f"Invalid GDAL datatype: {gdal_datatype_int}")
        return np.dtype(numeric_type)
    except Exception as e:
        raise ValueError(f"Failed to convert GDAL datatype {gdal_datatype_int} to NumPy dtype") from e


def _translate_dtype_numpy_to_gdal(numpy_datatype: Union[str, np.dtype, int]) -> int:
    """Translates the NumPy datatype into a GDAL datatype integer.

    Parameters
    ----------
    numpy_datatype : Union[str, np.dtype, int]
        The NumPy datatype, can be string, numpy.dtype or GDAL integer.

    Returns
    -------
    int
        The GDAL datatype integer.

    Raises
    ------
    TypeError
        If numpy_datatype is None or not of correct type.
    ValueError
        If numpy_datatype cannot be converted to GDAL type.
    """
    if numpy_datatype is None:
        raise TypeError("numpy_datatype cannot be None")

    if not isinstance(numpy_datatype, (np.dtype, str, int)):
        raise TypeError(f"numpy_datatype must be numpy.dtype, str or int. Got: {type(numpy_datatype)}")

    try:
        parsed = _parse_dtype(numpy_datatype)
        gdal_type = gdal_array.NumericTypeCodeToGDALTypeCode(parsed)

        if gdal_type is None:
            raise ValueError(f"Could not convert {numpy_datatype} to GDAL type")

        return gdal_type
    except Exception as e:
        raise ValueError(f"Failed to convert {numpy_datatype} to GDAL type") from e


def _get_default_nodata_value(dtype: Union[np.dtype, str, int]) -> Union[float, int]:
    """Returns the default fill value for masked numpy arrays.

    Parameters
    ----------
    dtype : Union[np.dtype, str, int]
        The data type, can be numpy dtype, string, or GDAL integer type

    Returns
    -------
    Union[float, int]
        The default fill value for masked arrays

    Raises
    ------
    TypeError
        If dtype is None or invalid type
    ValueError
        If dtype is not recognized
    """
    if dtype is None:
        raise TypeError("dtype cannot be None")

    if not isinstance(dtype, (np.dtype, str, int)):
        raise TypeError(f"dtype must be numpy.dtype, str or int, got: {type(dtype)}")

    datatypes = {
        "int8": -127,
        "int16": -32767,
        "int32": -2147483647,
        "int64": -9223372036854775807,
        "uint8": 255,
        "byte": 255,
        "uint16": 65535,
        "uint32": 4294967295,
        "uint64": 18446744073709551615,
        "float16": -9999.0,
        "float32": -9999.0,
        "float64": -9999.0,
        "cfloat32": -9999.0,
        "cfloat64": -9999.0,
    }

    try:
        if isinstance(dtype, int):
            dtype_name = _translate_dtype_gdal_to_numpy(dtype).name
        else:
            dtype_name = np.dtype(dtype).name

        if dtype_name.lower() in datatypes:
            return datatypes[dtype_name.lower()]

        raise ValueError(f"Unsupported dtype: {dtype_name}")
    except Exception as e:
        raise ValueError(f"Invalid dtype: {dtype}") from e


def _get_range_for_numpy_datatype(numpy_dtype: Union[str, np.dtype, int]) -> Tuple[Union[int, float], Union[int, float]]:
    """Returns the range of values that can be represented by a given numpy dtype.

    Parameters
    ----------
    numpy_dtype : Union[str, np.dtype, int]
        The numpy dtype. Can be a string, numpy.dtype, or GDAL integer type.

    Returns
    -------
    Tuple[Union[int, float], Union[int, float]]
        The minimum and maximum values that can be represented by the dtype.

    Raises
    ------
    TypeError
        If numpy_dtype is None or not of correct type.
    ValueError
        If numpy_dtype is not a recognized datatype.
    """
    if numpy_dtype is None:
        raise TypeError("numpy_dtype cannot be None")

    if not isinstance(numpy_dtype, (str, np.dtype, int)):
        raise TypeError(f"numpy_dtype must be str, numpy.dtype, or int. Got: {type(numpy_dtype)}")

    datatypes = {
        "int8": (-128, 127),
        "int16": (-32768, 32767),
        "int32": (-2147483648, 2147483647),
        "int64": (-9223372036854775808, 9223372036854775807),
        "uint8": (0, 255),
        "byte": (0, 255),
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

    try:
        if isinstance(numpy_dtype, int):
            dtype_name = _translate_dtype_gdal_to_numpy(numpy_dtype).name
        else:
            dtype_name = np.dtype(numpy_dtype).name

        if dtype_name in datatypes:
            return datatypes[dtype_name]

        raise ValueError(f"Unsupported dtype: {dtype_name}")
    except Exception as e:
        raise ValueError(f"Invalid dtype: {numpy_dtype}") from e


def _check_is_value_within_dtype_range(
    value: Union[int, float],
    numpy_dtype: Union[str, np.dtype, int]
) -> bool:
    """Checks if a value is within the range of a numpy datatype.

    Parameters
    ----------
    value : Union[int, float]
        The value to check.
    numpy_dtype : Union[str, np.dtype, int]
        The numpy dtype. Can be string, numpy.dtype, or GDAL integer.

    Returns
    -------
    bool
        True if the value is within range or is NaN, False otherwise.

    Raises
    ------
    TypeError
        If value is None or not numeric, or if numpy_dtype is invalid.
    """
    if value is None:
        raise TypeError("Value cannot be None")

    if not isinstance(value, (int, float)):
        raise TypeError(f"Value must be numeric, got: {type(value)}")

    if numpy_dtype is None:
        raise TypeError("numpy_dtype cannot be None")

    if np.isnan(value):
        return True

    min_val, max_val = _get_range_for_numpy_datatype(numpy_dtype)
    return min_val <= value <= max_val


def _check_is_gdal_dtype_float(gdal_dtype: int) -> bool:
    """Checks if a GDAL datatype integer is a floating-point datatype.

    Parameters
    ----------
    gdal_dtype : int
        The GDAL datatype integer.

    Returns
    -------
    bool
        True if the GDAL datatype is Float32, Float64, CFloat32 or CFloat64.

    Raises
    ------
    TypeError
        If gdal_dtype is None or not an integer.
    """
    if gdal_dtype is None:
        raise TypeError("gdal_dtype cannot be None")

    if not isinstance(gdal_dtype, int):
        raise TypeError(f"gdal_dtype must be an integer, got: {type(gdal_dtype)}")

    return gdal_dtype in [
        gdal.GDT_Float32,
        gdal.GDT_Float64,
        gdal.GDT_CFloat32,
        gdal.GDT_CFloat64
    ]

def _check_is_gdal_dtype_int(gdal_dtype: int) -> bool:
    """Checks if a GDAL datatype integer is an integer datatype.

    Parameters
    ----------
    gdal_dtype : int
        The GDAL datatype integer.

    Returns
    -------
    bool
        True if the GDAL datatype is Byte, Int16, Int32, UInt16, or UInt32.

    Raises
    ------
    TypeError
        If gdal_dtype is None or not an integer.
    """
    if gdal_dtype is None:
        raise TypeError("gdal_dtype cannot be None")

    if not isinstance(gdal_dtype, int):
        raise TypeError(f"gdal_dtype must be an integer, got: {type(gdal_dtype)}")

    return gdal_dtype in [
        gdal.GDT_Byte,
        gdal.GDT_Int8,
        gdal.GDT_Int16,
        gdal.GDT_Int32,
        gdal.GDT_UInt16,
        gdal.GDT_UInt32
    ]


def _parse_dtype(
    dtype: Union[str, np.dtype, int, Type[np.integer]]
) -> np.dtype:
    """Parses a numpy dtype from a string, numpy dtype, GDAL datatype integer, or numpy integer type.

    Parameters
    ----------
    dtype : Union[str, np.dtype, int, Type[np.integer]]
        The input dtype to parse.

    Returns
    -------
    np.dtype
        The parsed numpy dtype.

    Raises
    ------
    TypeError
        If dtype is None or of invalid type.
    ValueError
        If dtype cannot be parsed into a valid numpy dtype.
    """
    if dtype is None:
        raise TypeError("dtype cannot be None")

    try:
        if isinstance(dtype, str):
            return np.dtype(dtype.lower())
        if isinstance(dtype, int):
            return _translate_dtype_gdal_to_numpy(dtype)
        if isinstance(dtype, type) and (issubclass(dtype, np.integer) or issubclass(dtype, np.floating)):
            return np.dtype(dtype)
        if isinstance(dtype, np.dtype):
            return dtype

        raise TypeError(f"Invalid dtype type: {type(dtype)}")
    except Exception as e:
        raise ValueError(f"Could not parse dtype: {dtype}") from e


def _check_is_int_numpy_dtype(
    dtype: Union[str, np.dtype, int, Type[np.integer]]
) -> bool:
    """Checks if a numpy dtype is an integer type.

    Parameters
    ----------
    dtype : Union[str, np.dtype, int, Type[np.integer]]
        The dtype to check. Can be a string, numpy.dtype, GDAL integer, or numpy integer type.

    Returns
    -------
    bool
        True if the dtype represents an integer type, False otherwise.

    Raises
    ------
    TypeError
        If dtype is None.
    ValueError
        If dtype cannot be parsed into a valid numpy dtype.
    """
    if dtype is None:
        raise TypeError("dtype cannot be None")

    try:
        parsed_dtype = _parse_dtype(dtype)
        return parsed_dtype.kind in ["i", "u"]
    except Exception as e:
        raise ValueError(f"Invalid dtype: {dtype}") from e


def _check_is_float_numpy_dtype(
    dtype: Union[str, np.dtype, int, Type[np.integer]]
) -> bool:
    """Checks if a numpy dtype is a floating point type.

    Parameters
    ----------
    dtype : Union[str, np.dtype, int, Type[np.integer]]
        The dtype to check. Can be a string, numpy.dtype, GDAL integer, or numpy integer type.

    Returns
    -------
    bool
        True if the dtype represents a floating point type, False otherwise.

    Raises
    ------
    TypeError
        If dtype is None.
    ValueError
        If dtype cannot be parsed into a valid numpy dtype.
    """
    if dtype is None:
        raise TypeError("dtype cannot be None")

    try:
        parsed_dtype = _parse_dtype(dtype)
        return parsed_dtype.kind == "f"
    except Exception as e:
        raise ValueError(f"Invalid dtype: {dtype}") from e


def _safe_numpy_casting(
    arr: np.ndarray,
    target_dtype: Union[str, np.dtype, Type[np.integer]],
) -> np.ndarray:
    """Safe casting of numpy arrays.
    Clips to min/max of destinations and rounds properly.

    Parameters
    ----------
    arr : np.ndarray
        The array to cast.
    target_dtype : Union[str, np.dtype, Type[np.integer]]
        The target dtype.

    Returns
    -------
    np.ndarray
        The casted array.

    Raises
    ------
    TypeError
        If input array is not a numpy array or target_dtype is invalid.
    ValueError
        If the dtype cannot be parsed.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if not isinstance(target_dtype, (str, np.dtype, type)):
        raise TypeError("target_dtype must be a string, numpy dtype, or type")

    target_dtype = _parse_dtype(target_dtype)

    if arr.dtype == target_dtype:
        return arr

    min_val, max_val = _get_range_for_numpy_datatype(target_dtype.name)

    if _check_is_int_numpy_dtype(target_dtype):
        return np.clip(np.rint(arr), min_val, max_val).astype(target_dtype)

    return np.clip(arr, min_val, max_val).astype(target_dtype)
