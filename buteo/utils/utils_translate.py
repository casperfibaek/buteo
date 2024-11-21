"""### GDAL Enum-like Functions. ###

Functions to translate between **GDAL** and **NumPy** datatypes.
"""

# Standard Library
from typing import List, Tuple, Union, Dict, Type, Optional

# External
import numpy as np
from osgeo import gdal, gdal_array



def _get_available_drivers() -> List[Dict[str, str]]:
    """Returns a list of all available drivers.

    Returns
    -------
    List[Dict[str, str]]
        List of dicts containing available drivers. Each dict has the following keys:
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


def _get_valid_raster_driver_extensions() -> List[str]:
    """Returns a list of valid raster driver extensions.

    Returns
    -------
    List[str]
        List of valid raster driver extensions.
    """
    available_raster_drivers, _available_vector_drivers = _get_available_drivers()

    valid_raster_driver_extensions = []

    for driver in available_raster_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:

            if driver["extension"] == "gpkg": continue

            valid_raster_driver_extensions.append(driver["extension"])

    return valid_raster_driver_extensions


def _get_valid_vector_driver_extensions() -> List[str]:
    """Returns a list of valid vector driver extensions.

    Returns
    -------
    List[str]
        List of valid vector driver extensions.
    """
    _available_raster_drivers, available_vector_drivers = _get_available_drivers()

    valid_vector_driver_extensions = []

    for driver in available_vector_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:
            valid_vector_driver_extensions.append(driver["extension"])

    return valid_vector_driver_extensions


def _get_valid_driver_extensions() -> List[str]:
    """Returns a list of all valid driver extensions (**GDAL** + **OGR**).

    Returns
    -------
    List[str]
        List of all valid driver extensions.
    """
    available_raster_drivers, available_vector_drivers = _get_available_drivers()

    valid_driver_extensions = []

    for driver in available_raster_drivers + available_vector_drivers:
        if driver["extension"] != "" or len(driver["extension"]) > 0:
            valid_driver_extensions.append(driver["extension"])

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
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

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
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

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
    assert isinstance(ext, str), "Extension must be a string."
    assert len(ext) > 0, "Extension must be a non-empty string."

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
    """
    assert _check_is_valid_raster_driver_extension(ext), f"Invalid extension: {ext}"

    raster_drivers, _vector_drivers = _get_available_drivers()

    for driver in raster_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


def _get_vector_shortname_from_ext(ext: str) -> str:
    """Converts a vector file extension to an **OGR** driver short_name name.

    Parameters
    ----------
    ext : str
        The file extension.

    Returns
    -------
    str
        The driver short name (e.g. GPKG).
    """
    assert _check_is_valid_vector_driver_extension(ext), f"Invalid extension: {ext}"

    _raster_drivers, vector_drivers = _get_available_drivers()

    for driver in vector_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


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
    """
    assert _check_is_valid_vector_driver_extension(ext) or _check_is_valid_raster_driver_extension(ext), f"Invalid extension: {ext}"

    raster_drivers, vector_drivers = _get_available_drivers()

    for driver in raster_drivers + vector_drivers:
        if driver["extension"] == ext:
            return driver["short_name"]

    raise RuntimeError(f"Invalid extension: {ext}")


def _translate_resample_method(method: str) -> int:
    """Translate a string of a resampling method to a GDAL integer (e.g. gdal.GRA_NearestNeighbour).

    Parameters
    ----------
    method : str
        The resampling method.

    Returns
    -------
    int
        The GDAL resampling method integer.
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
    }

    # Check GDAL version and add methods available in newer versions
    gdal_version = tuple(map(int, gdal.VersionInfo().split('.')))
    if gdal_version >= (3, 3, 0):
        extended_methods = {
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
        methods.update(extended_methods)
    else:
        print("Warning: Old version of GDAL running, only a subset of resampling methods are supported.")

    if method in methods:
        return methods[method]

    raise ValueError("Unknown resample method: " + method)


def _translate_dtype_gdal_to_numpy(gdal_datatype_int: int) -> np.dtype:
    """Translates the **GDAL** datatype integer into a **NumPy** datatype.

    Parameters
    ----------
    gdal_datatype_int : int
        The GDAL datatype integer.

    Returns
    -------
    np.dtype
        The **NumPy** datatype.
    """
    assert isinstance(gdal_datatype_int, int), f"gdal_datatype must be an integer. Received: {gdal_datatype_int}"

    dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_datatype_int))

    return dtype


def _translate_dtype_numpy_to_gdal(numpy_datatype: np.dtype) -> int:
    """Translates the **NumPy** datatype into a **GDAL** datatype integer.

    Parameters
    ----------
    numpy_datatype : np.dtype
        The **NumPy** datatype.

    Returns
    -------
    int
        The GDAL datatype integer.
    """
    assert isinstance(numpy_datatype, (np.dtype, str, int)), f"numpy_datatype must be a numpy.dtype, str or int. Received: {numpy_datatype}"

    parsed = _parse_dtype(numpy_datatype)
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(parsed)

    return dtype


def _get_default_nodata_value(dtype: Union[np.dtype, str, int]) -> Union[float, int]:
    """Returns the default fill value for masked numpy arrays.

    Parameters
    ----------
    dtype : Union[np.dtype, str, int]
        The data type of the array, can be either a numpy dtype object, a string representing a
            data type (e.g. 'float32') or an integer representing a numpy data type (e.g. 5 for 'float32').

    Returns
    -------
    Union[float, int]
        The default fill value for masked numpy arrays.
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
        test_type = _translate_dtype_gdal_to_numpy(dtype).name

    test_type = test_type.lower()

    if test_type in datatypes:
        return datatypes[test_type]

    raise ValueError("Unknown numpy datatype: " + test_type)


def _get_range_for_numpy_datatype(numpy_dtype: Union[str, np.dtype, int]) -> Tuple[Union[int, float], Union[int, float]]:
    """Returns the range of values that can be represented by a given numpy dtype.
    `(min_value, max_value)`

    Parameters
    ----------
    numpy_dtype : Union[str, np.dtype]
        The numpy dtype.

    Returns
    -------
    Tuple[Union[int, float], Union[int, float]]
        The range of values that can be represented by a given numpy dtype.
        `(min_value, max_value)`
    """
    assert isinstance(numpy_dtype, (str, np.dtype, int)), "numpy_dtype must be a numpy.dtype or string."

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
        test_type = _translate_dtype_gdal_to_numpy(numpy_dtype).name

    if test_type in datatypes:
        return datatypes[test_type]

    raise ValueError("Unknown numpy datatype: " + test_type)


def _check_is_value_within_dtype_range(
    value: Union[int, float],
    numpy_dtype: Union[str, np.dtype],
) -> bool:
    """Checks if a value is within the range of a numpy datatype.

    Parameters
    ----------
    value : Union[int, float]
        The value to check.

    numpy_dtype : Union[str, np.dtype]
        The numpy dtype.

    Returns
    -------
    bool
        True if the value is within the range of the numpy dtype, otherwise False.
    """
    if value is np.nan:
        return True

    min_val, max_val = _get_range_for_numpy_datatype(numpy_dtype)

    return min_val <= value <= max_val


def _check_is_gdal_dtype_float(gdal_dtype: int) -> bool:
    """Checks if a GDAL datatype integer is a floating-point datatype:
    (Float32, Float64, cFloat32, cFloat64)

    Parameters
    ----------
    gdal_dtype : int
        The GDAL datatype integer.

    Returns
    -------
    bool
        True if the GDAL datatype integer is a floating-point datatype, otherwise False.
    """
    assert isinstance(gdal_dtype, int), f"gdal_dtype must be an integer. Received: {gdal_dtype}"

    floats = [gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CFloat32, gdal.GDT_CFloat64]

    if gdal_dtype in floats:
        return True

    return False


def _parse_dtype(
    dtype: Union[str, np.dtype, int, Type[np.int64]]
) -> np.dtype:
    """Parses a numpy dtype from a string, numpy dtype, or GDAL datatype integer.

    Parameters
    ----------
    dtype : Union[str, np.dtype, int, type(np.int64)]
        The numpy dtype.

    Returns
    -------
    np.dtype
        The numpy dtype.
    """
    try:
        if isinstance(dtype, str):
            dtype = np.dtype(dtype.lower())
        elif isinstance(dtype, int):
            dtype = _translate_dtype_gdal_to_numpy(dtype)
        elif isinstance(dtype, type(np.int64)):
            dtype = np.dtype(dtype)
        elif isinstance(dtype, np.dtype):
            pass
        else:
            raise ValueError(f"Could not parse dtype: {dtype}")

    except Exception as e:
        raise ValueError(f"Could not parse dtype: {dtype}", e) from None

    return dtype


def _check_is_int_numpy_dtype(
    dtype: Union[str, np.dtype, int, Type[np.int64]],
):
    """Checks if a numpy dtype is an integer.

    Parameters
    ----------
    dtype : Union[str, np.dtype, int, type(np.int64)]
        The numpy dtype.

    Returns
    -------
    bool
        True if the numpy dtype is an integer, otherwise False.
    """
    dtype = _parse_dtype(dtype)

    if dtype.kind == "i":
        return True

    return False


def _safe_numpy_casting(
    arr: np.ndarray,
    target_dtype: Union[str, np.dtype, Type[np.int8]],
):
    """Safe casting of numpy arrays.
    Clips to min/max of destinations and rounds properly.

    Parameters
    ----------
    arr : np.ndarray
        The array to cast.

    target_dtype : Union[str, np.dtype, type(np.int8)]
        The target dtype.

    Returns
    -------
    np.ndarray
        The casted array.
    """
    assert isinstance(arr, np.ndarray), "arr must be a numpy array."
    assert isinstance(target_dtype, (str, np.dtype, type(np.int8))), "target_dtype must be a string or numpy dtype."

    target_dtype = _parse_dtype(target_dtype)

    if arr.dtype == target_dtype:
        return arr

    min_val, max_val = _get_range_for_numpy_datatype(target_dtype.name)
    outarr = np.zeros(arr.shape, dtype=target_dtype)

    if _check_is_int_numpy_dtype(target_dtype) and not _check_is_int_numpy_dtype(arr.dtype):
        outarr[:] = np.rint(arr)
        np.clip(outarr, min_val, max_val, out=outarr)
    else:
        np.clip(arr, min_val, max_val, out=outarr)

    return outarr
