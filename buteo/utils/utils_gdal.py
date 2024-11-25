"""### Utility functions to work with GDAL. ###

These functions are used to interact with basic GDAL objects.
"""

# Standard Library
from typing import Optional, Union, List, Any, Tuple
from warnings import warn
import inspect

# External
from osgeo import gdal, ogr
import psutil
import numpy as np

# Internal
from buteo.utils import utils_path, utils_translate



def _get_default_creation_options(
    options: Optional[List[str]] = None,
) -> List[str]:
    """Takes a list of GDAL creation options and adds default values if not specified.

    Default options are:
    ```python
    >>> default_options = [
    ...     "TILED=YES",
    ...     "NUM_THREADS=ALL_CPUS",
    ...     "BIGTIFF=IF_SAFER",
    ...     "COMPRESS=LZW",
    ...     "BLOCKXSIZE=256",
    ...     "BLOCKYSIZE=256",
    ... ]
    ```

    Parameters
    ----------
    options : Optional[List[str]], optional
        A list of GDAL creation options. Default: None

    Returns
    -------
    List[str]
        A list of GDAL creation options with defaults added.

    Raises
    ------
    TypeError
        If options is not None or a list, or if any option is not a string.
    """
    if options is not None and not isinstance(options, list):
        raise TypeError("options must be a list or None")

    if options is not None and not all(isinstance(opt, str) for opt in options):
        raise TypeError("all options must be strings")

    internal_options = list(options) if options is not None else []
    opt_str = " ".join(internal_options).upper()

    default_pairs = [
        ("TILED", "TILED=YES"),
        ("NUM_THREADS", "NUM_THREADS=ALL_CPUS"),
        ("BIGTIFF", "BIGTIFF=IF_SAFER"),
        ("COMPRESS", "COMPRESS=LZW"),
        ("BLOCKXSIZE", "BLOCKXSIZE=256"),
        ("BLOCKYSIZE", "BLOCKYSIZE=256"),
    ]

    for key, value in default_pairs:
        if key not in opt_str:
            internal_options.append(value)

    return internal_options


def get_gdal_memory() -> List[str]:
    """Get a list of all active memory layers in GDAL.

    Returns
    -------
    List[str]
        A list of all active memory layers in GDAL. Empty list if no memory layers exist
        or if GDAL memory listing is not supported.
    """
    datasets: List[str] = []

    try:
        if hasattr(gdal, "listdir"):
            for ds in gdal.listdir("/vsimem"):
                if ds is None or not isinstance(ds.name, str):
                    continue

                name = ds.name
                datasets.append(
                    name if name.startswith("/vsimem") else f"/vsimem/{name}"
                )

        elif hasattr(gdal, "ReadDir"):
            read_datasets = gdal.ReadDir("/vsimem")
            if read_datasets is not None:
                datasets = [f"/vsimem/{ds}" for ds in read_datasets if ds is not None]

    except (RuntimeError, AttributeError, IOError) as e:
        warn(f"Error getting memory datasets: {str(e)}", RuntimeWarning)

    return datasets


def clear_gdal_memory() -> bool:
    """Clears all GDAL memory datasets.

    Returns
    -------
    bool
        True if all memory datasets were cleared successfully, False otherwise.

    Notes
    -----
    This function attempts to clear all GDAL memory datasets through multiple methods.
    It is not guaranteed to work in all cases.
    """
    try:
        memory_datasets = get_gdal_memory()
        if not memory_datasets:
            return True

        # First attempt: Using gdal.Unlink
        for dataset in memory_datasets:
            if dataset is not None:
                gdal.Unlink(dataset)

        remaining_datasets = get_gdal_memory()
        if not remaining_datasets:
            return True

        # Second attempt: Using driver.Delete
        for dataset_path in remaining_datasets:
            if dataset_path is None:
                continue

            try:
                # Try to open as raster
                ds = gdal.Open(dataset_path)
                if ds is not None:
                    driver = ds.GetDriver()
                    ds = None  # Close dataset before deletion
                    if driver is not None:
                        driver.Delete(dataset_path)
                    continue

                # Try to open as vector
                ds = ogr.Open(dataset_path)
                if ds is not None:
                    driver = ds.GetDriver()
                    ds = None  # Close dataset before deletion
                    if driver is not None:
                        driver.Delete(dataset_path)
            except (RuntimeError, AttributeError) as e:
                warn(f"Failed to delete dataset {dataset_path}: {str(e)}", RuntimeWarning)

        # Final check
        if not get_gdal_memory():
            return True

        warn("Some GDAL memory datasets could not be cleared", RuntimeWarning)
        return False

    except (RuntimeError, AttributeError) as e:
        warn(f"Error while clearing GDAL memory: {str(e)}", RuntimeWarning)
        return False


def _check_is_valid_ext(ext: str) -> bool:
    """Check if a file extension has a valid GDAL or OGR driver.

    Parameters
    ----------
    ext : str
        The file extension to check.

    Returns
    -------
    bool
        True if the file extension is a valid GDAL or OGR driver, False otherwise.

    Raises
    ------
    TypeError
        If ext is not a string
    """
    if ext is None:
        return False

    if not isinstance(ext, str):
        raise TypeError("ext must be a string")

    if len(ext.strip()) == 0:
        return False

    # Normalize extension by removing leading dots and converting to lowercase
    clean_ext = ext.lstrip(".").lower()

    try:
        valid_extensions = utils_translate._get_valid_driver_extensions()
        return clean_ext in valid_extensions
    except (RuntimeError, AttributeError):
        return False


def _check_is_valid_raster_ext(ext: str) -> bool:
    """Check if a file extension has a valid GDAL driver.

    Parameters
    ----------
    ext : str
        The file extension to check.

    Returns
    -------
    bool
        True if the file extension is a valid GDAL Raster driver, False otherwise.

    Raises
    ------
    TypeError
        If ext is not a string
    """
    if ext is None:
        return False

    if not isinstance(ext, str):
        raise TypeError("ext must be a string")

    if len(ext.strip()) == 0:
        return False

    # Normalize extension by removing leading dots and converting to lowercase
    clean_ext = ext.lstrip(".").lower()

    try:
        valid_extensions = utils_translate._get_valid_raster_driver_extensions()
        return clean_ext in valid_extensions
    except (RuntimeError, AttributeError):
        return False


def _check_is_valid_vector_ext(ext: str) -> bool:
    """Check if a file extension has a valid OGR driver.

    Parameters
    ----------
    ext : str
        The file extension to check.

    Returns
    -------
    bool
        True if the file extension is a valid OGR Vector driver, False otherwise.

    Raises
    ------
    TypeError
        If ext is not a string
    """
    if ext is None:
        return False

    if not isinstance(ext, str):
        raise TypeError("ext must be a string")

    if len(ext.strip()) == 0:
        return False

    # Normalize extension by removing leading dots and converting to lowercase
    clean_ext = ext.lstrip(".").lower()

    try:
        valid_extensions = utils_translate._get_valid_vector_driver_extensions()
        return clean_ext in valid_extensions
    except (RuntimeError, AttributeError):
        return False


def _check_is_file_valid_ext(file_path: str) -> bool:
    """Check if a file path has a valid GDAL or OGR driver.

    Parameters
    ----------
    file_path : str
        The file path to check.

    Returns
    -------
    bool
        True if the file path is a valid GDAL or OGR driver, False otherwise.

    Raises
    ------
    TypeError
        If file_path is not a string
    ValueError
        If file_path is empty
    """
    if file_path is None:
        return False

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    if len(file_path.strip()) == 0:
        raise ValueError("file_path cannot be empty")

    try:
        ext = utils_path._get_ext_from_path(file_path)
        if ext is None:
            return False

        valid_extensions = utils_translate._get_valid_driver_extensions()
        return ext.lower() in valid_extensions
    except (RuntimeError, AttributeError):
        return False


def _check_is_file_valid_raster_ext(file_path: str) -> bool:
    """Check if a file path has a valid GDAL driver.

    Parameters
    ----------
    file_path : str
        The file path to check.

    Returns
    -------
    bool
        True if the file path is a valid GDAL Raster driver, False otherwise.

    Raises
    ------
    TypeError
        If file_path is not a string
    ValueError
        If file_path is empty
    """
    if file_path is None:
        return False

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    if len(file_path.strip()) == 0:
        raise ValueError("file_path cannot be empty")

    try:
        ext = utils_path._get_ext_from_path(file_path)
        if ext is None:
            return False

        valid_extensions = utils_translate._get_valid_raster_driver_extensions()
        return ext.lower() in valid_extensions
    except (RuntimeError, AttributeError):
        return False


def _check_is_file_valid_vector_ext(file_path: str) -> bool:
    """Check if a file path has a valid OGR driver.

    Parameters
    ----------
    file_path : str
        The file path to check.

    Returns
    -------
    bool
        True if the file path is a valid OGR Vector driver, False otherwise.

    Raises
    ------
    TypeError
        If file_path is not a string
    ValueError
        If file_path is empty
    """
    if file_path is None:
        return False

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    if len(file_path.strip()) == 0:
        raise ValueError("file_path cannot be empty")

    try:
        ext = utils_path._get_ext_from_path(file_path)
        if ext is None:
            return False

        valid_extensions = utils_translate._get_valid_vector_driver_extensions()
        return ext.lower() in valid_extensions
    except (RuntimeError, AttributeError):
        return False


def _get_default_driver_raster() -> gdal.Driver:
    """Get the default GDAL raster driver.

    Returns
    -------
    gdal.Driver
        The default GDAL raster driver (GTiff).

    Raises
    ------
    RuntimeError
        If the GTiff driver cannot be loaded.
    """
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("Could not load GTiff driver. GDAL may not be properly configured.")

    return driver


def _get_default_driver_vector() -> ogr.Driver:
    """Get the default OGR vector driver.

    Returns
    -------
    ogr.Driver
        The default OGR vector driver (GPKG).

    Raises
    ------
    RuntimeError
        If the GPKG driver cannot be loaded.
    """
    driver = ogr.GetDriverByName("GPKG")
    if driver is None:
        raise RuntimeError("Could not load GPKG driver. GDAL/OGR may not be properly configured.")

    return driver


def _get_driver_name_from_path(file_path: str) -> str:
    """Convert a file path to a GDAL or OGR driver ShortName (e.g. "GTiff" for "new_york.tif")

    Parameters
    ----------
    file_path : str
        The file path to convert.

    Returns
    -------
    str
        The GDAL or OGR driver ShortName.

    Raises
    ------
    TypeError
        If file_path is not a string.
    ValueError
        If file_path is empty or if no valid driver could be found.
    """
    if file_path is None:
        raise TypeError("file_path cannot be None")

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    if len(file_path.strip()) == 0:
        raise ValueError("file_path cannot be empty")

    ext = utils_path._get_ext_from_path(file_path)
    if ext is None:
        raise ValueError(f"No file extension found in path: {file_path}")

    if _check_is_file_valid_ext(file_path):
        driver_name = utils_translate._get_driver_shortname_from_ext(ext)
        if driver_name is None:
            raise ValueError(f"No valid driver found for extension: {ext}")
        return driver_name

    raise ValueError(f"Unable to parse GDAL or OGR driver from path: {file_path}")


def _get_vector_driver_name_from_path(file_path: str) -> str:
    """Convert a file path to an OGR driver ShortName (e.g. "FlatGeoBuf" for "new_york.fgb")

    Parameters
    ----------
    file_path : str
        The file path to convert.

    Returns
    -------
    str
        The OGR driver ShortName.

    Raises
    ------
    TypeError
        If file_path is not a string
    ValueError
        If file_path is empty, no extension found, or no valid driver found
    """
    if file_path is None:
        raise TypeError("file_path cannot be None")

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    if len(file_path.strip()) == 0:
        raise ValueError("file_path cannot be empty")

    ext = utils_path._get_ext_from_path(file_path)
    if ext is None:
        raise ValueError(f"No file extension found in path: {file_path}")

    if _check_is_file_valid_vector_ext(file_path):
        driver_name = utils_translate._get_vector_shortname_from_ext(ext)

        if driver_name is None or not isinstance(driver_name, str):
            raise ValueError(f"No valid vector driver found for extension: {ext}")

        return driver_name

    raise ValueError(f"Unable to parse OGR driver from path: {file_path}")


def _get_raster_driver_name_from_path(file_path: str) -> str:
    """Convert a file path to a GDAL driver ShortName (e.g. "GTiff" for "new_york.tif")

    Parameters
    ----------
    file_path : str
        The file path to convert.

    Returns
    -------
    str
        The GDAL driver ShortName.

    Raises
    ------
    TypeError
        If file_path is not a string
    ValueError
        If file_path is empty, no extension found, or no valid driver found
    """
    if file_path is None:
        raise TypeError("file_path cannot be None")

    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    if len(file_path.strip()) == 0:
        raise ValueError("file_path cannot be empty")

    ext = utils_path._get_ext_from_path(file_path)
    if ext is None:
        raise ValueError(f"No file extension found in path: {file_path}")

    if _check_is_file_valid_raster_ext(file_path):
        driver_name = utils_translate._get_raster_shortname_from_ext(ext)

        if driver_name is None or not isinstance(driver_name, str):
            raise ValueError(f"No valid raster driver found for extension: {ext}")

        return driver_name

    raise ValueError(f"Unable to parse GDAL driver from path: {file_path}")


def _check_is_dataset_in_memory(
    raster_or_vector: Union[str, gdal.Dataset, ogr.DataSource],
) -> bool:
    """Check if a raster or vector dataset is in memory. Also works for /vsimem/ paths. etc..

    Parameters
    ----------
    raster_or_vector : Union[str, gdal.Dataset, ogr.DataSource]
        The raster or vector dataset to check.

    Returns
    -------
    bool
        True if the dataset is in memory, False otherwise.

    Raises
    ------
    TypeError
        If raster_or_vector is not a string, gdal.Dataset, or ogr.DataSource.
    """
    if raster_or_vector is None:
        return False

    if not isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)):
        raise TypeError("raster_or_vector must be a string, gdal.Dataset, or ogr.DataSource")

    # Handle string paths
    if isinstance(raster_or_vector, str):
        # Check if path points to in-memory dataset
        return bool(utils_path._check_file_exists_vsimem(raster_or_vector))

    # Handle GDAL/OGR datasets
    try:
        driver = raster_or_vector.GetDriver()
        if driver is None:
            return False

        # Get driver name, handling both GDAL and OGR drivers
        driver_name = getattr(driver, "ShortName", None)
        if driver_name is None:
            driver_name = driver.GetName()

        # Check if using memory driver
        memory_drivers = {"MEM", "Memory", "VirtualMem", "VirtualOGR", "Virtual"}
        if driver_name in memory_drivers:
            return True

        # Check if path points to in-memory dataset
        path = raster_or_vector.GetDescription()
        return bool(utils_path._check_file_exists_vsimem(path))

    except (AttributeError, RuntimeError):
        return False


def _check_is_memory_dataset(dataset: Union[str, gdal.Dataset, ogr.DataSource]) -> bool:
    """Check if a dataset is a Memory or MEM dataset.

    Parameters
    ----------
    dataset : Union[gdal.Dataset, ogr.DataSource]
        The dataset to check.

    Returns
    -------
    bool
        True if the dataset is a Memory or MEM dataset, False otherwise.

    Raises
    ------
    TypeError
        If dataset is not a gdal.Dataset or ogr.DataSource.
    """
    if dataset is None:
        return False

    if isinstance(dataset, str):
        return False

    if not isinstance(dataset, (gdal.Dataset, ogr.DataSource)):
        raise TypeError("dataset must be a string, gdal.Dataset or ogr.DataSource")

    try:
        driver = dataset.GetDriver()
        if driver is None:
            return False

        driver_name = getattr(driver, "ShortName", None)
        if driver_name is None:
            driver_name = driver.GetName()

        return driver_name in {"MEM", "Memory"}

    except (RuntimeError, AttributeError):
        return False


def delete_dataset_if_in_memory(
    raster_or_vector: Union[str, gdal.Dataset, ogr.DataSource],
) -> bool:
    """Delete raster or vector if it is in memory.

    Parameters
    ----------
    raster_or_vector : Union[str, gdal.Dataset, ogr.DataSource]
        The raster or vector dataset to delete.

    Returns
    -------
    bool
        True if the dataset was successfully deleted or wasn't in memory.
        False if deletion failed.

    Raises
    ------
    TypeError
        If raster_or_vector is None or not a string, gdal.Dataset, or ogr.DataSource.
    RuntimeError
        If using deprecated MEM/Memory drivers instead of /vsimem/.
    """
    if raster_or_vector is None:
        raise TypeError("raster_or_vector cannot be None")

    if not isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)):
        raise TypeError("raster_or_vector must be a string, gdal.Dataset, or ogr.DataSource")

    # Skip if not in memory
    if not _check_is_dataset_in_memory(raster_or_vector):
        return True

    try:
        # Check for deprecated memory drivers
        if _check_is_memory_dataset(raster_or_vector):
            raise RuntimeError("Using MEM/Memory drivers is deprecated. Use /vsimem/ paths instead.")

        # Handle /vsimem/ datasets
        if isinstance(raster_or_vector, str):
            path = raster_or_vector
        else:
            path = _get_path_from_dataset(raster_or_vector)

        if path.startswith("/vsimem/"):
            gdal.Unlink(path)
            return path not in get_gdal_memory()

        # Final attempt to delete any remaining in-memory datasets
        if isinstance(raster_or_vector, (gdal.Dataset, ogr.DataSource)):
            try:
                raster_or_vector.FlushCache()
            except (RuntimeError, AttributeError):
                pass
            raster_or_vector = None # type: ignore
            return True

    except (RuntimeError, AttributeError) as e:
        warn(f"Error deleting dataset: {str(e)}", RuntimeWarning)
        return False

    return True


def delete_dataset_if_in_memory_list(
    list_of_raster_or_vectors: List[Union[str, gdal.Dataset, ogr.DataSource]],
) -> bool:
    """Deletes a list of raster or vector datasets if they are in memory.

    Parameters
    ----------
    list_of_raster_or_vectors : List[Union[str, gdal.Dataset, ogr.DataSource]]
        A list of rasters or vectors to delete.

    Returns
    -------
    bool
        True if all datasets were successfully processed (either deleted if in memory
        or skipped if not in memory), False if any deletion failed.

    Raises
    ------
    TypeError
        If input is not a list or contains invalid types.
    ValueError
        If input list contains None values.
    """
    if list_of_raster_or_vectors is None:
        raise TypeError("Input cannot be None")

    if not isinstance(list_of_raster_or_vectors, list):
        raise TypeError("Input must be a list")

    if not list_of_raster_or_vectors:
        return True

    valid_types = (str, gdal.Dataset, ogr.DataSource)
    if not all(isinstance(x, valid_types) for x in list_of_raster_or_vectors if x is not None):
        raise TypeError("All elements must be string, gdal.Dataset, or ogr.DataSource")

    if any(x is None for x in list_of_raster_or_vectors):
        raise ValueError("List contains None values")

    try:
        results = [delete_dataset_if_in_memory(dataset) for dataset in list_of_raster_or_vectors]
        return all(results)
    except (RuntimeError, AttributeError) as e:
        warn(f"Error while deleting datasets: {str(e)}", RuntimeWarning)
        return False


def _delete_raster_or_vector(
    raster_or_vector: Union[str, gdal.Dataset, ogr.DataSource],
) -> bool:
    """Delete raster or vector. Can be used on both in memory and on disk datasets.

    Parameters
    ----------
    raster_or_vector : Union[str, gdal.Dataset, ogr.DataSource]
        The raster or vector to delete.

    Returns
    -------
    bool
        True if the raster or vector was successfully deleted, False otherwise.

    Raises
    ------
    TypeError
        If raster_or_vector is None or not a string, gdal.Dataset, or ogr.DataSource.
    """
    if raster_or_vector is None:
        raise TypeError("raster_or_vector cannot be None")

    if not isinstance(raster_or_vector, (str, gdal.Dataset, ogr.DataSource)):
        raise TypeError("raster_or_vector must be a string, gdal.Dataset, or ogr.DataSource")

    try:
        # Handle in-memory datasets first
        if delete_dataset_if_in_memory(raster_or_vector):
            return True

        # Get the path whether input is string or dataset
        path = (raster_or_vector if isinstance(raster_or_vector, str)
                else _get_path_from_dataset(raster_or_vector))
        path = path[0] if isinstance(path, list) else path

        # Skip if file doesn't exist
        if not utils_path._check_file_exists(path):
            return True

        # Get appropriate driver and delete
        driver_shortname = _get_driver_name_from_path(path)
        driver = gdal.GetDriverByName(driver_shortname)

        if driver is None:
            return False

        # Close dataset if it's open
        if isinstance(raster_or_vector, (gdal.Dataset, ogr.DataSource)):
            try:
                raster_or_vector.FlushCache()
            except (RuntimeError, AttributeError):
                pass
            raster_or_vector = None # type: ignore

        # Attempt deletion
        driver.Delete(path)

        # Verify deletion
        return not utils_path._check_file_exists(path)

    except (RuntimeError, IOError, OSError, AttributeError) as e:
        warn(f"Error deleting dataset: {str(e)}", RuntimeWarning)
        return False


def _check_is_raster_empty(
    raster: gdal.Dataset,
) -> bool:
    """Check if a raster is empty (has no bands, zero dimensions, or is None).

    Parameters
    ----------
    raster : gdal.Dataset
        The raster to check.

    Returns
    -------
    bool
        True if the raster is empty (no bands, zero dimensions, or None), False otherwise.

    Raises
    ------
    TypeError
        If raster is not a gdal.Dataset
    """
    if raster is None:
        return True

    if not isinstance(raster, gdal.Dataset):
        raise TypeError("raster must be a gdal.Dataset")

    try:
        # Check if raster has no bands
        if raster.RasterCount <= 0:
            return True

        # Check if raster has zero dimensions
        if raster.RasterXSize <= 0 or raster.RasterYSize <= 0:
            return True

        # Additional check for invalid metadata
        if raster.GetGeoTransform() is None:
            return True

        return False

    except (RuntimeError, AttributeError):
        # Return True if we can't access raster properties
        return True


def _check_is_vector_empty(
    vector: ogr.DataSource,
) -> bool:
    """Check if a vector is empty (has no layers, features, or valid geometries).

    Parameters
    ----------
    vector : ogr.DataSource
        The vector to check.

    Returns
    -------
    bool
        True if the vector is empty, False otherwise.

    Raises
    ------
    TypeError
        If vector is not an ogr.DataSource
    """
    if not isinstance(vector, ogr.DataSource):
        raise TypeError("vector must be an ogr.DataSource")

    try:
        layer_count = vector.GetLayerCount()
        if layer_count == 0:
            return True

        for layer_idx in range(layer_count):
            layer = vector.GetLayerByIndex(layer_idx)
            if layer is None:
                continue

            layer.ResetReading()
            feature_count = layer.GetFeatureCount()

            if feature_count <= 0:
                continue

            for _ in range(feature_count):
                feature = layer.GetNextFeature()
                if feature is None:
                    continue

                geom = feature.GetGeometryRef()
                if geom is not None and not geom.IsEmpty():
                    return False

        return True

    except Exception:
        raise RuntimeError("Unable to check if vector is empty") from None


def _check_is_raster(
    potential_raster: Any,
    empty_is_invalid: bool = True,
) -> bool:
    """Checks if a variable is a valid raster.

    Parameters
    ----------
    potential_raster : Any
        The variable to check. Expected to be either str path to raster or gdal.Dataset.

    empty_is_invalid : bool, optional
        If True, an empty raster is considered invalid. Default: True.

    Returns
    -------
    bool
        True if the variable is a valid raster, False otherwise.

    Notes
    -----
    A raster is considered valid if:
    - It exists (if path)
    - Can be opened by GDAL
    - Is not empty (if empty_is_invalid=True)
    """
    if potential_raster is None:
        return False

    # Handle string paths
    if isinstance(potential_raster, str):
        try:
            if not utils_path._check_file_exists(potential_raster):
                return False

            opened = gdal.Open(str(potential_raster), 0)
            if opened is None:
                return False

            if empty_is_invalid and _check_is_raster_empty(opened):
                return False

            opened = None
            return True

        except (RuntimeError, TypeError, ValueError):
            return False

    # Handle GDAL datasets
    if isinstance(potential_raster, gdal.Dataset):
        try:
            if empty_is_invalid and _check_is_raster_empty(potential_raster):
                return False
            return True

        except (RuntimeError, AttributeError):
            return False

    return False


def _check_is_raster_list(
    potential_raster_list: List[Any],
    empty_is_invalid: bool = True,
) -> bool:
    """Checks if a list of variables contains only valid rasters.

    Parameters
    ----------
    potential_raster_list : List[Any]
        The list of variables to check. Expected to contain str paths to rasters
        or gdal.Dataset objects.
    empty_is_invalid : bool, optional
        If True, an empty raster is considered invalid. Default: True.

    Returns
    -------
    bool
        True if all elements in the list are valid rasters, False otherwise.

    Notes
    -----
    A raster is considered valid if:
    - It exists (if path)
    - Can be opened by GDAL
    - Is not empty (if empty_is_invalid=True)
    - Is not None
    """
    if potential_raster_list is None:
        return False

    if not isinstance(potential_raster_list, list):
        return False

    if len(potential_raster_list) == 0:
        return False

    if any(element is None for element in potential_raster_list):
        return False

    try:
        return all(_check_is_raster(element, empty_is_invalid=empty_is_invalid)
                  for element in potential_raster_list)
    except (RuntimeError, AttributeError):
        return False


def _check_is_vector(
    potential_vector: Any,
    empty_is_invalid: bool = True,
) -> bool:
    """Checks if a variable is a valid vector.

    Parameters
    ----------
    potential_vector : Any
        The variable to check. Expected to be either str path to vector or ogr.DataSource.
    empty_is_invalid : bool, optional
        If True, an empty vector is considered invalid. Default: True.

    Returns
    -------
    bool
        True if the variable is a valid vector, False otherwise.

    Notes
    -----
    A vector is considered valid if:
    - It exists (if path)
    - Can be opened by OGR
    - Is not empty (if empty_is_invalid=True)
    - Is not None
    """
    if potential_vector is None:
        return False

    # Handle OGR DataSources
    if isinstance(potential_vector, ogr.DataSource):
        try:
            if empty_is_invalid and _check_is_vector_empty(potential_vector):
                return False
            return True
        except (RuntimeError, AttributeError):
            return False

    # Handle string paths
    if isinstance(potential_vector, str):
        try:
            if not utils_path._check_file_exists(potential_vector):
                # Special handling for memory vectors
                ext = utils_path._get_ext_from_path(potential_vector)
                if ext and ext.lower() in ["memory", "mem"]:
                    driver = ogr.GetDriverByName("Memory")
                    if driver is None:
                        return False
                    ds = driver.Open(potential_vector)
                    if ds is None:
                        return False
                    if empty_is_invalid and _check_is_vector_empty(ds):
                        return False
                    return True
                return False

            ds = ogr.Open(str(potential_vector))
            if ds is None:
                return False

            if empty_is_invalid and _check_is_vector_empty(ds):
                return False

            ds = None
            return True

        except (RuntimeError, TypeError, ValueError):
            return False

    return False


def _check_is_vector_list(
    potential_vector_list: List[Any],
    empty_is_invalid: bool = True,
) -> bool:
    """Checks if a variable is a valid list of vectors.

    Parameters
    ----------
    potential_vector_list : List[Any]
        The list of variables to check. Expected to contain str paths to vectors
        or ogr.DataSource objects.
    empty_is_invalid : bool, optional
        If True, an empty vector is considered invalid. Default: True.

    Returns
    -------
    bool
        True if all elements in the list are valid vectors, False otherwise.

    Notes
    -----
    A vector is considered valid if:
    - It exists (if path)
    - Can be opened by OGR
    - Is not empty (if empty_is_invalid=True)
    - Is not None
    """
    if potential_vector_list is None:
        return False

    if not isinstance(potential_vector_list, list):
        return False

    if len(potential_vector_list) == 0:
        return False

    if any(element is None for element in potential_vector_list):
        return False

    try:
        return all(_check_is_vector(element, empty_is_invalid=empty_is_invalid)
                  for element in potential_vector_list)
    except (RuntimeError, AttributeError):
        return False


def _check_is_raster_or_vector(
    potential_raster_or_vector: Any,
    empty_is_invalid: bool = True,
) -> bool:
    """Checks if a variable is a valid raster or vector.

    Parameters
    ----------
    potential_raster_or_vector : Union[str, gdal.Dataset, ogr.DataSource]
        The variable to check. Can be a path string, gdal.Dataset or ogr.DataSource.
    empty_is_invalid : bool, optional
        If True, an empty raster or vector is considered invalid. Default: True.

    Returns
    -------
    bool
        True if the variable is a valid raster or vector, False otherwise.

    Notes
    -----
    A dataset is considered valid if:
    - It exists (if path)
    - Can be opened by GDAL/OGR
    - Is not empty (if empty_is_invalid=True)
    - Is not None
    """
    if potential_raster_or_vector is None:
        return False

    try:
        # Check if it's a raster
        if _check_is_raster(potential_raster_or_vector, empty_is_invalid=empty_is_invalid):
            return True

        # Check if it's a vector
        if _check_is_vector(potential_raster_or_vector, empty_is_invalid=empty_is_invalid):
            return True

        return False

    except (RuntimeError, AttributeError):
        # Return False for any unexpected errors
        return False


def _check_is_raster_or_vector_list(
    potential_raster_or_vector_list: List[Any],
    empty_is_invalid: bool = True,
) -> bool:
    """Checks if a variable is a valid list of rasters or vectors.

    Parameters
    ----------
    potential_raster_or_vector_list : List[Any]
        The list of variables to check. Expected to contain paths to rasters/vectors,
        gdal.Dataset or ogr.DataSource objects.
    empty_is_invalid : bool, optional
        If True, an empty raster or vector is considered invalid. Default: True.

    Returns
    -------
    bool
        True if all elements in the list are valid rasters or vectors, False otherwise.

    Notes
    -----
    A dataset is considered valid if:
    - It exists (if path)
    - Can be opened by GDAL/OGR
    - Is not empty (if empty_is_invalid=True)
    - Is not None
    """
    if potential_raster_or_vector_list is None:
        return False

    if not isinstance(potential_raster_or_vector_list, list):
        return False

    if len(potential_raster_or_vector_list) == 0:
        return False

    if any(element is None for element in potential_raster_or_vector_list):
        return False

    try:
        return all(_check_is_raster_or_vector(element, empty_is_invalid=empty_is_invalid)
                  for element in potential_raster_or_vector_list)
    except (RuntimeError, AttributeError):
        # Return False for any unexpected errors
        return False


def _get_path_from_dataset(
    dataset: Union[str, gdal.Dataset, ogr.DataSource],
    dataset_type: Optional[str] = None,
) -> str:
    """Gets the path from a dataset. Can be vector or raster, string or opened dataset.

    Parameters
    ----------
    dataset : Union[str, gdal.Dataset, ogr.DataSource]
        The dataset to get the path from.
    dataset_type : Optional[str], optional
        The type of the dataset. Can be "raster", "vector" or None.
        If None, the type is guessed. Default: None.

    Returns
    -------
    str
        The unix-style path from the dataset.

    Raises
    ------
    TypeError
        If dataset is None or invalid type.
        If dataset_type is not None or str.
    ValueError
        If dataset is not a valid raster or vector.
        If specified dataset_type doesn't match actual type.
        If dataset path cannot be retrieved.
    """
    if dataset is None:
        raise TypeError("dataset cannot be None")

    if not isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)):
        raise TypeError("dataset must be a string, gdal.Dataset, or ogr.DataSource")

    if dataset_type is not None and not isinstance(dataset_type, str):
        raise TypeError("dataset_type must be None or str")

    if dataset_type is not None and dataset_type.lower() not in ["raster", "vector"]:
        raise ValueError("dataset_type must be 'raster', 'vector' or None")

    try:
        # Handle raster datasets
        if (dataset_type is None or dataset_type.lower() == "raster") and _check_is_raster(dataset, empty_is_invalid=False):
            if isinstance(dataset, str):
                raster = gdal.Open(dataset, 0)
                if raster is None:
                    raise ValueError(f"Could not open raster: {dataset}")
            else:
                raster = dataset

            try:
                path = raster.GetDescription()
                if not path:
                    raise ValueError("Could not get path from raster dataset")

                if isinstance(dataset, str):
                    raster = None

                return utils_path._get_unix_path(path)
            finally:
                if isinstance(dataset, str):
                    raster = None

        # Handle vector datasets
        if (dataset_type is None or dataset_type.lower() == "vector") and _check_is_vector(dataset, empty_is_invalid=False):
            if isinstance(dataset, str):
                vector = ogr.Open(dataset, 0)
                if vector is None:
                    raise ValueError(f"Could not open vector: {dataset}")
            else:
                vector = dataset

            try:
                path = vector.GetDescription()
                if not path:
                    raise ValueError("Could not get path from vector dataset")

                if isinstance(dataset, str):
                    vector = None

                return utils_path._get_unix_path(path)
            finally:
                if isinstance(dataset, str):
                    vector = None

        # If we reach here, dataset is neither valid raster nor vector
        if dataset_type:
            raise ValueError(f"Dataset is not a valid {dataset_type}")

        raise ValueError("Unable to retrieve path from dataset")

    except (RuntimeError, AttributeError, IOError) as e:
        if isinstance(e, (TypeError, ValueError)):
            raise ValueError(f"Error retrieving path from dataset: {str(e)}") from e

    raise ValueError(f"Error retrieving path from dataset: {dataset}")

def _get_path_from_dataset_list(
    datasets: List[Union[str, gdal.Dataset, ogr.DataSource]],
    allow_mixed: bool = False,
    dataset_type: Optional[str] = None,
) -> List[str]:
    """Gets the paths from a list of datasets.

    Parameters
    ----------
    datasets : List[Union[str, gdal.Dataset, ogr.DataSource]]
        The datasets to get the paths from.
    allow_mixed : bool, optional
        If True, vectors and rasters can be mixed. Default: False.
    dataset_type : Optional[str], optional
        The type of the datasets. Can be "raster", "vector" or None.
        If None, the type is guessed. Default: None.

    Returns
    -------
    List[str]
        The paths of the datasets in unix-style format.

    Raises
    ------
    TypeError
        If datasets is not a list.
        If dataset_type is not None or str.
        If allow_mixed is not bool.
    ValueError
        If datasets is empty.
        If dataset_type is invalid.
        If mixed types are found and allow_mixed is False.
        If any dataset is invalid or None.
    """
    # Type checking
    if datasets is None:
        raise TypeError("datasets cannot be None")

    if not isinstance(datasets, list):
        raise TypeError("datasets must be a list")

    if not datasets:
        raise ValueError("datasets cannot be empty")

    if not isinstance(allow_mixed, bool):
        raise TypeError("allow_mixed must be a bool")

    if dataset_type is not None:
        if not isinstance(dataset_type, str):
            raise TypeError("dataset_type must be None or str")
        if dataset_type.lower() not in ["raster", "vector"]:
            raise ValueError("dataset_type must be 'raster', 'vector' or None")

    # Remove None values and check for validity
    valid_datasets = [ds for ds in datasets if ds is not None]
    if len(valid_datasets) != len(datasets):
        raise ValueError("datasets cannot contain None values")

    rasters = False
    vectors = False
    outputs: List[str] = []

    # Process each dataset
    for ds in valid_datasets:
        current_type = None

        # Check dataset type
        if (dataset_type is None or dataset_type.lower() == "raster") and _check_is_raster(ds, empty_is_invalid=False):
            current_type = "raster"
            rasters = True
        elif (dataset_type is None or dataset_type.lower() == "vector") and _check_is_vector(ds, empty_is_invalid=False):
            current_type = "vector"
            vectors = True
        else:
            raise ValueError(f"Invalid dataset: {ds}")

        # Check for mixed types
        if rasters and vectors and not allow_mixed:
            raise ValueError("Mixed raster and vector datasets not allowed when allow_mixed=False")

        # Get path and enforce specific type if provided
        if dataset_type is not None and current_type != dataset_type.lower():
            raise ValueError(f"Dataset {ds} is not of type {dataset_type}")

        path = _get_path_from_dataset(ds, dataset_type=current_type)
        path = path[0] if isinstance(path, list) else path

        outputs.append(path)

    return outputs


def _get_raster_size(
    raster: Union[gdal.Dataset, str],
) -> Tuple[float, float]:
    """Gets the pixel size (resolution) of a raster in x and y dimensions.

    Parameters
    ----------
    raster : Union[gdal.Dataset, str]
        The raster to get the size from. Can be either a GDAL dataset or path to a raster.

    Returns
    -------
    Tuple[float, float]
        The x and y pixel sizes (resolution) as (xres, yres).
        Note: yres is always positive even though GDAL typically stores it as negative.

    Raises
    ------
    TypeError
        If raster is None or not a gdal.Dataset or str.
    ValueError
        If raster is not a valid raster or cannot be opened.
        If geotransform is invalid or None.
    """
    if raster is None:
        raise TypeError("raster cannot be None")

    if not isinstance(raster, (gdal.Dataset, str)):
        raise TypeError("raster must be a gdal.Dataset or str")

    if not _check_is_raster(raster, empty_is_invalid=False):
        raise ValueError("raster is not a valid raster")

    try:
        # Open raster if string path provided
        if isinstance(raster, str):
            reference = gdal.Open(raster, 0)
            if reference is None:
                raise ValueError(f"Could not open raster: {raster}")
        else:
            reference = raster

        # Get geotransform
        transform = reference.GetGeoTransform()
        if transform is None:
            raise ValueError("Could not get geotransform from raster")

        # Close dataset if we opened it
        if isinstance(raster, str):
            reference = None

        # Extract resolutions
        x_res = float(abs(transform[1]))  # Pixel width
        y_res = float(abs(transform[5]))  # Pixel height

        return x_res, y_res

    except Exception as e:
        if isinstance(e, (TypeError, ValueError)):
            raise
        raise ValueError(f"Error getting raster size: {str(e)}") from e


def _get_dynamic_memory_limit(
    proportion: float = 0.8,
    *,
    min_mb: int = 100,
    max_mb: Optional[int] = None,
    available: bool = False,
) -> int:
    """Returns a dynamic memory limit taking into account total memory and CPU cores.
    The return is in bytes if >= 10000, otherwise in megabytes. For GDAL.

    The value is interpreted as being in megabytes if the value is less than 10000.
    For values >= 10000, this is interpreted as bytes.

    Parameters
    ----------
    proportion : float, optional
        The proportion of total memory to use (between 0 and 1). Default: 0.8
    min_mb : int, optional
        The minimum number of megabytes to be returned. Default: 100
    max_mb : Optional[int], optional
        The maximum number of megabytes to be returned. Default: None
    available : bool, optional
        If True, consider available memory instead of total memory. Default: False

    Returns
    -------
    int
        The dynamic memory limit in bytes (if >= 10000) or megabytes (if < 10000).

    Raises
    ------
    TypeError
        If inputs are not of the correct type.
    ValueError
        If proportion is not between 0 and 1, or if min_mb/max_mb are invalid.
    """
    # Type checking
    if not isinstance(proportion, (int, float)):
        raise TypeError("proportion must be a number")
    if not isinstance(min_mb, int):
        raise TypeError("min_mb must be an integer")
    if max_mb is not None and not isinstance(max_mb, int):
        raise TypeError("max_mb must be an integer or None")
    if not isinstance(available, bool):
        raise TypeError("available must be a boolean")

    # Value validation
    if proportion <= 0.0 or proportion > 1.0:
        raise ValueError("proportion must be > 0 and <= 1")
    if min_mb <= 0:
        raise ValueError("min_mb must be > 0")
    if max_mb is not None and max_mb <= 0:
        raise ValueError("max_mb must be > 0")
    if max_mb is not None and max_mb < min_mb:
        raise ValueError("max_mb cannot be less than min_mb")

    try:
        # Calculate memory limit in MB
        vm = psutil.virtual_memory()
        memory = vm.available if available else vm.total
        dyn_limit = int(np.rint((memory * proportion) / (1024 ** 2)))

        # Apply limits
        dyn_limit = max(dyn_limit, min_mb)
        if max_mb is not None:
            dyn_limit = min(dyn_limit, max_mb)

        # Convert to bytes if >= 10000
        if dyn_limit >= 10000:
            dyn_limit = dyn_limit * (1024 ** 2)

        return int(dyn_limit)

    except Exception as e:
        raise RuntimeError(f"Error calculating memory limit: {str(e)}") from e


def _convert_to_band_list(
    band_number: Union[int, List[int]],
    band_count: int,
) -> List[int]:
    """Converts a band number or list of band numbers to a list of band numbers.

    Parameters
    ----------
    band_number : Union[int, List[int]]
        The band number or list of band numbers to convert.
        If -1, returns a list of all bands.
        Values must be between 1 and band_count inclusive.

    band_count : int
        The number of bands in the raster.
        Must be greater than 0.

    Returns
    -------
    List[int]
        The list of band numbers.

    Raises
    ------
    TypeError
        If band_number is not an int or List[int].
        If band_count is not an int.
        If band_number is None.
    ValueError
        If band_count is less than 1.
        If any band number is less than -1 or greater than band_count.
        If band_number list is empty.
    """
    # Input validation
    if band_number is None:
        raise TypeError("band_number cannot be None")

    if not isinstance(band_number, (int, list)):
        raise TypeError(f"band_number must be int or List[int], got {type(band_number)}")

    if not isinstance(band_count, int):
        raise TypeError(f"band_count must be int, got {type(band_count)}")

    if band_count < 1:
        raise ValueError("band_count must be greater than 0")

    # Handle all bands case (-1)
    if band_number == -1:
        return list(range(1, band_count + 1))

    # Convert to list if single integer
    input_list = band_number if isinstance(band_number, list) else [band_number]

    # Check for empty list
    if len(input_list) == 0:
        raise ValueError("band_number list cannot be empty")

    # Validate and convert all values
    try:
        output_list = [int(val) for val in input_list]
    except (TypeError, ValueError) as e:
        raise TypeError("all band numbers must be integers") from e

    # Validate band numbers are in valid range
    invalid_bands = [val for val in output_list if val < 1 or val > band_count]
    if invalid_bands:
        raise ValueError(
            f"band numbers must be between 1 and {band_count}, got {invalid_bands}"
        )

    return output_list
