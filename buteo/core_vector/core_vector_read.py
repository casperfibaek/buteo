"""### Basic IO functions for working with vectors. ###

The basic module for interacting with vector data

    * More attribute functions
    * Repair vector functions
    * Sanity checks
    * Joins (by attribute, location, summary, etc..)
    * Union, Erase, ..
    * Sampling functions
    * Vector intersects, etc..

"""
# Standard library
from typing import Union, Optional, List, Sequence

# External
from osgeo import ogr, gdal

# Internal
from buteo.utils import (
    utils_base,
    utils_path,
    utils_io,
)



def _open_vector(
    vector: Union[str, ogr.DataSource],
    *,
    writeable: bool = False,
) -> ogr.DataSource:
    """Opens a vector in read or write mode.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    writeable : bool, optional
        If True, opens in write mode. Default: False

    Returns
    -------
    ogr.DataSource
        The opened vector dataset

    Raises
    ------
    TypeError
        If vector is not str or ogr.DataSource
    ValueError
        If vector path doesn't exist or file cannot be opened
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(writeable, [bool], "writeable")

    if isinstance(vector, ogr.DataSource):
        return vector

    if not utils_path._check_file_exists(vector):
        raise ValueError(f"Input vector does not exist: {vector}")

    if vector.startswith("/vsizip/"):
        writeable = False

    gdal.PushErrorHandler("CPLQuietErrorHandler")
    dataset = ogr.Open(vector, 1 if writeable else 0)
    gdal.PopErrorHandler()

    if dataset is None:
        raise ValueError(f"Could not open vector: {vector}")

    return dataset


def open_vector(
    vector: Union[str, ogr.DataSource, Sequence[Union[str, ogr.DataSource]]],
    *,
    writeable: bool = False,
) -> Union[ogr.DataSource, List[ogr.DataSource]]:
    """Opens one or more vector in read or write mode.

    Parameters
    ----------
    vector : str, ogr.DataSource, or Sequence[Union[str, ogr.DataSource]]
        Path(s) to vectors(s) or ogr DataSources(s)
    writeable : bool, optional
        Open in write mode. Default: False

    Returns
    -------
    Union[v, List[ogr.DataSource]]
        Single ogr DataSource or list of datasources

    Raises
    ------
    TypeError
        If input types are invalid
    ValueError
        If raster(s) cannot be opened
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "raster")
    utils_base._type_check(writeable, [bool], "writeable")

    input_is_sequence = isinstance(vector, Sequence) and not isinstance(vector, str)
    vectors = utils_io._get_input_paths(vector, "vector") # type: ignore

    opened = []
    for v in vectors:
        try:
            opened.append(_open_vector(v, writeable=writeable))
        except Exception as e:
            raise ValueError(f"Could not open raster: {v}") from e

    return opened if input_is_sequence else opened[0]


def _vector_get_layer(
    vector: ogr.DataSource,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> List[ogr.Layer]:
    """
    Get a layer from a vector dataset by name or index.

    Parameters
    ----------
    vector : ogr.DataSource
        The vector dataset
    layer_name_or_id : str or int, optional
        The name or index of the layer to get. If None, all layers are returned.
    
    Returns
    -------
    List[ogr.Layer]
        The layers in the vector dataset
    """
    utils_base._type_check(vector, [ogr.DataSource], "vector")
    utils_base._type_check(layer_name_or_id, [str, int, type(None)], "layer_name_or_id")

    if layer_name_or_id is not None:
        if isinstance(layer_name_or_id, int):
            layer = vector.GetLayer(layer_name_or_id)
            if layer is None:
                raise ValueError(f"Layer with index {layer_name_or_id} does not exist")
            layers = [layer]
        else:
            layer = vector.GetLayerByName(layer_name_or_id)
            if layer is None:
                raise ValueError(f"Layer with name '{layer_name_or_id}' does not exist")
            layers = [layer]
    else:
        layers = [vector.GetLayer(i) for i in range(vector.GetLayerCount())]

    return layers
