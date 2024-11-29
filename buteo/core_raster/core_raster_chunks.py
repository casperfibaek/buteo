"""### Basic IO functions for working with Rasters. ###"""

# Standard library
from typing import List, Optional, Union, Tuple

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import utils_base
from buteo.core_raster.core_raster_array import raster_to_array
from buteo.core_raster.core_raster_info import get_metadata_raster
from buteo.core_raster.core_raster_offsets import _get_chunk_offsets_fixed_size, _get_chunk_offsets



class raster_to_array_chunks:
    """A class for reading raster data in chunks. The array will be split into x and y
    amount of chunks in the x and y directions. The output will be the read array
    and the offsets of the chunk in the raster. The offset can be used to reconstitute
    the array into the original raster or a new raster representing the chunk,
    using the :func:`array_to_raster` function.

    Parameters
    ----------
    raster : Union[gdal.Dataset, str]
        The raster to read.
    chunks : int
        The number of chunks to read. The area is chunked in way that ensures
        that the chunks are as square as possible. Default: 1.
    chunk_size : list or tuple, optional
        The raster can be split into chunks of a fixed size,
        instead of splitting into a fixed number of chunks.
        The list should be in the format [x_size, y_size].
        If this is provided, the chunks parameter is ignored. Default: None.
    overlap : int, optional
        The number of pixels to overlap. Default: 0.
    bands : list or str or int, optional
        The bands to read. Can be "all", an int, or a list of integers, or a single
        integer. Please note that bands are 1-indexed. Default: "all".
    filled : bool, optional
        Whether to fill masked values. Default: False.
    fill_value : int or float, optional
        The value to fill masked values with. Default: None.
    skip : int, optional
        The number of chunks to skip when iterating. Default: 0.
    border_strategy : int, optional
        The border strategy to use when splitting the raster into chunks.
        border_strategy ignored when chunk_size and overlaps are provided.
        Only applied when chunk_size is provided. Can be 1 or 2. Default: 1.
        1. Ignore the border chunks if they do not fit the chunk size.
        2. Oversample the border chunks to fit the chunk size.
        3. Shrink the last chunk to fit the image size. (Creates uneven chunks.)
    cast : type or str, optional
        The data type to cast the output to. Default: None.

    Returns
    -------
    generator
        A generator that yields the raster data in chunks and the offsets of the chunk
        in the raster in a tuple.

    Examples
    --------
    ```python
    >>> # Read a raster into array via chunks.
    >>> import buteo as beo
    >>>
    >>> raster = "/path/to/raster/raster.tif"
    >>>
    >>> shape = beo.raster_to_metadata(raster)["shape"]
    >>> shape
    >>> (100, 100)
    >>>
    >>> for chunk, offsets in beo.raster_to_array_chunks(raster, chunks=4):
    >>>     print(chunk.shape, offsets)
    >>>     (25, 25), (0, 0, 25, 25)
    ```
    """

    def __init__(
        self,
        raster: Union[gdal.Dataset, str],
        chunks: int = 1,
        chunk_size: Optional[Union[List[int], Tuple[int, int]]] = None,
        *,
        overlap: int = 0,
        bands: Union[List[int], str, int] = 'all',
        filled: bool = False,
        fill_value: Optional[Union[int, float]] = None,
        skip: int = 0,
        border_strategy: int = 1,
        cast: Optional[Union[np.dtype, str]] = None,
    ):
        utils_base._type_check(raster, [gdal.Dataset, str], "raster")
        utils_base._type_check(chunks, [int], "chunks")
        utils_base._type_check(chunk_size, [list, tuple, type(None)], "chunk_size")
        utils_base._type_check(overlap, [int], "overlap")
        utils_base._type_check(bands, [list, str, int], "bands")
        utils_base._type_check(filled, [bool], "filled")
        utils_base._type_check(fill_value, [int, float, type(None)], "fill_value")
        utils_base._type_check(skip, [int], "skip")
        utils_base._type_check(border_strategy, [int], "border_strategy")

        self.raster = raster
        self.chunks = chunks
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.bands = bands
        self.filled = filled
        self.fill_value = fill_value
        self.skip = skip
        self.border_strategy = border_strategy
        self.cast = cast
        self.current_chunk = 0

        self.shape = get_metadata_raster(self.raster)["shape"]

        if self.chunks < 1:
            raise ValueError("The number of chunks must be greater than 0.")
        if self.overlap < 0:
            raise ValueError("Overlap must be greater than or equal to 0.")
        if self.chunks <= self.shape[1]:
            raise ValueError("The number of chunks must be less than or equal to the number of columns in the raster.")
        if self.chunks <= self.shape[0]:
            raise ValueError("The number of chunks must be less than or equal to the number of rows in the raster.")
        if self.chunk_size is not None:
            if len(self.chunk_size) != 2:
                raise ValueError("Chunk size must be a list or tuple of length 2.")
            if any([val < 1 for val in self.chunk_size]):
                raise ValueError("Chunk size must be greater than 0.")
        if self.border_strategy not in [1, 2, 3]:
            raise ValueError("The border strategy must be 1, 2, or 3.")

        if self.chunk_size is not None:
            # These now assume channel_first
            self.offsets = _get_chunk_offsets_fixed_size(
                self.shape,
                self.chunk_size[0],
                self.chunk_size[1],
                self.border_strategy,
                self.overlap,
            )

        else:
            # These now assume channel_first
            self.offsets = _get_chunk_offsets(
                self.shape,
                self.chunks,
                self.overlap,
            )

        self.total_chunks = len(self.offsets)

    def __iter__(self):
        self.current_chunk = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if self.current_chunk >= self.total_chunks:
            raise StopIteration

        offset = self.offsets[self.current_chunk]
        self.current_chunk += 1 + self.skip

        return (
            raster_to_array(
                self.raster,
                bands=self.bands,
                filled=self.filled,
                fill_value=self.fill_value,
                pixel_offsets=offset,
                cast=self.cast,
            ),
            offset,
        )

    def __len__(self):
        return self.total_chunks
