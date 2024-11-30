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
from buteo.core_raster.core_raster_offsets import _get_patch_offsets_fixed_size, _get_patch_offsets


class raster_to_array_iterator:
    """A class for reading raster data in patches. The array will be split into x and y
    amount of patches in the x and y directions. The output will be the read array
    and the offsets of the patch in the raster. The offset can be used to reconstitute
    the array into the original raster or a new raster representing the patch,
    using the :func:`array_to_raster` function.

    Parameters
    ----------
    raster : Union[gdal.Dataset, str]
        The raster to read.
    patches : int
        The number of patches to read. The area is patched in a way that ensures
        that the patches are as square as possible. Default: 1.
    patch_size : list or tuple, optional
        The raster can be split into patches of a fixed size,
        instead of splitting into a fixed number of patches.
        The list should be in the format [x_size, y_size].
        If this is provided, the patches parameter is ignored. Default: None.
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
        The number of patches to skip when iterating. Default: 0.
    border_strategy : int, optional
        The border strategy to use when splitting the raster into patches.
        border_strategy ignored when patch_size and overlaps are provided.
        Only applied when patch_size is provided. Can be 1 or 2. Default: 1.
        1. Ignore the border patches if they do not fit the patch size.
        2. Oversample the border patches to fit the patch size.
        3. Shrink the last patch to fit the image size. (Creates uneven patches.)
    cast : type or str, optional
        The data type to cast the output to. Default: None.

    Returns
    -------
    generator
        A generator that yields the raster data in patches and the offsets of the patch
        in the raster in a tuple.

    Examples
    --------
    ```python
    >>> # Read a raster into array via patches.
    >>> import buteo as beo
    >>>
    >>> raster = "/path/to/raster/raster.tif"
    >>>
    >>> shape = beo.raster_to_metadata(raster)["shape"]
    >>> shape
    >>> (100, 100)
    >>>
    >>> for patch, offsets in beo.raster_to_array_patches(raster, patches=4):
    >>>     print(patch.shape, offsets)
    >>>     (25, 25), (0, 0, 25, 25)
    ```
    """

    def __init__(
        self,
        raster: Union[gdal.Dataset, str],
        patches: int = 1,
        patch_size: Optional[Union[List[int], Tuple[int, int]]] = None,
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
        utils_base._type_check(patches, [int], "patches")
        utils_base._type_check(patch_size, [list, tuple, type(None)], "patch_size")
        utils_base._type_check(overlap, [int], "overlap")
        utils_base._type_check(bands, [list, str, int], "bands")
        utils_base._type_check(filled, [bool], "filled")
        utils_base._type_check(fill_value, [int, float, type(None)], "fill_value")
        utils_base._type_check(skip, [int], "skip")
        utils_base._type_check(border_strategy, [int], "border_strategy")

        self.raster = raster
        self.patches = patches
        self.patch_size = patch_size
        self.overlap = overlap
        self.bands = bands
        self.filled = filled
        self.fill_value = fill_value
        self.skip = skip
        self.border_strategy = border_strategy
        self.cast = cast
        self.current_patch = 0

        self.shape = get_metadata_raster(self.raster)["shape"]

        if self.patches < 1:
            raise ValueError("The number of patches must be greater than 0.")
        if self.overlap < 0:
            raise ValueError("Overlap must be greater than or equal to 0.")
        if self.patch_size is not None:
            if len(self.patch_size) != 2:
                raise ValueError("Patch size must be a list or tuple of length 2.")
            if any([val < 1 for val in self.patch_size]):
                raise ValueError("Patch size must be greater than 0.")
            if self.patch_size[0] > self.shape[1] or self.patch_size[1] > self.shape[2]:
                raise ValueError("Patch size must be smaller than the raster size.")
        if self.border_strategy not in [1, 2, 3]:
            raise ValueError("The border strategy must be 1, 2, or 3.")

        if self.patch_size is not None:
            self.offsets = _get_patch_offsets_fixed_size(
                self.shape,
                self.patch_size[0],
                self.patch_size[1],
                self.border_strategy,
                self.overlap,
            )

        else:
            self.offsets = _get_patch_offsets(self.shape, self.patches, self.overlap)

        self.total_patches = len(self.offsets)

    def __iter__(self):
        self.current_patch = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if self.current_patch >= self.total_patches:
            raise StopIteration

        offset = self.offsets[self.current_patch]
        self.current_patch += 1 + self.skip

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
        return self.total_patches


class raster_to_array_iterator_random:
    """A class for reading raster data in random patches of fixed size.
    The array will be read from random locations in the raster.
    The output will be the read array and the offsets of the patch in the raster.

    Parameters
    ----------
    raster : Union[gdal.Dataset, str]
        The raster to read.
    patch_size : Union[List[int], Tuple[int, int]]
        The size of the patches to read, in the format [x_size, y_size].
    max_iter : int
        The maximum number of patches to read. Default: 1000000.
    bands : Union[List[int], str, int], optional
        The bands to read. Can be "all", an int, or a list of integers.
    filled : bool, optional
        Whether to fill masked values. Default: False.
    fill_value : Optional[Union[int, float]], optional
        The value to fill masked values with. Default: None.
    cast : Optional[Union[np.dtype, str]], optional
        The data type to cast the output to. Default: None.

    Yields
    ------
    Tuple[np.ndarray, Tuple[int, int, int, int]]
        A tuple containing the array patch and the offsets in the format (x_offset, y_offset, x_size, y_size).
    """

    def __init__(
        self,
        raster: Union[gdal.Dataset, str],
        patch_size: Union[List[int], Tuple[int, int]],
        max_iter: int = 1000000,
        *,
        bands: Union[List[int], str, int] = 'all',
        filled: bool = False,
        fill_value: Optional[Union[int, float]] = None,
        cast: Optional[Union[np.dtype, str]] = None,
    ):
        utils_base._type_check(raster, [gdal.Dataset, str], "raster")
        utils_base._type_check(patch_size, [list, tuple], "patch_size")
        utils_base._type_check(max_iter, [int], "max_iter")
        utils_base._type_check(bands, [list, str, int], "bands")
        utils_base._type_check(filled, [bool], "filled")
        utils_base._type_check(fill_value, [int, float, type(None)], "fill_value")
        utils_base._type_check(cast, [np.dtype, str, type(None)], "cast")

        self.raster = raster
        self.patch_size = patch_size
        self.max_iter = max_iter
        self.bands = bands
        self.filled = filled
        self.fill_value = fill_value
        self.cast = cast
        self.current_patch = 0

        self.shape = get_metadata_raster(self.raster)["shape"]

        if len(self.patch_size) != 2:
            raise ValueError("Patch size must be a list or tuple of length 2.")
        if self.patch_size[0] <= 0 or self.patch_size[1] <= 0:
            raise ValueError("Patch size must be greater than 0.")

        self.max_x = self.shape[2] - self.patch_size[0]
        self.max_y = self.shape[1] - self.patch_size[1]

        if self.max_x < 0 or self.max_y < 0:
            raise ValueError("Patch size is larger than the raster dimensions.")

    def __iter__(self):
        self.current_patch = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        if self.current_patch >= self.max_iter:
            raise StopIteration

        x_offset = np.random.randint(0, self.max_x + 1)
        y_offset = np.random.randint(0, self.max_y + 1)
        x_size = self.patch_size[0]
        y_size = self.patch_size[1]

        offset = (x_offset, y_offset, x_size, y_size)
        self.current_patch += 1

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
        return self.max_iter
