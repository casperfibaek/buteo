"""
### Perform convolutions on arrays.  ###
"""

# Standard Library
from typing import Tuple

# External
import numpy as np
from numba import jit



@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline="always")
def _distance_2D(p1: np.ndarray, p2: np.ndarray) -> float:
    """ Returns the distance between two points. (2D) """
    d1 = (p1[0] - p2[0]) ** 2
    d2 = (p1[1] - p2[1]) ** 2

    return np.sqrt(d1 + d2)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def _area_covered(square, radius):
    """
    Calculates the area covered by a circle within a square.
    Monte-carlo(ish) method. Can be parallelized.
    """
    n_points = 100
    min_y = square[:, 0].min()
    max_y = square[:, 0].max()
    min_x = square[:, 1].min()
    max_x = square[:, 1].max()

    steps = int(np.rint(np.sqrt(n_points)))
    range_y = np.linspace(min_x, max_x, steps)
    range_x = np.linspace(min_y, max_y, steps)

    center = np.array([0.0, 0.0], dtype=np.float32)
    adjusted_radius = radius + 0.5

    points_within = 0
    for y in range_y:
        for x in range_x:
            point = np.array([y, x], dtype=np.float32)
            if _distance_2D(center, point) <= adjusted_radius:
                points_within += 1

    area = points_within / (steps ** 2)

    return area


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def _circular_kernel_2D(radius):
    """ Creates a circular 2D kernel. Supports fractional radii. """
    size = np.int64(np.ceil(radius) * 2 + 1)
    kernel = np.zeros((size, size), dtype=np.float32)

    center = np.array([0.0, 0.0], dtype=np.float32)

    step = size // 2
    for idx_i, col in enumerate(range(-step, step + 1)):
        for idx_j, row in enumerate(range(-step, step + 1)):
            square = np.zeros((4, 2), dtype=np.float32)
            square[0] = np.array([col - 0.5, row - 0.5], dtype=np.float32)
            square[1] = np.array([col + 0.5, row - 0.5], dtype=np.float32)
            square[2] = np.array([col + 0.5, row + 0.5], dtype=np.float32)
            square[3] = np.array([col - 0.5, row + 0.5], dtype=np.float32)

            within = np.zeros(4, dtype=np.uint8)
            for i in range(4):
                within[i] = _distance_2D(center, square[i]) <= radius + 0.5

            # Case 1: completely inside
            if within.sum() == 4:
                kernel[idx_i][idx_j] = 1.0

            # Case 2: completely outside
            elif within.sum() == 0:
                kernel[idx_i][idx_j] = 0.0

            # Case 3: partially inside
            else:
                kernel[idx_i][idx_j] = _area_covered(square, radius)

    return kernel


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def _distance_weighted_kernel_2D(radius, method, decay=0.2, sigma=2.0):
    """
    Creates a distance weighted kernel.
    
    Parameters
    ----------

    radius : float
        Radius of the kernel.
    
    method : int
        Method to use for weighting.
        0. linear
        1. sqrt
        2. power
        3. log
        4. gaussian
        5. constant
    """
    size = np.int64(np.ceil(radius) * 2 + 1)
    kernel = np.zeros((size, size), dtype=np.float32)

    center = np.array([0.0, 0.0], dtype=np.float32)

    step = size // 2
    for idx_i, col in enumerate(range(-step, step + 1)):
        for idx_j, row in enumerate(range(-step, step + 1)):
            point = np.array([col, row], dtype=np.float32)
            distance = _distance_2D(center, point)

            # Linear
            if method == 0:
                kernel[idx_i, idx_j] = np.power((1 - decay), distance)

            # Sqrt
            elif method == 1:
                kernel[idx_i, idx_j] = np.power(np.sqrt((1 - decay)), distance)

            # Power
            elif method == 2:
                kernel[idx_i, idx_j] = np.power(np.power((1 - decay), 2), distance)

            # Gaussian
            elif method == 3:
                kernel[idx_i, idx_j] = np.exp(-(np.power(distance, 2)) / (2 * np.power(sigma, 2)))

            # Constant
            else:
                kernel[idx_i, idx_j] = 1.0

    return kernel


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def get_kernel(
    radius: float,
    circular: bool = False,
    distance_weighted: bool = False,
    normalised: bool = True,
    hole: bool = False,
    method: int = 0,
    decay: float = 0.2,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Creates a 2D kernel for convolution.

    Parameters
    ----------
    radius : float
        Radius of the kernel.
    
    circular : bool
        Whether to use a circular kernel.
    
    distance_weighted : bool
        Whether to use a distance weighted kernel.
    
    normalised : bool
        Whether to normalise the kernel.
    
    hole : bool
        Whether to create a hole in the center of the kernel.

    method : int
        Method to use for weighting.
        0. linear
        1. sqrt
        2. power
        3. gaussian
        4. constant
    
    decay : float
        Decay rate for distance weighted kernels. Only used if `distance_weighted` is True.

    sigma : float
        Sigma for gaussian distance weighted kernels. Only used if `distance_weighted` is True and `method` is 3.

    Returns
    -------
    kernel : np.ndarray
        The kernel.
    """
    size = np.int64(np.ceil(radius) * 2 + 1)
    kernel = np.ones((size, size), dtype=np.float32)

    if hole:
        kernel[size // 2, size // 2] = 0.0

    if circular:
        circular_kernel = _circular_kernel_2D(radius)
        kernel *= circular_kernel

    if distance_weighted:
        distance_weighted_kernel = _distance_weighted_kernel_2D(
            radius,
            method,
            decay,
            sigma,
        )
        kernel *= distance_weighted_kernel

    if normalised:
        kernel /= np.sum(kernel)

    return kernel



@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def get_kernel_shift(
    x_offset: float,
    y_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D shift kernel.

    This function returns a kernel that can be used to shift a raster by a fractional
    number of pixels in the x and y directions. The kernel can also be used to simulate
    channel misalignment in image augmentation.

    Parameters
    ----------
    x_offset : float
        The horizontal (x) offset to apply.

    y_offset : float
        The vertical (y) offset to apply.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array contains the (x, y) offsets
        of the kernel values. The second array contains the corresponding weights
        of each kernel value.
    """
    if x_offset == 0.0 and y_offset == 0.0:
        offsets = np.array([[0, 0]], dtype=np.int64)
        weights = np.array([1.0], dtype=np.float32)

        return offsets, weights

    y0 = [int(np.floor(y_offset)), int(np.ceil(y_offset))] if y_offset != 0 else [0, 0]
    x0 = [int(np.floor(x_offset)), int(np.ceil(x_offset))] if x_offset != 0 else [0, 0]

    if x_offset == 0.0 or x_offset % 1 == 0.0:
        offsets = np.zeros((2, 2), dtype=np.int64)
        weights = np.zeros(2, dtype=np.float32)

        offsets[0] = [int(x_offset) if x_offset % 1 == 0.0 else 0, y0[0]]
        offsets[1] = [int(x_offset) if x_offset % 1 == 0.0 else 0, y0[1]]

        weights[0] = y_offset - y0[0]
        weights[1] = 1 - weights[0]

    elif y_offset == 0.0 or y_offset % 1 == 0.0:
        offsets = np.zeros((2, 2), dtype=np.int64)
        weights = np.zeros(2, dtype=np.float32)

        offsets[0] = [x0[0], int(y_offset) if y_offset % 1 == 0.0 else 0]
        offsets[1] = [x0[1], int(y_offset) if y_offset % 1 == 0.0 else 0]

        weights[0] = x_offset - x0[0]
        weights[1] = 1 - weights[0]

    else:
        offsets = np.zeros((4, 2), dtype=np.int64)
        weights = np.zeros(4, dtype=np.float32)

        offsets[0] = [x0[0], y0[0]]
        offsets[1] = [x0[0], y0[1]]
        offsets[2] = [x0[1], y0[0]]
        offsets[3] = [x0[1], y0[1]]

        weights[0] = (1 - (x_offset - offsets[0][0])) * (1 - (y_offset - offsets[0][1]))
        weights[1] = (1 - (x_offset - offsets[1][0])) * (1 + (y_offset - offsets[1][1]))
        weights[2] = (1 + (x_offset - offsets[2][0])) * (1 - (y_offset - offsets[2][1]))
        weights[3] = (1 + (x_offset - offsets[3][0])) * (1 + (y_offset - offsets[3][1]))

    return offsets, weights


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def get_kernel_unsharp(
    radius: float = 1.0,
    intensity: float = 1.0,
) -> np.ndarray:
    """
    Create a 2D unsharp kernel.

    This function returns a kernel that can be used to apply an unsharp mask to an image.

    Parameters
    ----------
    radius : float, optional
        The radius of the kernel. Default: 1.0.

    intensity : float, optional
        The intensity of the unsharp mask. Default: 1.0.

    Returns
    -------
    np.ndarray
        The kernel.
    """
    kernel = get_kernel(
        radius=radius,
        circular=True,
        distance_weighted=True,
        method=3,
        normalised=True,
        hole=True,
    )

    kernel *= intensity
    kernel_sum = np.sum(kernel) + 1.0
    kernel *= -1.0
    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = kernel_sum

    return kernel


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def get_kernel_sobel(
    radius=1,
    scale=2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a 2D Sobel style kernel consisting of a horizontal and vertical component.
    This function returns a kernel that can be used to apply a Sobel filter to an image.

    The kernels for radis=2, scale=2 are:
    ```python
    gx = [
        [ 0.56  0.85  0.   -0.85 -0.56],
        [ 0.85  1.5   0.   -1.5  -0.85],
        [ 1.    2.    0.   -2.   -1.  ],
        [ 0.85  1.5   0.   -1.5  -0.85],
        [ 0.56  0.85  0.   -0.85 -0.56],
    ]

    gy = [
        [ 0.56  0.85  1.    0.85  0.56],
        [ 0.85  1.5   2.    1.5   0.85],
        [ 0.    0.    0.    0.    0.  ],
        [-0.85 -1.5  -2.   -1.5  -0.85],
        [-0.56 -0.85 -1.   -0.85 -0.56],
    ]
    ```

    Parameters
    ----------
    radius : float, optional
        The radius of the kernel. Default: 1.0.

    scale : float, optional
        The scale of the kernel. Default: 2.0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The kernels 
    
    """
    size = np.int64(np.ceil(radius) * 2 + 1)
    kernel_base = np.zeros((size, size), dtype=np.float32)

    center = np.array([0.0, 0.0], dtype=np.float32)

    step = size // 2
    for idx_i, col in enumerate(range(-step, step + 1)):
        for idx_j, row in enumerate(range(-step, step + 1)):
            point = np.array([col, row], dtype=np.float32)
            distance = _distance_2D(center, point)
            if col == 0 and row == 0:
                kernel_base[idx_i, idx_j] = 0
            else:
                weight = np.power((1 - 0.5), distance) * 2
                kernel_base[idx_i, idx_j] = weight * scale

    # vertical
    kernel_gx = kernel_base.copy()
    kernel_gx[:, size // 2:] *= -1
    kernel_gx[:, size // 2] = 0

    # horisontal
    kernel_gy = kernel_base.copy()
    kernel_gy[size // 2:, :] *= -1
    kernel_gy[size // 2, :] = 0

    return kernel_gx, kernel_gy


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def get_offsets_and_weights(
    kernel: np.ndarray,
    remove_zero_weights: bool = True,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generates a list of offsets, weights, and the center pixel index for a given kernel.

    Parameters
    ----------
    kernel : np.ndarray
        The kernel to generate offsets and weights for.

    remove_zero_weights : bool, optional
        Whether to remove offsets and weights with zero weights. Default: True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        The offsets, weights and the center pixel index.
    """
    count = np.count_nonzero(kernel) if remove_zero_weights else kernel.size

    offsets = np.zeros((count, 2), dtype=np.int64)
    weights = np.zeros((count), dtype=np.float32)
    center_idx = kernel.shape[0] // 2

    step = kernel.shape[0] // 2
    index = 0
    for col in range(-step, step + 1):
        for row in range(-step, step + 1):
            if kernel[col + step, row + step] != 0.0:
                offsets[index][0] = col
                offsets[index][1] = row
                weights[index] = kernel[col + step, row + step]

                if col == 0 and row == 0:
                    center_idx = index

                index += 1

    return offsets, weights, center_idx


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _simple_blur_kernel_2d_3x3() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D blur kernel.

    The kernel has the following form:
    ```python
    >>> weights = [
    ...     0.08422299, 0.12822174, 0.08422299,
    ...     0.12822174, 0.15022110, 0.12822174,
    ...     0.08422299, 0.12822174, 0.08422299,
    ... ]
    ```
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array contains the (x, y) offsets
        of the kernel values. The second array contains the corresponding weights
        of each kernel value.
    """
    offsets = np.array([
        [ 1, -1], [ 1, 0], [ 1, 1],
        [ 0, -1], [ 0, 0], [ 0, 1],
        [-1, -1], [-1, 0], [-1, 1],
    ])

    weights = np.array([
        0.08422299, 0.12822174, 0.08422299,
        0.12822174, 0.1502211 , 0.12822174,
        0.08422299, 0.12822174, 0.08422299,
    ], dtype=np.float32)

    return offsets, weights


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _simple_unsharp_kernel_2d_3x3() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D unsharp kernel.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array contains the (x, y) offsets
        of the kernel values. The second array contains the corresponding weights
        of each kernel value.
    """
    offsets = np.array([
        [ 1, -1], [ 1, 0], [ 1, 1],
        [ 0, -1], [ 0, 0], [ 0, 1],
        [-1, -1], [-1, 0], [-1, 1],
    ])

    weights = np.array([
        -0.09911165, -0.15088834, -0.09911165,
        -0.15088834,  2.        , -0.15088834,
        -0.09911165, -0.15088834, -0.09911165,
    ], dtype=np.float32)

    return offsets, weights


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _simple_shift_kernel_2d(
    x_offset: float,
    y_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Create a 2D shift kernel. For augmentations. """
    return get_kernel_shift(x_offset, y_offset)
