
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _distance_2D(p1: np.ndarray, p2: np.ndarray) -> float:
    """ Returns the distance between two points. (2D) """
    d1 = (p1[0] - p2[0]) ** 2
    d2 = (p1[1] - p2[1]) ** 2

    return np.sqrt(d1 + d2)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def get_sobel_kernel(
    radius=1,
    scale=2,
):
    """ Get a sobel kernel of arbitrary size. """
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

    # return (
    #     (gx.astype("float32"), np.array(weights_x, dtype="float32"), np.array(offsets_x, dtype=int)),
    #     (gy.astype("float32"), np.array(weights_y, dtype="float32"), np.array(offsets_y, dtype=int)),
    # )


kernel = np.round(get_sobel_kernel(2, 2), 2)
print(kernel[0])
print()
print(kernel[1])