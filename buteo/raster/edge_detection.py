"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")

# External
import numpy as np

# Internal
from buteo.raster.convolution import convolve_array, get_kernel


DEFAULT_SPHERICAL = False
DEFAULT_WEIGHT = None
DEFAULT_DECAY = 0.2
DEFAULT_SIGMA = 1.0


def adjust_kernel(
    kernel,
    spherical=DEFAULT_SPHERICAL,
    distance_weight=DEFAULT_WEIGHT,
    distance_decay=DEFAULT_DECAY,
    distance_sigma=DEFAULT_SIGMA,
):
    """ Adjust an edge kernel by a normal kernel. Only works for symmetric kernels."""
    normal_kernel, _weights, _offsets = get_kernel(
        kernel.shape[0],
        depth=1,
        normalise=False,
        spherical=spherical,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
    )

    if spherical is True or distance_weight is not None:
        kernel_pre = np.abs(kernel).sum()

        kernel = kernel * normal_kernel

        kernel_post = np.abs(kernel).sum()

        kernel = kernel * (kernel_pre / kernel_post)

    return kernel


def get_sobel_kernel(
    size=3,
    scale=1,
    spherical=DEFAULT_SPHERICAL,
    distance_weight=DEFAULT_WEIGHT,
    distance_decay=DEFAULT_DECAY,
    distance_sigma=DEFAULT_SIGMA,
):
    """ Get a sobel kernel of arbitrary size. """

    assert size % 2 != 0

    gx = np.zeros((size, size, 1), dtype=np.float32)
    gy = np.zeros((size, size, 1), dtype=np.float32)

    indices = np.indices((size, size, 1), dtype=np.float32)
    cols = indices[0] - (size // 2)
    rows = indices[1] - (size // 2)

    squared = np.power(cols, 2) + np.power(rows, 2)
    np.divide(cols, squared, out=gy, where=squared!=0) # in-place
    np.divide(rows, squared, out=gx, where=squared!=0) # in-place

    gx = gx * scale
    gy = gy * scale

    gx = adjust_kernel(
        gx,
        spherical=spherical,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
    )
    gy = adjust_kernel(
        gy,
        spherical=spherical,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
    )

    weights_x = []
    weights_y = []

    offsets_x = []
    offsets_y = []

    for idx_x in range(0, gx.shape[0]):
        for idx_y in range(0, gx.shape[1]):
            for idx_z in range(0, gx.shape[2]):
                current_weight_x = gx[idx_x][idx_y][idx_z]
                current_weight_y = gy[idx_x][idx_y][idx_z]

                if current_weight_x != 0.0:
                    offsets_x.append(
                        [
                            idx_x - (gx.shape[0] // 2),
                            idx_y - (gx.shape[1] // 2),
                            idx_z,
                        ]
                    )

                    weights_x.append(current_weight_x)

                if current_weight_y != 0.0:
                    offsets_y.append(
                        [
                            idx_x - (gy.shape[0] // 2),
                            idx_y - (gy.shape[1] // 2),
                            idx_z,
                        ]
                    )

                    weights_y.append(current_weight_y)

    return (
        (gx.astype("float32"), np.array(weights_x, dtype="float32"), np.array(offsets_x, dtype=int)),
        (gy.astype("float32"), np.array(weights_y, dtype="float32"), np.array(offsets_y, dtype=int)),
    )


def get_roberts_prewitt_kernel(
    roberts=True,
    spherical=DEFAULT_SPHERICAL,
    distance_weight=DEFAULT_WEIGHT,
    distance_decay=DEFAULT_DECAY,
    distance_sigma=DEFAULT_SIGMA,
):
    """ Get a roberts cross or prewitt kernel, only available for size 2x2 and 3x3 """

    if roberts:
        gx = np.array([
            [ 0, 0, 0 ],
            [ 0, 1, 0 ],
            [ 0, 0,-1 ],
        ])[:, :, np.newaxis]

        gy = np.array([
            [ 0, 0, 0 ],
            [ 0, 0, 1 ],
            [ 0,-1, 0 ],
        ])[:, :, np.newaxis]
    else:
        gx = np.array([
            [-1, 0, 1 ],
            [-1, 0, 1 ],
            [-1, 0, 1 ],
        ])[:, :, np.newaxis]

        gy = np.array([
            [ 1, 1, 1 ],
            [ 0, 0, 0 ],
            [-1,-1,-1 ],
        ])[:, :, np.newaxis]

    gx = adjust_kernel(
        gx,
        spherical=spherical,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
    )
    gy = adjust_kernel(
        gy,
        spherical=spherical,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
    )

    weights_x = []
    weights_y = []

    offsets_x = []
    offsets_y = []

    for idx_x in range(0, gx.shape[0]):
        for idx_y in range(0, gx.shape[1]):
            for idx_z in range(0, gx.shape[2]):
                current_weight_x = gx[idx_x][idx_y][idx_z]
                current_weight_y = gy[idx_x][idx_y][idx_z]

                if current_weight_x != 0.0:
                    offsets_x.append(
                        [
                            idx_x - (gx.shape[0] // 2),
                            idx_y - (gx.shape[1] // 2),
                            idx_z,
                        ]
                    )

                    weights_x.append(current_weight_x)

                if current_weight_y != 0.0:
                    offsets_y.append(
                        [
                            idx_x - (gy.shape[0] // 2),
                            idx_y - (gy.shape[1] // 2),
                            idx_z,
                        ]
                    )

                    weights_y.append(current_weight_y)

    return (
        (gx.astype("float32"), np.array(weights_x, dtype="float32"), np.array(offsets_x, dtype=int)),
        (gy.astype("float32"), np.array(weights_y, dtype="float32"), np.array(offsets_y, dtype=int)),
    )


def edge_detection(
    arr,
    method,
    filter_size=3,
    spherical=DEFAULT_SPHERICAL,
    distance_weight=DEFAULT_WEIGHT,
    distance_decay=DEFAULT_DECAY,
    distance_sigma=DEFAULT_SIGMA,
    scale=1.0,
    merge_results=True,
    gradient_output=False,
    nodata=False,
    nodata_value=-9999.9,
):
    """ Perform an detection method. """

    if method == "sobel":
        gx, gy = get_sobel_kernel(
            size=filter_size,
            scale=scale,
            spherical=spherical,
            distance_weight=distance_weight,
            distance_decay=distance_decay,
            distance_sigma=distance_sigma,
        )
    elif method == "roberts":
        gx, gy = get_roberts_prewitt_kernel(
            roberts=True,
            spherical=spherical,
            distance_weight=distance_weight,
            distance_decay=distance_decay,
            distance_sigma=distance_sigma,
        )
    elif method == "prewitt":
        gx, gy = get_roberts_prewitt_kernel(
            roberts=False,
            spherical=spherical,
            distance_weight=distance_weight,
            distance_decay=distance_decay,
            distance_sigma=distance_sigma,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    _kernel_x, weights_x, offsets_x = gx
    _kernel_y, weights_y, offsets_y = gy

    pad_size = filter_size // 2
    gx = convolve_array(arr, offsets=offsets_x, weights=weights_x, method="sum", nodata=nodata, nodata_value=nodata_value, pad=True, pad_size=pad_size)
    gy = convolve_array(arr, offsets=offsets_y, weights=weights_y, method="sum", nodata=nodata, nodata_value=nodata_value, pad=True, pad_size=pad_size)

    if gradient_output:
        grad_mag = np.sqrt(np.add(np.power(gx, 2), np.power(gy, 2)))
        grad_dir = np.arctan(np.true_divide(gx, gy, where=gy!=0))

        return grad_mag, grad_dir

    if merge_results:
        return np.sqrt(np.add(np.power(gx, 2), np.power(gy, 2)))

    return gx, gy


def edge_detection_sobel(arr, filter_size=3, spherical=DEFAULT_SPHERICAL, distance_weight=DEFAULT_WEIGHT, distance_decay=DEFAULT_DECAY, distance_sigma=DEFAULT_SIGMA, scale=1.0, merge_results=True, gradient_output=False, nodata=False, nodata_value=-9999.9):
    """ Detect edges using the sobel filtering method. """

    return edge_detection(
        arr,
        method="sobel",
        filter_size=filter_size,
        spherical=spherical,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        scale=scale,
        merge_results=merge_results,
        gradient_output=gradient_output,
        nodata=nodata,
        nodata_value=nodata_value,
    )


def edge_detection_roberts(arr, spherical=DEFAULT_SPHERICAL, distance_weight=DEFAULT_WEIGHT, distance_decay=DEFAULT_DECAY, distance_sigma=DEFAULT_SIGMA, merge_results=True, gradient_output=False, nodata=False, nodata_value=-9999.9):
    """ Detect edges using the roberts filtering method. """

    return edge_detection(
        arr,
        method="roberts",
        spherical=spherical,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        merge_results=merge_results,
        gradient_output=gradient_output,
        nodata=nodata,
        nodata_value=nodata_value,
    )


def edge_detection_prewitt(arr, spherical=DEFAULT_SPHERICAL, distance_weight=DEFAULT_WEIGHT, distance_decay=DEFAULT_DECAY, distance_sigma=DEFAULT_SIGMA, merge_results=True, gradient_output=False, nodata=False, nodata_value=-9999.9):
    """ Detect edges using the prewitt filtering method. """

    return edge_detection(
        arr,
        method="prewitt",
        spherical=spherical,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        merge_results=merge_results,
        gradient_output=gradient_output,
        nodata=nodata,
        nodata_value=nodata_value,
    )
