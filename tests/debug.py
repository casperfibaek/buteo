""" This is a debug script, used for ad-hoc testing. """

# Standard library
import sys; sys.path.append("../")

from typing import Optional, Tuple
import numpy as np


def weight_distance(
    arr: np.ndarray,
    method: Optional[str] = None,
    decay: float = 0.2,
    sigma: float = 1.0,
    center: float = 0.0,
    spherical: bool = False,
    radius: float = 3.0,
) -> float:
    """
    Weights the kernel by distance using various methods.

    Args:
        arr (numpy.ndarray): The input array.
        method (str, default=None): The weighting method to use.
            "none": No weighting (default).
            "linear": Linear decay.
            "sqrt": Square root decay.
            "power": Power decay.
            "log": Logarithmic decay.
            "gaussian": Gaussian decay.
        decay (float, default=0.2): The decay rate for the `linear`, `sqrt`, and `power` methods.
        sigma (float, default=1.0): The standard deviation for the Gaussian method.
        center (float, default=0.0): The center of the array.
        spherical (bool, default=False): If True, adjust weights based on the radius.
        radius (float, default=3.0): The radius for spherical adjustments.

    Returns:
        float: The computed weight.
    """

    if center == 0.0:
        normed = np.linalg.norm(arr)
    else:
        normed = np.linalg.norm(arr - np.array([0, 0, center]))

    if normed == 0.0:
        weights = 1.0

    if method is None or method == "none" or method == "":
        weights = 1.0
    elif method == "linear":
        weights = np.power((1 - decay), normed)
    elif method == "sqrt":
        weights = np.power(np.sqrt((1 - decay)), normed)
    elif method == "power":
        weights = np.power(np.power((1 - decay), 2), normed)
    elif method == "log":
        weights = np.log(normed + 2) # +2 to avoid log(0)
    elif method == "gaussian":
        weights = np.exp(-(np.power(normed, 2)) / (2 * np.power(sigma, 2)))
    else:
        raise ValueError("Unable to parse parameters for weight_distance.")

    if spherical:
        sqrt_2 = np.sqrt(2)
        half_sqrt_2 = np.divide(sqrt_2, 2)

        if normed > radius + half_sqrt_2:
            return 0.0
        elif normed < radius - half_sqrt_2:
            return 1.0

        dist_min = radius - half_sqrt_2
        dist_max = radius + half_sqrt_2

        normed = 1 - np.interp(normed, [dist_min, dist_max], [0, 1])
        return weights * normed

    return weights


def rotate_kernel(bottom_right: np.ndarray) -> np.ndarray:
    """
    Creates a whole kernel from a quadrant.

    Args:
        bottom_right (numpy.ndarray): The bottom-right quadrant of the kernel.

    Returns:
        numpy.ndarray: The complete kernel generated from the given quadrant.
    """
    size = ((bottom_right.shape[0] - 1) * 2) + 1
    depth = bottom_right.shape[2]
    kernel = np.zeros((size, size, depth), dtype="float32")

    top_right = np.flipud(bottom_right)
    lower_left = np.fliplr(bottom_right)
    top_left = np.flipud(lower_left)

    kernel[size // 2:, size // 2:, :] = bottom_right
    kernel[:1 + -size // 2, :1 + -size // 2, :] = top_left
    kernel[1 + size // 2:, :size // 2, :] = lower_left[1:, :-1, :]
    kernel[:size // 2, 1 + size // 2:, :] = top_right[:-1, 1:, :]

    return kernel


def get_kernel(
    size: int,
    depth: int = 1,
    hole: bool = False,
    inverted: bool = False,
    normalise: bool = True,
    multi_dimensional: bool = False,
    multi_dimensional_center: int = 0,
    spherical: bool = False,
    distance_weight: Optional[str] = None,
    distance_decay: float = 0.2,
    distance_sigma: float = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a square kernel for convolutions.

    Args:
        size (int): Size of the kernel (must be odd).
        depth (int, default=1): Depth of the kernel.
        hole (bool, default=False): Create a hole in the center of the kernel.
        inverted (bool, default=False): Invert the kernel values.
        normalise (bool, default=True): Normalize the kernel values.
        multi_dimensional (bool, default=False): Consider the kernel multi-dimensional.
        multi_dimensional_center (int, default=0): Center of the
            multi-dimensional kernel.
        spherical (bool, default=False): Consider the kernel spherical.
        distance_weight (str or None, default=None): Distance weighting method.
        distance_decay (float, default=0.2): Distance decay factor.
        distance_sigma (float, default=1): Distance sigma for Gaussian distance weighting.

    Returns:
        tuple: A tuple containing the kernel, weights, and offsets.
    """
    assert size >= 3, "Kernel must have atleast size 3."
    assert size % 2 != 0, "Kernel must be an uneven size."
    assert isinstance(size, int), "Kernel must be an integer."
    assert depth >= 1, "Depth must be a positive integer"
    assert isinstance(depth, int), "Depth must be an integer."

    if distance_weight is False:
        distance_weight = None

    quadrant = np.zeros((1 + size // 2, 1 + size // 2, depth), dtype="float32")

    for idx_x in range(0, quadrant.shape[0]):
        for idx_y in range(0, quadrant.shape[1]):
            for idx_z in range(0, quadrant.shape[2]):

                z_value = idx_z if multi_dimensional else 0

                weighted = weight_distance(
                    np.array([idx_x, idx_y, z_value], dtype="float32"),
                    method=distance_weight,
                    decay=distance_decay,
                    sigma=distance_sigma,
                    center=multi_dimensional_center,
                    spherical=spherical,
                    radius=size / 2,
                )

                quadrant[idx_x, idx_y, idx_z] = weighted

    if hole:
        for idx_z in range(0, quadrant.shape[2]):
            quadrant[0, 0, idx_z] = 0

    kernel = rotate_kernel(quadrant)

    if distance_weight == "log":
        kernel = kernel.max() - kernel

    if inverted:
        kernel = 1 - kernel

    if normalise:
        if multi_dimensional:
            summed = kernel.sum()
            if summed != 0.0:
                kernel = kernel / summed
        else:
            summed = kernel.sum(axis=(0, 1))

            for dim in range(0, depth):
                kernel[:, :, dim] = kernel[:, :, dim] / summed[dim]

    weights = []
    offsets = []

    for idx_x in range(0, kernel.shape[0]):
        for idx_y in range(0, kernel.shape[1]):
            for idx_z in range(0, kernel.shape[2]):
                current_weight = kernel[idx_x][idx_y][idx_z]

                if current_weight <= 0.0:
                    continue

                offsets.append(
                    [
                        idx_x - (kernel.shape[0] // 2),
                        idx_y - (kernel.shape[1] // 2),
                        idx_z
                    ]
                )

                weights.append(current_weight)

    return kernel, np.array(weights, dtype="float32"), np.array(offsets, dtype=int)
