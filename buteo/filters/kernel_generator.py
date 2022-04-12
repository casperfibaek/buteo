import numpy as np


def scale_vectors(points, abs_dist):
    scalar = 1 - (abs_dist / np.linalg.norm(points, axis=1))
    scalar = scalar[:, np.newaxis]

    return points * scalar


def points_intersects_ellipsoid(ellipsoid, points, axis=1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(np.sum(np.power(np.divide(points, ellipsoid), 2), axis=axis) <= 1)


# TODO: Switch from brute-force to something more efficient
def cube_sphere_intersection_area(
    cube_center,
    circle_center,
    circle_radius,
    ellipsoid=None,
    cube_width=1,
    resolution=0.05,
    epsilon=1e-7,
):
    assert len(cube_center) == 3, "Cube center must be a len 3 array."
    assert len(circle_center) == 3, "Circle center must be a len 3 array."
    assert circle_radius >= 0, "Radius must be a positive number."
    assert cube_width >= 0, "Width of cube must be a positive number."

    sides = cube_width / 2

    step = resolution
    box_sides = sides * step
    radius_add = 0.5

    # The radius of the encompasing sphere.
    corner_distance = np.linalg.norm(np.array([sides, sides, sides], dtype="float32"))

    dist = np.linalg.norm(np.subtract(circle_center, cube_center))

    if dist > circle_radius + corner_distance:
        return 0.0
    elif dist == 0:
        return 1.0

    z_start = (cube_center[0] - sides) + box_sides
    x_start = (cube_center[1] - sides) + box_sides
    y_start = (cube_center[2] - sides) + box_sides

    z_end = (cube_center[0] + sides) - box_sides
    x_end = (cube_center[1] + sides) - box_sides
    y_end = (cube_center[2] + sides) - box_sides

    z = np.arange(z_start, z_end + epsilon, step)
    x = np.arange(x_start, x_end + epsilon, step)
    y = np.arange(y_start, y_end + epsilon, step)

    zz, xx, yy = np.meshgrid(z, x, y)

    coord_grid = np.zeros((zz.size, 3), dtype="float32")
    coord_grid[:, 0] = zz.ravel()
    coord_grid[:, 1] = xx.ravel()
    coord_grid[:, 2] = yy.ravel()

    if ellipsoid is not None:
        coord_grid_scaled = scale_vectors(coord_grid, radius_add)
        intersections = points_intersects_ellipsoid(ellipsoid, coord_grid_scaled)

        if intersections == 0:
            return 0.0

        volume = cube_width ** 3
        overlap = intersections / zz.size

        return volume * overlap

    dist = np.linalg.norm(np.subtract(circle_center, coord_grid), axis=1)
    return (step * step * step) * np.sum(dist <= circle_radius)


def create_circle_kernel(kernel_size=5, circle_radius=5, remove_zero_weights=True):
    tmp = np.zeros((kernel_size + 1, kernel_size + 1), dtype="float32")
    kernel = np.zeros((kernel_size * 2 + 1, kernel_size * 2 + 1), dtype="float32")

    for x in range(kernel_size + 1):
        for y in range(kernel_size + 1):
            tmp[x][y] = cube_sphere_intersection_area(
                (0, x, y), (0, 0, 0), circle_radius
            )

    tmp = np.fliplr(np.flipud(tmp))
    kernel[: kernel_size + 1, : kernel_size + 1] = tmp
    kernel[: kernel_size + 1, kernel_size:] = np.fliplr(tmp)
    kernel[kernel_size + 1 :, :] = np.flipud(kernel[:kernel_size, :])

    idx_offsets = []
    weights = []
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            current_weight = kernel[x][y]

            if remove_zero_weights and current_weight == 0.0:
                continue

            idx_offsets.append(
                [
                    x - ((kernel_size * 2 + 1) // 2),
                    y - ((kernel_size * 2 + 1) // 2),
                    0,
                ]
            )

            weights.append(current_weight)

    kernel = kernel[:, :, np.newaxis]

    return (
        kernel,
        np.array(idx_offsets, dtype=int),
        np.array(weights, dtype=float),
    )


# TODO: Add fractional kernels
def create_kernel(
    shape,
    sigma=1,
    holed=False,
    inverted=False,
    normalised=True,
    spherical=True,
    edge_weights=True,
    distance_calc="gaussian",
    decay=0.25,
    offsets=False,
    remove_zero_weights=False,
    radius_method="2d",
    output_2d=False,
):
    """Channel last"""

    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)

    assert shape[0] % 2 != 0, "Kernel width has to be an uneven number."
    assert shape[1] % 2 != 0, "Kernel height has to be an uneven number."

    if shape[2] == 1:
        radius_method = "2d"

    kernel = np.zeros(shape, dtype="float32")

    edge_x = shape[0] // 2
    edge_y = shape[1] // 2
    edge_z = shape[2] // 2

    radius = None
    radius_add = np.sqrt(2)
    target = 0.5 if (shape[2] % 2) == 0 else 0.0

    if radius_method == "2d" or edge_z == 0:
        radius = min(edge_x, edge_y) + radius_add
    elif radius_method == "3d":
        radius = min(edge_x, edge_y, edge_z) + radius_add
    elif radius_method == "ellipsoid":
        radius = min(edge_x, edge_y) + radius_add
    else:
        raise ValueError("Unable to parse radius_method. Must be 2d, 3d or ellipsoid")

    for x in range(edge_x + 1):
        for y in range(edge_y + 1):
            for z in range(edge_z + 1):

                weight = 0
                normed = np.linalg.norm(np.array([x - target, y - target, z - target]))

                if distance_calc == "linear":
                    weight = (1 - decay) ** normed
                elif distance_calc == "sqrt":
                    weight = np.sqrt((1 - decay)) ** normed
                elif distance_calc == "power":
                    weight = np.power((1 - decay), 2) ** normed
                elif distance_calc == "gaussian" or distance_calc == True:
                    weight = np.exp(-(normed ** 2) / (2 * sigma ** 2))
                elif distance_calc == False or distance_calc == None:
                    weight = 1
                else:
                    raise ValueError("Unable to parse parameters distance_calc.")

                ellipsoid = None
                if radius_method == "ellipsoid":
                    ellipsoid = np.array([edge_x, edge_y, edge_z], dtype="float32")

                if spherical:
                    adj = cube_sphere_intersection_area(
                        np.array([x, y, z], dtype="float32"),
                        np.array([0, 0, 0], dtype="float32"),
                        radius,
                        ellipsoid=ellipsoid,
                    )

                    if edge_weights:
                        weight *= adj
                    else:
                        if adj == 0:
                            weight = 0

                if inverted:
                    weight = 1 - weight

                kernel[edge_x - x][edge_y - y][edge_z - z] = weight

    z_adjust = 1 if (shape[2] % 2) == 0 else 0

    # We're copying the one quadrant to the other three quadrants
    kernel[edge_x + 1 :, :, :] = np.flip(kernel[:edge_x, :, :], axis=0)
    kernel[:, edge_y + 1 :, :] = np.flip(kernel[:, :edge_y, :], axis=1)
    kernel[:, :, edge_z + (1 - z_adjust) :] = np.flip(kernel[:, :, :edge_z], axis=2)

    if holed:
        kernel[edge_x, edge_y, edge_z] = 0

    if normalised:
        kernel = np.divide(kernel, kernel.sum())

    idx_offsets = []
    weights = []
    if offsets:
        for x in range(kernel.shape[0]):
            for y in range(kernel.shape[1]):
                for z in range(kernel.shape[2]):
                    current_weight = kernel[x][y][z]

                    if remove_zero_weights and current_weight == 0.0:
                        continue

                    if output_2d:
                        idx_offsets.append(
                            [
                                x - (kernel.shape[0] // 2),
                                y - (kernel.shape[1] // 2),
                            ]
                        )
                    else:
                        idx_offsets.append(
                            [
                                x - (kernel.shape[0] // 2),
                                y - (kernel.shape[1] // 2),
                                z - (kernel.shape[2] // 2),
                            ]
                        )
                    weights.append(current_weight)

    if output_2d:
        kernel = kernel[:, :, 0]

    if offsets:
        return (
            kernel,
            np.array(idx_offsets, dtype=int),
            np.array(weights, dtype=float),
        )

    return kernel
