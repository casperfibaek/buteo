import numpy as np


def cube_sphere_intersection_area(cube_center, circle_center, circle_radius, cube_width=1, resolution=0.01, epsilon=1e-7):
    assert(len(cube_center) == 3), "Cube center must be a len 3 array."
    assert(len(circle_center) == 3), "Circle center must be a len 3 array."
    assert(circle_radius >= 0), "Radius must be a positive number."
    assert(cube_width >= 0), "Width of cube must be a positive number."
    
    sides = cube_width / 2

    step = resolution
    box_sides = sides * step

    x_start = (cube_center[0] - sides) + box_sides
    y_start = (cube_center[1] - sides) + box_sides
    z_start = (cube_center[2] - sides) + box_sides

    x_end = (cube_center[0] + sides) - box_sides
    y_end = (cube_center[1] + sides) - box_sides
    z_end = (cube_center[2] + sides) - box_sides

    x = np.arange(x_start, x_end + epsilon, step)
    y = np.arange(y_start, y_end + epsilon, step)
    z = np.arange(z_start, z_end + epsilon, step)

    xx, yy, zz = np.meshgrid(x, y, z)

    coord_grid = np.zeros((xx.size, 3), dtype="float32")
    coord_grid[:, 0] = xx.ravel()
    coord_grid[:, 1] = yy.ravel()
    coord_grid[:, 2] = zz.ravel()

    dist = np.linalg.norm(np.subtract(circle_center, coord_grid), axis=1)

    area = (step * step * step) * np.sum(dist <= circle_radius)

    return area


def create_kernel(shape, sigma=2, holed=False, inverted=False, normalised=True, spherical=True, distance_calc="gaussian"):
    assert(shape[0] % 2 != 0), "Kernel depth has to be an uneven number."
    assert(shape[1] % 2 != 0), "Kernel width has to be an uneven number."
    assert(shape[2] % 2 != 0), "Kernel height has to be an uneven number."

    kernel = np.zeros(shape, dtype="float32")

    center_x = shape[0] // 2
    center_y = shape[1] // 2
    center_z = shape[2] // 2

    radius = np.linalg.norm(np.array([center_x, center_y, center_z]))
    center = np.array([center_x, center_y, center_z], dtype="float32")
    max_distance = np.linalg.norm(np.subtract(np.array([0, 0, 0]), center)) + (np.sqrt(3) - 1)

    for x in range(center_x + 1):
        for y in range(center_y + 1):
            for z in range(center_z + 1):

                weight = 0
                if distance_calc != "gaussian":
                    dist = np.linalg.norm(np.subtract(np.array([x, y, z]), center))
                    weight= 1 - (dist / max_distance)
                else:
                    # if distance_calc == 'sqrt':
                    #     weight = 1 - np.sqrt(dist / radius)
                    # if distance_calc == 'power':
                    #     weight = 1 - np.power(dist / radius), 2)
                    # if distance_calc == 'linear':
                    #     weight = 1 - (dist / radius)

                    diff = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2)
                    weight = np.exp(-(diff ** 2) / (2 * sigma ** 2))

                if spherical:
                    adj = cube_sphere_intersection_area(
                        np.array([x, y, z], dtype="float32"),
                        np.array([center_x, center_y, center_z], dtype="float32"),
                        radius,
                    )

                    weight *= adj

                if inverted == True: weight = 1 - weight

                kernel[x][y][z] = weight

    kernel[center_x + 1:, :, :] = np.flip(kernel[:center_x, :, :], axis=0)
    kernel[:, center_y + 1:, :] = np.flip(kernel[:, :center_y, :], axis=1)
    kernel[:, :, center_z + 1:] = np.flip(kernel[:, :, :center_z], axis=2)

    if holed == True: kernel[center_x, center_y, center_z] = 0

    if normalised == True: kernel = np.divide(kernel, kernel.sum())

    return kernel


def generate_offsets(kernel):
    offsets = []
    weights = []
    for x in range(kernel.shape[0]):
        for y in range(kernel.shape[1]):
            for z in range(kernel.shape[2]):
                offsets.append([x - (kernel.shape[0] // 2), y - (kernel.shape[1] // 2), z - (kernel.shape[2] // 2)])
                weights.append(kernel[x][y][z])
    return (np.array(offsets, dtype=int), np.array(weights, dtype=float))


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    kernel = create_kernel((3, 5, 5), sigma=2, spherical=False)
    print(kernel)
    # import pdb; pdb.set_trace()