import numpy as np
from math import floor, sqrt
from shapely.geometry import Point, Polygon

np.set_printoptions(suppress=True)

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


def kernel3d(shape, sigma=1, holed=False, inverted=False, normalised=True, spherical=True, distance_calc="gaussian"):
    kernel = np.zeros(shape, dtype="float32")

    center_z = shape[0] // 2
    center_x = shape[1] // 2
    center_y = shape[2] // 2

    radius_z = shape[0] - center_z if center_z != 0 else 0
    radius_x = shape[1] - center_x if center_x != 0 else 0
    radius_y = shape[2] - center_y if center_y != 0 else 0

    radius = np.linalg.norm(np.array([radius_z, radius_x, radius_y])) + 0.5
    center = np.array([center_z, center_x, center_y], dtype="float32")

    for x in range(center_x + 1):
        for y in range(center_y + 1):
            for z in range(center_z + 1):
                weight = 0
                if distance_calc != "gaussian":
                    weight = np.linalg.norm(np.subtract(np.array([z, x, y]), center))
                else:
                    diff = np.sqrt((x - radius_x) ** 2 + (y - radius_y) ** 2 + (z - radius_z) ** 2)
                    weight = np.exp(-(diff ** 2) / (2 * sigma ** 2))

                if spherical:
                    weight *= cube_sphere_intersection_area(
                        np.array([x, y, z], dtype="float32"),
                        np.array([0, 0, 0], dtype="float32"),
                        radius,
                    )

                if inverted == True: weight = 1 - weight

                kernel[z][x][y] = weight

    kernel[:, :, center_y + 1:] = kernel[:, :, :center_y]
    kernel[:, center_x + 1:, :] = kernel[:, :center_x, :]

    if holed == True: kernel[center_z, center_x, center_y] = 0

    if normalised == True: kernel = np.divide(kernel, kernel.sum())

    import pdb; pdb.set_trace()

    return kernel

def _create_kernel_1d(width, holed=False, normalise=True, inverted=False, offset=1, weighted_distance=True, distance_calc='gaussian', sigma=2, dtype=np.double):
    radius = floor(width / 2) # 4
    kernel = np.zeros((width), dtype=dtype)

    if distance_calc == 'gaussian' and weighted_distance is True:
        for i in range(width):
            diff = np.sqrt((i - radius) ** 2)
            kernel[i] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    else:
        for x in range(width):
            xm = x - radius

            dist = abs(xm)

            weight = 1

            if weighted_distance == True:
                if xm == 0:
                    weight = 1
                else:
                    if distance_calc == 'sqrt':
                        weight = 1 - sqrt(dist / (radius + offset))
                    if distance_calc == 'power':
                        weight = 1 - pow(dist / (radius + offset), 2)
                    if distance_calc == 'linear':
                        weight = 1 - (dist / (radius + offset))

                    if weight < 0: weight = 0

            kernel[x] = weight

    if holed == True:
        kernel[radius] = 0

    if inverted == True:
        for x in range(width):
            kernel[x] = 1 - kernel[x]

    if normalise == True:
        kernel = np.divide(kernel, kernel.sum())

    return kernel


def _create_kernel_2d(width, circular=True, weighted_edges=True, holed=False, offset=1, normalise=True, inverted=False, weighted_distance=True, distance_calc='gaussian', sigma=2, dtype=np.double):
    radius = floor(width / 2) # 4
    kernel = np.zeros((width, width), dtype=dtype)
    pixel_distance = sqrt(0.5)

    if distance_calc == 'gaussian' and weighted_distance is True:
        for x in range(width):
            for y in range(width):
                diff = np.sqrt((x - radius) ** 2 + (y - radius) ** 2)
                kernel[x][y] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    else:
        for x in range(width):
            for y in range(width):
                xm = x - radius
                ym = y - radius

                dist =  np.linalg.norm(np.array([xm, ym]))

                weight = 1

                if weighted_distance == True:
                    if xm == 0 and ym == 0:
                        weight = 1
                    else:
                        if distance_calc == 'sqrt':
                            weight = 1 - sqrt(dist / (radius + offset))
                        if distance_calc == 'power':
                            weight = 1 - pow(dist / (radius + offset), 2)
                        if distance_calc == 'linear':
                            weight = 1 - (dist / (radius + offset))

                        if weight < 0: weight = 0

                kernel[x][y] = weight

    if circular == True:
        for x in range(width):
            for y in range(width):
                xm = x - radius
                ym = y - radius
                
                dist =  np.linalg.norm(np.array([xm, ym]))

                if weighted_edges == False:
                    if dist - radius >= pixel_distance:
                        kernel[x][y] = 0
                else:
                    circle = Point(0, 0).buffer(radius + 0.5)
                    polygon = Polygon([(xm - 0.5, ym - 0.5), (xm - 0.5, ym + 0.5), (xm + 0.5, ym + 0.5), (xm + 0.5, ym - 0.5)])
                    intersection = polygon.intersection(circle)

                    # Area of a pixel is 1, no need to normalise.
                    kernel[x][y] *= intersection.area

    if holed == True:
        kernel[radius][radius] = 0

    if inverted == True:
        for x in range(width):
            for y in range(width):
                kernel[x][y] = 1 - kernel[x][y]

    if normalise == True:
        kernel = np.divide(kernel, kernel.sum())

    return kernel

def _create_kernel_3d(width, circular=True, weighted_edges=True, holed=False, offset=1, normalise=True, inverted=False, weighted_distance=True, distance_calc='gaussian', sigma=2, dim=3, depth=1, dtype=np.double):
    if depth == 1:
        return _create_kernel_2d(width, circular=circular, weighted_edges=weighted_edges, holed=holed, offset=offset, normalise=normalise, inverted=inverted, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma, dtype=dtype)

    radius = width / 2
    radius_pixels = floor(radius) # 4
    kernel = np.zeros((depth, width, width), dtype=dtype)
    pixel_distance = sqrt(0.5)
    center_plane = floor(depth / 2)

    if distance_calc == 'gaussian' and weighted_distance is True:
        for x in range(width):
            for y in range(width):
                for z in range(depth):
                    xm = (x - radius_pixels)
                    if xm < 0: xm = xm + 0.5
                    if xm > 0: xm = xm - 0.5
                        
                    ym = (y - radius_pixels)
                    if ym < 0: ym = ym + 0.5
                    if ym > 0: ym = ym - 0.5

                    zm = (z - radius_pixels)
                    if zm < 0: zm = zm + 0.5
                    if zm > 0: zm = zm - 0.5

                    diff = np.sqrt((x - radius_pixels) ** 2 + (y - radius_pixels) ** 2 + (z - center_plane) ** 2)
                    kernel[z][x][y] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    else:
        for x in range(width):
            for y in range(width):
                for z in range(depth):
                    xm = (x - radius_pixels)
                    if xm < 0: xm = xm + 0.5
                    if xm > 0: xm = xm - 0.5
                        
                    ym = (y - radius_pixels)
                    if ym < 0: ym = ym + 0.5
                    if ym > 0: ym = ym - 0.5

                    zm = (z - radius_pixels)
                    if zm < 0: zm = zm + 0.5
                    if zm > 0: zm = zm - 0.5

                    dist =  np.linalg.norm(np.array([xm, ym, zm]))

                    weight = 1

                    if weighted_distance == True:
                        if xm == 0 and ym == 0 and zm == 0:
                            weight = 1
                        else:
                            if distance_calc == 'sqrt':
                                weight = 1 - sqrt(dist / (radius_pixels + offset))
                            if distance_calc == 'power':
                                weight = 1 - pow(dist / (radius_pixels + offset), 2)
                            if distance_calc == 'linear':
                                weight = 1 - (dist / (center_plane + offset))

                            if weight < 0: weight = 0

                    kernel[z][x][y] = weight

    if circular == True:
        for x in range(width):
            for y in range(width):
                for z in range(depth):
                    xm = (x - radius_pixels)
                    if xm < 0: xm = xm + 0.5
                    if xm > 0: xm = xm - 0.5
                        
                    ym = (y - radius_pixels)
                    if ym < 0: ym = ym + 0.5
                    if ym > 0: ym = ym - 0.5

                    zm = (z - radius_pixels)
                    if zm < 0: zm = zm + 0.5
                    if zm > 0: zm = zm - 0.5
                    
                    dist =  np.linalg.norm(np.array([xm, ym]))

                    if weighted_edges == False:
                        if dist - radius_pixels >= pixel_distance:
                            kernel[z][x][y] = 0
                    else:
                        kernel[z][x][y] *= cube_sphere_intersection_area(
                            np.array([xm, ym, zm], dtype="float32"),
                            np.array([0, 0, 0], dtype="float32"),
                            radius,
                        )

    if holed == True:
        kernel[center_plane][radius_pixels][radius_pixels] = 0

    if inverted == True:
        for x in range(width):
            for y in range(width):
                for z in range(depth):
                    kernel[z][x][y] = 1 - kernel[z][x][y]

    if normalise == True:
        kernel = np.divide(kernel, kernel.sum())
    
    return kernel


def create_kernel(width, circular=True, weighted_edges=True, holed=False, offset=1, normalise=True, inverted=False, weighted_distance=True, distance_calc='gaussian', sigma=2, dim=2, depth=1, dtype=np.double):
    assert(width % 2 != 0), "Kernel width has to be an uneven number."
    assert(dim <= 3), "Function can only handle three of less dimensions"
    assert(depth % 2 != 0), "Depth has to be an uneven number"
 
    if dim == 1:
        return _create_kernel_1d(width, holed=holed, normalise=normalise, inverted=inverted, offset=offset, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma, dtype=dtype)

    if dim == 2:
        return _create_kernel_2d(width, circular=circular, weighted_edges=weighted_edges, holed=holed, offset=offset, normalise=normalise, inverted=inverted, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma, dtype=dtype)
    
    if dim == 3:
        return _create_kernel_3d(width, circular=circular, weighted_edges=weighted_edges, holed=holed, offset=offset, normalise=normalise, inverted=inverted, weighted_distance=weighted_distance, distance_calc=distance_calc, sigma=sigma, dim=dim, depth=depth, dtype=dtype)


if __name__ == "__main__":
    kernel = kernel3d((3, 3, 3))
    print(kernel)
    # import pdb; pdb.set_trace()