import numpy as np
from math import floor, sqrt
from shapely.geometry import Point, Polygon


def create_kernel(width, circular=True, weighted_edges=True, holed=False, normalise=True, inverted=False, weighted_distance=True, distance_calc='guassian', sigma=1, plot=False, dtype='float32'):
    assert(width % 2 != 0)

    radius = floor(width / 2) # 4
    kernel = np.zeros((width, width), dtype=dtype)
    pixel_distance = sqrt(0.5)

    if distance_calc == 'guassian':
        for i in range(width):
            for j in range(width):
                diff = np.sqrt((i - radius) ** 2 + (j - radius) ** 2)
                kernel[i][j] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
    else:
        for x in range(width):
            for y in range(width):
                xm = x - radius
                ym = y - radius

                dist = sqrt(pow(xm, 2) + pow(ym, 2))

                weight = 1

                if weighted_distance == True:
                    if xm == 0 and ym == 0:
                        weight = 1
                    else:
                        scale = sqrt((radius ** 2) + (radius ** 2)) + pixel_distance
                        if distance_calc == 'sqrt':
                            weight = 1 - (sqrt(dist) / sqrt(scale))
                        if distance_calc == 'power':
                            weight = 1 - (pow(dist, 2) / pow(scale, 2))
                        if distance_calc == 'linear':
                            weight = 1 - (dist / scale)

                kernel[x][y] = weight

    if circular == True:
        for x in range(width):
            for y in range(width):
                xm = x - radius
                ym = y - radius

                if weighted_edges == False:
                    if (dist - radius) >= pixel_distance:
                        weight = 0
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