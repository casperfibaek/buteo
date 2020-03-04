import numpy as np


def generate_get_neighborhood(
    x_max, y_max, kernel, radius, width, weights=False, sort=True
):
    number_of_nonzero = 0
    x_max = x_max - 1
    y_max = y_max - 1

    for x in range(width):
        for y in range(width):
            if kernel[x, y] != 0:
                number_of_nonzero += 1

    offsets = np.empty((number_of_nonzero, 2), dtype="int32")

    if weights is True:
        offsets_weights = np.zeros(number_of_nonzero, dtype=kernel.dtype)
    else:
        offsets_weights = np.ones(number_of_nonzero, dtype="uint8")

    step = 0
    for x in range(width):
        for y in range(width):
            if kernel[x, y] != 0:
                offsets[step] = np.array([x - radius, y - radius], dtype="int32")
                offsets_weights[step] = kernel[x][y]
                step += 1

    def get_neighborhood(x, y, arr):
        _offsets = offsets
        _non_zero = number_of_nonzero
        _x_max = x_max
        _y_max = y_max
        _calc_weights = weights
        _weights = offsets_weights

        _neighborhood = np.empty(_non_zero, dtype=arr.dtype)

        for n in range(_non_zero):
            _offset = _offsets[n]

            offset_x = x + _offset[0]
            offset_y = y + _offset[1]

            if offset_x < 0:
                offset_x = 0
            elif offset_x > _x_max:
                offset_x = _x_max

            if offset_y < 0:
                offset_y = 0
            elif offset_y > _y_max:
                offset_y = _y_max

            _neighborhood[n] = arr[offset_x, offset_y]

        if sort is True:
            _idx = np.argsort(_neighborhood)
            _neighborhood = _neighborhood[_idx]
            _weights = _weights[_idx]

        if _calc_weights is True:
            return (_neighborhood, _weights)
        else:
            return _neighborhood

    return get_neighborhood


def weighted_quantile(values, weights):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = weights[sorter]

    weighted_quantiles = (np.cumsum(sample_weight) - (0.5 * sample_weight)) / np.sum(
        sample_weight
    )

    # return weighted_quantiles
    return weighted_quantiles, np.interp(0.5, weighted_quantiles, values)


def weighted_sum_neighborhood(arr, kernel):
    x_max = arr.shape[0]
    y_max = arr.shape[1]

    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 is not 0
    assert kernel.shape[1] % 2 is not 0

    width = kernel.shape[0]
    radius = width // 2

    get_neighbors = generate_get_neighborhood(
        x_max, y_max, kernel, radius, width, weights=True, sort=False
    )

    result = np.zeros((x_max, y_max), dtype=arr.dtype)

    for x in range(x_max):
        for y in range(y_max):
            neighbors = get_neighbors(x, y, arr)
            result[x, y] = np.multiply(neighbors[0], neighbors[1]).sum()

    return result


if __name__ == "__main__":
    # arr = np.arange(25).reshape(5, 5)
    # kernel = np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1],
    # ])
    # print(arr)
    # sum_neighborhood(arr, kernel)
    # print(arr)

    # bob = weighted_quantile(np.array([1, 2, 3, 4, 5, 6]), np.array([1, 1, 1, 1, 1, 2.5]))
    # print(bob)

    bob = np.array([0.06666667, 0.2, 0.33333333, 0.46666667, 0.6, 0.83333333])
    val = np.array([1, 2, 3, 4, 5, 6])

    def interp(v, w):
        for x in range(len(w)):
            if w[x] >= 0.5:
                if x == 0 or w[x] == 0.5:
                    return v[x]

                top = w[x] - 0.5
                bot = 0.5 - w[x - 1]

                s = top + bot
                top = 1 - top / s
                bot = 1 - bot / s

                return v[x - 1] * bot + v[x] * top

        return w[len(w) - 1]
                

    print(interp(val, bob))
    print(np.interp(0.5, bob, val))
    # bob = weighted_quantile(np.array([3, 1, 8, 4, 5]), np.array([0.1, 0.5, 0.2, 1, 1]))
    # print(bob)
