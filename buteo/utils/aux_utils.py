"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""

# External
import numpy as np
from numba import jit, prange



@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latitude(lat):
    """ Latitude goes from -90 to 90 """
    lat_adj = lat + 90.0
    lat_max = 180

    encoded_sin = ((np.sin(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_longitude(lng):
    """ Longitude goes from -180 to 180 """
    lng_adj = lng + 180.0
    lng_max = 360

    encoded_sin = ((np.sin(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latlng(latlng):
    """
    Encode latitude and longitude values to be used as input to the model.
    """
    lat = latlng[0]
    lng = latlng[1]

    encoded_lat = encode_latitude(lat)
    encoded_lng = encode_longitude(lng)

    return np.concatenate((encoded_lat, encoded_lng)).astype(np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latlngs(latlngs):
    """ Encode multiple latitude and longitude values. """
    if latlngs.ndim == 1:
        encoded_latlngs = np.apply_along_axis(encode_latlng, 0, latlngs)
    elif latlngs.ndim == 2:
        encoded_latlngs = np.apply_along_axis(encode_latlng, 1, latlngs)
    elif latlngs.ndim == 3:
        rows = latlngs.shape[0]
        cols = latlngs.shape[1]

        output_shape = (rows, cols, 4)
        encoded_latlngs = np.zeros(output_shape, dtype=np.float32)

        for i in prange(rows):
            for j in range(cols):
                latlng = latlngs[i, j]
                encoded_latlngs[i, j] = encode_latlng(latlng)
    else:
        raise ValueError(
            f"The input array must have 1, 2 or 3 dimensions, not {latlngs.ndim}"
        )

    return encoded_latlngs


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_latitude(encoded_sin, encoded_cos):
    """
    Decode encoded latitude values to the original latitude value.
    """
    lat_max = 180
    lat_max_half = lat_max / 2.0

    # Calculate the sin and cos values from the encoded values
    sin_val = (2 * encoded_sin) - 1
    cos_val = (2 * encoded_cos) - 1

    # Calculate the latitude adjustment
    lat_adj = np.arctan2(sin_val, cos_val)

    # Convert the adjusted latitude to the original latitude value
    sign = np.sign(lat_adj)
    sign_adj = np.where(sign == 0, 1, sign) * lat_max_half

    lat = ((lat_adj / (2 * np.pi)) * lat_max) - sign_adj
    lat = np.where(lat == -lat_max_half, lat_max_half, lat)

    return lat

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_longitude(encoded_sin, encoded_cos):
    """
    Decode encoded longitude values to the original longitude value.
    """
    lng_max = 360
    lng_max_half = lng_max / 2.0

    # Calculate the sin and cos values from the encoded values
    sin_val = (2 * encoded_sin) - 1
    cos_val = (2 * encoded_cos) - 1

    # Calculate the longitude adjustment
    lng_adj = np.arctan2(sin_val, cos_val)

    # Convert the adjusted longitude to the original longitude value
    sign = np.sign(lng_adj)
    sign_adj = np.where(sign == 0, 1, sign) * lng_max_half

    lng = ((lng_adj / (2 * np.pi)) * lng_max) - sign_adj

    lng = np.where(lng == -lng_max_half, lng_max_half, lng)

    return lng


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_latlng(encoded_latlng):
    """
    Decode encoded latitude and longitude values to the original values.
    """
    lat = decode_latitude(encoded_latlng[0], encoded_latlng[1])
    lng = decode_longitude(encoded_latlng[2], encoded_latlng[3])

    return np.array([lat, lng], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_latlngs(encoded_latlngs):
    """ Decode multiple latitude and longitude values. """
    latlngs = np.apply_along_axis(decode_latlng, 1, encoded_latlngs)
    return latlngs

def channel_first_to_last(arr):
    """ Converts a numpy array from channel first to channel last format. """
    if arr.ndim != 3:
        raise ValueError("Input array should be 3-dimensional with shape (channels, height, width)")

    # Swap the axes to change from channel first to channel last format
    arr = np.transpose(arr, (1, 2, 0))

    return arr

def channel_last_to_first(arr):
    """ Converts a numpy array from channel last to channel first format. """
    if arr.ndim != 3:
        raise ValueError("Input array should be 3-dimensional with shape (height, width, channels)")

    # Swap the axes to change from channel last to channel first format
    arr = np.transpose(arr, (2, 0, 1))

    return arr

def scale_to_range(arr, min_val, max_val):
    """ Scales the values in the input array to the specified range. """

    # Scale the values in the array to the specified range
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (max_val - min_val) * arr + min_val

    return arr

@jit(nopython=True)
def create_grid(range_rows, range_cols):
    """ Create a grid of rows and columns """
    rows_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)
    cols_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)

    for i in range(len(range_rows)):
        for j in range(len(range_cols)):
            cols_grid[i, j] = range_rows[j]
            rows_grid[i, j] = range_cols[i]

    return rows_grid, cols_grid


# def split_into_offsets(shape, offsets_x=2, offsets_y=2, overlap_x=0, overlap_y=0):
#     """ Split a shape into offsets. Usually used for splitting an image into offsets to reduce RAM needed. """
#     width = shape[0]
#     height = shape[1]

#     x_remainder = width % offsets_x
#     y_remainder = height % offsets_y

#     x_offsets = [0]
#     x_sizes = []
#     for _ in range((offsets_x - 1)):
#         x_offsets.append(x_offsets[-1] + (width // offsets_x))
#     x_offsets[-1] -= x_remainder

#     for idx, _ in enumerate(x_offsets):
#         if idx == len(x_offsets) - 1:
#             x_sizes.append(width - x_offsets[idx])
#         elif idx == 0:
#             x_sizes.append(x_offsets[1])
#         else:
#             x_sizes.append(x_offsets[idx + 1] - x_offsets[idx])

#     y_offsets = [0]
#     y_sizes = []
#     for _ in range((offsets_y - 1)):
#         y_offsets.append(y_offsets[-1] + (height // offsets_y))
#     y_offsets[-1] -= y_remainder

#     for idx, _ in enumerate(y_offsets):
#         if idx == len(y_offsets) - 1:
#             y_sizes.append(height - y_offsets[idx])
#         elif idx == 0:
#             y_sizes.append(y_offsets[1])
#         else:
#             y_sizes.append(y_offsets[idx + 1] - y_offsets[idx])

#     offsets = []

#     for idx_col, _ in enumerate(y_offsets):
#         for idx_row, _ in enumerate(x_offsets):
#             offsets.append([
#                 x_offsets[idx_row],
#                 y_offsets[idx_col],
#                 x_sizes[idx_row],
#                 y_sizes[idx_col],
#             ])

#     return offsets


def split_into_offsets(shape, offsets_x=2, offsets_y=2, overlap_x=0, overlap_y=0):
    """ Split a shape into offsets. Usually used for splitting an image into offsets to reduce RAM needed. """
    width = shape[0]
    height = shape[1]

    x_remainder = width % offsets_x
    y_remainder = height % offsets_y

    x_offsets = [0]
    x_sizes = []
    for _ in range(offsets_x - 1):
        x_offsets.append(x_offsets[-1] + (width // offsets_x) - overlap_x)
    x_offsets[-1] -= x_remainder

    for idx, _ in enumerate(x_offsets):
        if idx == len(x_offsets) - 1:
            x_sizes.append(width - x_offsets[idx])
        elif idx == 0:
            x_sizes.append(x_offsets[1] + overlap_x)
        else:
            x_sizes.append(x_offsets[idx + 1] - x_offsets[idx] + overlap_x)

    y_offsets = [0]
    y_sizes = []
    for _ in range(offsets_y - 1):
        y_offsets.append(y_offsets[-1] + (height // offsets_y) - overlap_y)
    y_offsets[-1] -= y_remainder

    for idx, _ in enumerate(y_offsets):
        if idx == len(y_offsets) - 1:
            y_sizes.append(height - y_offsets[idx])
        elif idx == 0:
            y_sizes.append(y_offsets[1] + overlap_y)
        else:
            y_sizes.append(y_offsets[idx + 1] - y_offsets[idx] + overlap_y)

    offsets = []

    for idx_col, _ in enumerate(y_offsets):
        for idx_row, _ in enumerate(x_offsets):
            offsets.append([
                x_offsets[idx_row],
                y_offsets[idx_col],
                x_sizes[idx_row],
                y_sizes[idx_col],
            ])

    return offsets


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def calculate_pixel_distances(array, target=1, maximum_distance=None, pixel_width=1, pixel_height=1):
    """ Calculate the distance from each pixel to the nearest target pixel. """
    binary_array = np.sum(array == target, axis=2, dtype=np.uint8)

    if maximum_distance is None:
        maximum_distance = np.sqrt(binary_array.shape[0] ** 2 + binary_array.shape[1] ** 2)

    radius_cols = int(np.ceil(maximum_distance / pixel_height))
    radius_rows = int(np.ceil(maximum_distance / pixel_width))

    kernel_cols = radius_cols * 2
    kernel_rows = radius_rows * 2

    if kernel_cols % 2 == 0:
        kernel_cols += 1

    if kernel_rows % 2 == 0:
        kernel_rows += 1

    middle_cols = int(np.floor(kernel_cols / 2))
    middle_rows = int(np.floor(kernel_rows / 2))

    range_cols = np.arange(-middle_cols, middle_cols + 1)
    range_rows = np.arange(-middle_rows, middle_rows + 1)

    cols_grid, rows_grid = create_grid(range_rows, range_cols)
    coord_grid = np.empty((cols_grid.size, 2), dtype=np.int64)
    coord_grid[:, 0] = cols_grid.flatten()
    coord_grid[:, 1] = rows_grid.flatten()

    coord_grid_projected = np.empty_like(coord_grid, dtype=np.float32)
    coord_grid_projected[:, 0] = coord_grid[:, 0] * pixel_height
    coord_grid_projected[:, 1] = coord_grid[:, 1] * pixel_width

    coord_grid_values = np.sqrt((coord_grid_projected[:, 0] ** 2) + (coord_grid_projected[:, 1] ** 2))

    selected_range = np.arange(coord_grid.shape[0])
    selected_range = selected_range[np.argsort(coord_grid_values)][1:]
    selected_range = selected_range[coord_grid_values[selected_range] <= maximum_distance]

    coord_grid = coord_grid[selected_range]
    coord_grid_values = coord_grid_values[selected_range]

    distances = np.full_like(binary_array, maximum_distance, dtype=np.float32)
    for col in prange(binary_array.shape[0]):
        for row in range(binary_array.shape[1]):
            if binary_array[col, row] == target:
                distances[col, row] = 0
            else:
                for idx, (col_adj, row_adj) in enumerate(coord_grid):
                    if (col + col_adj) >= 0 and (col + col_adj) < binary_array.shape[0] and \
                        (row + row_adj) >= 0 and (row + row_adj) < binary_array.shape[1] and \
                        binary_array[col + col_adj, row + row_adj] == target:

                        distances[col, row] = coord_grid_values[idx]
                        break

    return np.expand_dims(distances, axis=2)

@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def fill_nodata_with_nearest_average(array, nodata_value, mask=None, max_iterations=None, channel=0):
    """ Calculate the distance from each pixel to the nearest target pixel. """
    kernel_size = 3

    range_rows = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
    range_cols = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)

    cols_grid, rows_grid = create_grid(range_rows, range_cols)
    coord_grid = np.empty((cols_grid.size, 2), dtype=np.int64)
    coord_grid[:, 0] = cols_grid.flatten()
    coord_grid[:, 1] = rows_grid.flatten()

    coord_grid_values = np.sqrt((coord_grid[:, 0] ** 2) + (coord_grid[:, 1] ** 2))
    coord_grid_values_sort = np.argsort(coord_grid_values)[1:]
    coord_grid_values = coord_grid_values[coord_grid_values_sort]

    coord_grid = coord_grid[coord_grid_values_sort]

    weights = 1 / coord_grid_values
    weights = weights / np.sum(weights)
    weights = weights.astype(np.float32)

    main_filled = np.copy(array)

    if mask is None:
        mask = np.ones_like(main_filled, dtype=np.uint8)
    else:
        mask = (mask == 1).astype(np.uint8)

    main_filled = main_filled[:, :, channel]
    mask = mask[:, :, channel]

    nodata_value = np.array(nodata_value, dtype=array.dtype)
    uint8_1 = np.array(1, dtype=np.uint8)

    iterations = 0
    while True:
        local_filled = np.copy(main_filled)
        for row in prange(main_filled.shape[0]):
            for col in prange(main_filled.shape[1]):
                if main_filled[row, col] != nodata_value:
                    continue
                if mask[row, col] != uint8_1:
                    continue

                count = 0
                weights_sum = 0.0
                value_sum = 0.0

                for idx, (col_adj, row_adj) in enumerate(coord_grid):
                    if (row + row_adj) >= 0 and (row + row_adj) < main_filled.shape[0] and \
                        (col + col_adj) >= 0 and (col + col_adj) < main_filled.shape[1] and \
                        main_filled[row + row_adj, col + col_adj] != nodata_value and \
                        mask[row + row_adj, col + col_adj] == uint8_1:

                        weight = weights[idx]
                        value = main_filled[row + row_adj, col + col_adj]

                        value_sum += value * weight
                        weights_sum += weight
                        count += 1

                if count == 0:
                    local_filled[row, col] = nodata_value
                else:
                    local_filled[row, col] = value_sum * (1.0 / weights_sum)

        main_filled = local_filled
        iterations += 1

        if max_iterations is not None and iterations >= max_iterations:
            break

        if np.sum((main_filled == nodata_value) & (mask == uint8_1)) == 0:
            break

    return np.expand_dims(main_filled, axis=2)
