""" ### Time series analysis functions. ###"""

# External
import numpy as np



def timeseries_least_square_slope(arr):
    """Compute the least squares slope for a set of data points along the last channel.
    
    Parameters
    ----------
    arr : numpy.ndarray
        A 3D array with shape (height, width, time_steps)
    
    Returns
    -------
    numpy.ndarray
        Array of slopes with shape (height, width, 1)
        
    Notes
    -----
    This function calculates the slope of a linear trend for each pixel
    in the array along the time dimension (last axis). It uses a standard
    least squares method which is sensitive to outliers.
    
    For a more robust approach that handles outliers better, use
    timeseries_robust_least_squares_slope instead.
    """
    assert arr.ndim == 3, "Input array must be 3D"
    assert arr.shape[0] > 0 and arr.shape[1] > 0, "Height and width dimensions must be non-empty"
    assert arr.shape[2] >= 1, "Time dimension must have at least 1 point"

    y = arr
    x_range = np.arange(y.shape[-1], dtype=np.float32)
    x = np.tile(x_range, (y.shape[0], y.shape[1], 1))

    # Calculate the means
    x_mean = np.mean(x, axis=2, keepdims=True)
    y_mean = np.mean(y, axis=2, keepdims=True)

    # Compute the least squares slope using vectorized operations
    numerator = np.sum((x - x_mean) * (y - y_mean), axis=2, keepdims=True)
    denominator = np.sum((x - x_mean) ** 2, axis=2, keepdims=True)
    slope = numerator / denominator

    return slope


def timeseries_robust_least_squares_slope(arr, std_threshold=1.0, splits=10, report_progress=True):
    """Compute the robust least squares slope for a set of data points along the last channel.
    
    Parameters
    ----------
    arr : numpy.ndarray
        A 3D array with shape (height, width, time_steps)
    
    std_threshold : float, optional
        The threshold in standard deviations to use for outlier detection.
        Higher values include more points, lower values are more restrictive.
        Default: 1.0
        
    splits : int, optional
        Number of parts to split the array into for processing.
        Must be less than or equal to the height of the array.
        Default: 10
        
    report_progress : bool, optional
        Whether to print progress information.
        Default: True
    
    Returns
    -------
    numpy.ndarray
        Array of slopes with shape (height, width, 1)
    """
    assert arr.ndim == 3, "Input array must be 3D"
    assert arr.shape[0] >= splits, "Input array must have at least as many rows as splits"
    y_list = np.array_split(arr, splits, axis=0)
    slope_list = []

    for idx, y in enumerate(y_list):
        if report_progress:
            print(f"Processing split {idx + 1}/{splits}...")

        x_range = np.arange(y.shape[-1], dtype=np.float32)
        x = np.tile(x_range, (y.shape[0], y.shape[1], 1))

        n = x.shape[-1]

        # Calculate the slope between every pair of points
        i, j = np.triu_indices(n, k=1)
        slopes = np.empty((y.shape[0], y.shape[1], (n * (n - 1)) // 2), dtype=np.float32)
        slopes = (y[..., j] - y[..., i]) / (x[..., j] - x[..., i])

        # Calculate the median of the slopes
        median_slope = np.median(slopes, axis=2, keepdims=True)
        mad_std_slope = 1.4826 * np.median(np.abs(slopes - median_slope), axis=2, keepdims=True)

        # Mask out values outside of a STD threshold
        mask = np.logical_and(
            slopes >= (median_slope - (mad_std_slope * std_threshold)),
            slopes <= (median_slope + (mad_std_slope * std_threshold)),
        )

        # Calculate slope using the masked median of the slopes
        masked_slopes = np.ma.masked_array(slopes, mask=~mask)

        # Still takes into account the masked values
        slope = np.ma.median(masked_slopes, axis=2, keepdims=True).filled(0)
        slope_list.append(slope)

    return np.vstack(slope_list)
