import scipy.stats as s
import numpy as np

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

bob = np.array([3,5,4,2,1,6,8,4,3,2,6,4,2,1])

print(bob.std(), mad(bob))
