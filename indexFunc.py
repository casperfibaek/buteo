import numpy as np
from numba import jit
import math

np.seterr(divide='ignore', invalid='ignore') # disable warnings for division by zero

@jit(nopython = True, parallel = True, fastmath = True)
def msavi2(B08, B04):
    return (np.divide(2 * B08 + 1 - np.sqrt( np.power((2 * B08 + 1), 2) - 8 * (B08 - B04)), 2))

@jit(nopython = True, parallel = True, fastmath = True)
def mcari(B05_10m, B04, B03):
    return ((B05_10m - B04) - 0.2 * (B05_10m - B03)) * (np.divide(B05_10m, B04))

@jit(nopython = True, parallel = True, fastmath = True)
def s2rep(B07_10m, B06_10m, B05_10m, B04):
    return 705 + 35 * np.divide((np.divide((B07_10m + B04), 2) - B05_10m), (B06_10m - B05_10m))

@jit(nopython = True, parallel = True, fastmath = True)
def ndwi(B11_10m, B08):
    return np.divide((B08 - B11_10m), (B08 + B11_10m))

@jit(nopython = True, parallel = True, fastmath = True)
def ndvi(B08, B04):
    return np.divide((B08 - B04), (B08 + B04))

@jit(nopython = True, parallel = True, fastmath = True)
def cre(B07_10m, B05_10m):
    return np.power(np.divide(B07_10m, B05_10m), (-1))

@jit(nopython = True, parallel = True, fastmath = True)
def moist(B11_10m, B08A_10m):
    return np.divide((B08A_10m - B11_10m), (B08A_10m + B11_10m))

@jit(nopython = True, parallel = True, fastmath = True)
def evi(B08, B04, B02):
    return 2.5 * np.divide((B08 - B04), ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0))

@jit(nopython = True, parallel = True, fastmath = True)
def evi2(B08, B04):
    return 2.4 * np.divide((B08 - B04), (B08 + B04 + 1.0))

@jit(nopython = True, parallel = True, fastmath = True)
def nbr(B12_10m, B08):
    return np.divide((B08 - B12_10m), (B08 + B12_10m))

@jit(nopython = True, parallel = True, fastmath = True)
def ari(B05_10m, B03):
    return np.divide(1,  B03) - np.divide(1, B05_10m)

np.errstate(divide='ignore', invalid='ignore') # enable warnings for division by zero