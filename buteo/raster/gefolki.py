import numpy as np
from skimage import color, img_as_float, img_as_uint
from skimage.exposure import rescale_intensity
from skimage.util import view_as_blocks
from skimage.transform import resize as _resize
from scipy import signal, ndimage

"""
Adapted code from "Contrast Limited Adaptive Histogram Equalization" by Karel
Zuiderveld <karel@cv.ruu.nl>, Graphics Gems IV, Academic Press, 1994.

http://tog.acm.org/resources/GraphicsGems/gems.html#gemsvi

The Graphics Gems code is copyright-protected.  In other words, you cannot
claim the text of the code as your own and resell it. Using the code is
permitted in any program, product, or library, non-commercial or commercial.
Giving credit is not required, though is a nice gesture.  The code comes as-is,
and if there are any flaws or problems with any Gems code, nobody involved with
Gems - authors, editors, publishers, or webmasters - are to be held
responsible.  Basically, don't be a jerk, and remember that anything free
comes with no guarantee.
"""
def equalize_adapthist(image, ntiles_x=8, ntiles_y=8, clip_limit=0.01, nbins=256, nr_of_grey=16384):
    """Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : array-like
        Input image.
    ntiles_x : int, optional
        Number of tile regions in the X direction.  Ranges between 2 and 16.
    ntiles_y : int, optional
        Number of tile regions in the Y direction.  Ranges between 2 and 16.
    clip_limit : float: optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast).
    nbins : int, optional
        Number of gray bins for histogram ("dynamic range").

    Returns
    -------
    out : ndarray
        Equalized image.

    Notes
    -----
    * The algorithm relies on an image whose rows and columns are even
      multiples of the number of tiles, so the extra rows and columns are left
      at their original values, thus  preserving the input image shape.
    * For color images, the following steps are performed:
       - The image is converted to LAB color space
       - The CLAHE algorithm is run on the L channel
       - The image is converted back to RGB space and returned
    * For RGBA images, the original alpha channel is removed.

    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/gems.html#gemsvi
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """
    args = [None, ntiles_x, ntiles_y, clip_limit * nbins, nbins]
    if image.ndim > 2:
        lab_img = color.rgb2lab(img_as_float(image))
        l_chan = lab_img[:, :, 0]
        l_chan /= np.max(np.abs(l_chan))
        l_chan = img_as_uint(l_chan)
        args[0] = rescale_intensity(l_chan, out_range=(0, nr_of_grey - 1))
        new_l = _clahe(*args).astype(float)
        new_l = rescale_intensity(new_l, out_range=(0, 100))
        lab_img[:new_l.shape[0], :new_l.shape[1], 0] = new_l
        image = color.lab2rgb(lab_img)
        image = rescale_intensity(image, out_range=(0, 1))

    else:
        image = img_as_uint(image)
        args[0] = rescale_intensity(image, out_range=(0, nr_of_grey - 1))
        out = _clahe(*args)
        image[:out.shape[0], :out.shape[1]] = out
        image = rescale_intensity(image)

    return image


def _clahe(image, ntiles_x, ntiles_y, clip_limit, nbins=128, nr_of_grey=16384, max_reg_x=16, max_reg_y=16):
    """Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : array-like
        Input image.
    ntiles_x : int, optional
        Number of tile regions in the X direction.  Ranges between 2 and 16.
    ntiles_y : int, optional
        Number of tile regions in the Y direction.  Ranges between 2 and 16.
    clip_limit : float, optional
        Normalized clipping limit (higher values give more contrast).
    nbins : int, optional
        Number of gray bins for histogram ("dynamic range").
    nr_of_grey : int, optional
        Number of gray levels for image (2 ** bits_per_pixel).
    max_reg_x : int, optional
        Maximum number of contextual regions in the X direction.
    max_reg_y : int, optional
        Maximum number of contextual regions in the Y direction.

    Returns
    -------
    out : ndarray
        Equalized image.

    The number of "effective" greylevels in the output image is set by `nbins`;
    selecting a small value (eg. 128) speeds up processing and still produce
    an output image of good quality. The output image will have the same
    minimum and maximum value as the input image. A clip limit smaller than 1
    results in standard (non-contrast limited) AHE.
    """
    ntiles_x = min(ntiles_x, max_reg_x)
    ntiles_y = min(ntiles_y, max_reg_y)
    ntiles_y = max(ntiles_x, 2)
    ntiles_x = max(ntiles_y, 2)

    if clip_limit == 1.0:
        return image  # is OK, immediately returns original image.

    map_array = np.zeros((ntiles_y, ntiles_x, nbins), dtype=int)

    y_res = image.shape[0] - image.shape[0] % ntiles_y
    x_res = image.shape[1] - image.shape[1] % ntiles_x
    image = image[: y_res, : x_res]

    x_size = image.shape[1] // ntiles_x  # Actual size of contextual regions
    y_size = image.shape[0] // ntiles_y
    n_pixels = x_size * y_size

    if clip_limit > 0.0:  # Calculate actual cliplimit
        clip_limit = int(clip_limit * (x_size * y_size) / nbins)
        if clip_limit < 1:
            clip_limit = 1
    else:
        clip_limit = nr_of_grey  # Large value, do not clip (AHE)

    bin_size = 1 + nr_of_grey / nbins
    aLUT = np.arange(nr_of_grey)

    toto = 1.0 * aLUT
    toto //= bin_size
    aLUT = toto

    aLUT = np.round(aLUT)
    aLUT = np.abs(aLUT)
    aLUT = np.intp(aLUT)
    img_blocks = view_as_blocks(image, (y_size, x_size))

    # Calculate greylevel mappings for each contextual region
    for y in range(ntiles_y):
        for x in range(ntiles_x):
            sub_img = img_blocks[y, x]
            hist_indices = np.rint(sub_img.ravel())
            hist = aLUT[hist_indices.astype(int)] # pylint: disable=unsubscriptable-object
            hist = np.bincount(hist)
            hist = np.append(hist, np.zeros(nbins - hist.size, dtype=int))
            hist = clip_histogram(hist, clip_limit)
            hist = map_histogram(hist, 0, nr_of_grey - 1, n_pixels)
            map_array[y, x] = hist

    # Interpolate greylevel mappings to get CLAHE image
    ystart = 0
    for y in range(ntiles_y + 1):
        xstart = 0
        if y == 0:  # special case: top row
            ystep = y_size / 2.0
            yU = 0
            yB = 0
        elif y == ntiles_y:  # special case: bottom row
            ystep = y_size / 2.0
            yU = ntiles_y - 1
            yB = yU
        else:  # default values
            ystep = y_size
            yU = y - 1
            yB = yB + 1

        for x in range(ntiles_x + 1):
            if x == 0:  # special case: left column
                xstep = x_size / 2.0
                xL = 0
                xR = 0
            elif x == ntiles_x:  # special case: right column
                xstep = x_size / 2.0
                xL = ntiles_x - 1
                xR = xL
            else:  # default values
                xstep = x_size
                xL = x - 1
                xR = xL + 1

            mapLU = map_array[yU, xL]
            mapRU = map_array[yU, xR]
            mapLB = map_array[yB, xL]
            mapRB = map_array[yB, xR]

            xslice = np.arange(xstart, xstart + xstep)
            yslice = np.arange(ystart, ystart + ystep)
            interpolate(image, xslice, yslice, mapLU, mapRU, mapLB, mapRB, aLUT)

            xstart += xstep  # set pointer on next matrix */

        ystart += ystep

    return image


def clip_histogram(hist, clip_limit):
    """
    Perform clipping of the histogram and redistribution of bins.

    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).

    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.

    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = int(n_excess / hist.size)  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    hist[excess_mask] = clip_limit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = (hist >= upper) & (hist < clip_limit)
    mid = hist[mid_mask]
    n_excess -= mid.size * clip_limit - mid.sum()
    hist[mid_mask] = clip_limit

    while n_excess > 0:  # Redistribute remaining excess
        index = 0

        while n_excess > 0 and index < hist.size:
            step_size = int(hist[hist < clip_limit].size / n_excess)
            step_size = max(step_size, 1)
            indices = np.arange(index, hist.size, step_size)
            under = hist[indices] < clip_limit
            hist[under] += 1
            n_excess -= hist[under].size
            index += 1

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """
    Calculate the equalized lookup table (mapping).

    It does so by cumulating the input histogram.

    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.

    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    out = np.cumsum(hist).astype(float)
    scale = ((float)(max_val - min_val)) / n_pixels
    out *= scale
    out += min_val
    out[out > max_val] = max_val

    return out.astype(int)


def interpolate(image, xslice, yslice, mapLU, mapRU, mapLB, mapRB, aLUT):
    """
    Find the new grayscale level for a region using bilinear interpolation.

    Parameters
    ----------
    image : ndarray
        Full image.
    xslice, yslice : array-like
       Indices of the region.
    map* : ndarray
        Mappings of greylevels from histograms.
    aLUT : ndarray
        Maps grayscale levels in image to histogram levels.

    Returns
    -------
    out : ndarray
        Original image with the subregion replaced.

    Notes
    -----
    This function calculates the new greylevel assignments of pixels within
    a submatrix of the image. This is done by a bilinear interpolation between
    four different mappings in order to eliminate boundary artifacts.
    """
    norm = xslice.size * yslice.size  # Normalization factor

    # interpolation weight matrices
    x_coef, y_coef = np.meshgrid(np.arange(xslice.size),
                                 np.arange(yslice.size))
    x_inv_coef, y_inv_coef = x_coef[:, ::-1] + 1, y_coef[::-1] + 1

    view = image[int(yslice[0]):int(yslice[-1] + 1),
                 int(xslice[0]):int(xslice[-1] + 1)]
    view = np.rint(view).astype(int)

    im_slice = aLUT[view]
    new = ((y_inv_coef * (x_inv_coef * mapLU[im_slice]
                          + x_coef * mapRU[im_slice])
            + y_coef * (x_inv_coef * mapLB[im_slice]
                        + x_coef * mapRB[im_slice]))
           / norm)
    view[:, :] = new

    return image


def conv2SepMatlabbis(I, fen):
    rad = int((fen.size-1)/2)
    ligne = np.zeros((rad, I.shape[1]))
    I = np.append(ligne, I, axis=0)
    I = np.append(I, ligne, axis=0)

    colonne = np.zeros((I.shape[0], rad))
    I = np.append(colonne, I, axis=1)
    I = np.append(I, colonne, axis=1)

    res = conv2bis(conv2bis(I, fen.T), fen)

    return res


def EFolkiIter(I0, I1, iteration=5, radius=[8, 4], rank=4, uinit=None,vinit=None):
    talon=1.e-8
    if rank > 0:
        I0 = rank_filter_sup(I0, rank)
        I1 = rank_filter_sup(I1, rank)

    if uinit is None:
        u = np.zeros(I0.shape)
    else:
        u = uinit
    if vinit is None:
        v = np.zeros(I1.shape)
    else:
        v = vinit

    Iy, Ix = np.gradient(I0)

    cols, rows = I0.shape[1], I0.shape[0]
    x, y = np.meshgrid(range(cols), range(rows))

    for rad in radius:

        burt1D = np.array(np.ones([1, 2*rad+1]))/(2*rad + 1)
        W = lambda x: conv2SepMatlabbis(x, burt1D)

        Ixx = W(Ix*Ix) + talon
        Iyy = W(Iy*Iy) + talon
        Ixy = W(Ix*Iy)
        D = Ixx*Iyy - Ixy**2

        for _ in range(iteration):
            i1w = interp2(I1, x+u, y+v)

            it = I0 - i1w + u*Ix + v*Iy
            Ixt = W(Ix * it)
            Iyt = W(Iy * it)

            with np.errstate(divide='ignore', invalid='ignore'):
                u = (Iyy * Ixt - Ixy * Iyt) / D
                v = (Ixx * Iyt - Ixy * Ixt) / D

            invalid = np.isnan(u) | np.isinf(u) | np.isnan(v) | np.isinf(v)
            u[invalid] = 0
            v[invalid] = 0

    return u, v


def resize(image, shape, order=1):
    """Resize image to match a certain shape.

    Parameters
    ----------
    image : ndarray
        Input image.
    shape : tuple
        Shape of the output image.
    order : int, optional
        The order of the spline interpolation, default is 1.

    Returns
    -------
    out : ndarray
        Resized image.
    """
    return _resize(image, shape, order=order)


def GEFolkiIter(I0, I1, iteration=5, radius=[8, 4], rank=4, uinit=None, vinit=None):
    if rank > 0:
        R0 = rank_filter_sup(I0, rank)
        R1i = rank_filter_inf(I1, rank)
        R1s = rank_filter_sup(I1, rank)

    H0 = I0
    H1 = I1

    x = I0.shape[1]
    res_x = x % 8
    add_x = 8 - x % 8 if res_x > 0 else 0

    y = I0.shape[0]
    res_y = y % 8
    add_y = 8 - y % 8 if res_y > 0 else 0

    if res_x > 0 or res_y > 0:
        toto = resize(I0, (y+add_y, x+add_x), order=1)
    else:
        toto = I0

    toto = toto*255
    toto = toto.astype(np.uint8)
    H0 = equalize_adapthist(toto, 8, clip_limit=1, nbins=256)

    if res_x > 0 or res_y > 0:
        H0 = resize(H0, (y, x), order=1)

    H0 = H0.astype(np.float32)
    H0 = H0 / H0.max()

    x = I1.shape[1]
    res_x = x % 8
    add_x = 8 - x % 8 if res_x > 0 else 0

    y = I1.shape[0]
    res_y = y % 8
    add_y = 8 - y % 8 if res_y > 0 else 0

    if res_x > 0 or res_y > 0:
        toto = resize(I1, (y+add_y, x+add_x), order=1)
    else:
        toto = I1

    toto = toto * 255
    toto = toto.astype(np.uint8)
    H1 = equalize_adapthist(toto, 8, clip_limit=1, nbins=256)

    if res_x > 0 or res_y > 0:
        H1 = resize(H1, (y, x), order=1)

    H1 = H1.astype(np.float32)
    H1 = H1/H1.max()

    if uinit is None:
        u = np.zeros(I0.shape)
    else:
        u = uinit
    if vinit is None:
        v = np.zeros(I1.shape)
    else:
        v = vinit

    Iy, Ix = np.gradient(R0)

    cols, rows = I0.shape[1], I0.shape[0]
    x, y = np.meshgrid(range(cols), range(rows))
    for rad in radius:

        burt1D = np.array(np.ones([1, 2*rad+1]))/(2*rad + 1)
        W = lambda xin: conv2SepMatlabbis(xin, burt1D)

        Ixx = W(Ix*Ix)
        Iyy = W(Iy*Iy)
        Ixy = W(Ix*Iy)
        D = Ixx*Iyy - Ixy**2

        for _ in range(iteration):
            dx = x + u
            dy = y + v
            dx[dx < 0] = 0
            dy[dy < 0] = 0
            dx[dx > cols-1] = cols-1
            dy[dy > rows-1] = rows-1

            H1w = interp2(H1, dx, dy)

            crit1 = conv2SepMatlabbis(np.abs(H0-H1w), np.ones([2*rank+1, 1]))
            crit2 = conv2SepMatlabbis(np.abs(1-H0-H1w), np.ones([2*rank+1, 1]))

            R1w = interp2(R1s, x+u, y+v)
            R1w_1 = interp2(R1i, x+u, y+v)

            R1w[crit1 > crit2] = R1w_1[crit1 > crit2]
            it = R0 - R1w + u*Ix + v*Iy
            Ixt = W(Ix * it)
            Iyt = W(Iy * it)

            with np.errstate(divide='ignore', invalid='ignore'):
                u = (Iyy * Ixt - Ixy * Iyt) / D
                v = (Ixx * Iyt - Ixy * Ixt) / D

            invalid = np.isnan(u) | np.isinf(u) | np.isnan(v) | np.isinf(v)
            u[invalid] = 0
            v[invalid] = 0

    return u, v


def conv2(I, w):
    return signal.convolve2d(I, w, mode='valid')


def interp2(I, x, y, use_linear=True):
    if use_linear:
        return ndimage.map_coordinates(I, [y, x], order=1, mode='nearest').reshape(I.shape)

    return ndimage.map_coordinates(I, [y, x], order=3, mode='nearest')


def conv2bis(I, w):
    return signal.convolve2d(I, w, mode='valid')


def conv2Sep(I, w):
    return conv2(conv2(I, w.T), w)


class BurtOF:
    def __init__(self, flow, levels=4):
        self.flow = flow
        self.levels = levels

    def __call__(self, I0, I1, **kparams):

        if 'levels' in kparams:
            self.levels = kparams.pop('levels')

        I0 = (I0-I0.min())/(I0.max()-I0.min())
        I1 = (I1-I1.min())/(I1.max()-I1.min())

        Py0 = [I0]
        Py1 = [I1]

        for i in range(self.levels, 0, -1):
            Py0.append(self.pyrUp(Py0[-1]))
            Py1.append(self.pyrUp(Py1[-1]))

        u = np.zeros(Py0[-1].shape)
        v = np.zeros(Py0[-1].shape)

        for i in range(self.levels, -1, -1):
            kparams['uinit'] = u
            kparams['vinit'] = v
            u, v = self.flow(Py0[i], Py1[i], **kparams)
            if i > 0:
                col, row = Py0[i-1].shape[1], Py0[i-1].shape[0]
                u = 2 * self.pyrDown(u, (row, col))
                v = 2 * self.pyrDown(v, (row, col))

        return u, v

    def conv2SepMatlab(self, I, fen):
        rad = int((fen.size-1)/2)
        ligne = np.zeros((rad, I.shape[1]))
        I = np.append(ligne, I, axis=0)
        I = np.append(I, ligne, axis=0)

        colonne = np.zeros((I.shape[0], rad))
        I = np.append(colonne, I, axis=1)
        I = np.append(I, colonne, axis=1)

        res = conv2bis(conv2bis(I, fen.T), fen)

        return res

    def pyrUp(self, I):
        a = 0.4
        burt1D = np.array([[1./4.-a/2., 1./4., a, 1./4., 1./4.-a/2.]])

        M = self.conv2SepMatlab(I, burt1D)
        self.toto = M

        return M[::2, ::2]

    def pyrDown(self, I, shape):
        res = np.zeros(shape)
        I = np.repeat(np.repeat(I, 2, 0), 2, 1)
        col, row = I.shape[1], I.shape[0]
        col = min(shape[1], col)
        row = min(shape[0], row)
        res[:row, :col] = I[:row, :col]

        return res

def rank_filter_sup(I, rad):
    nl, nc = I.shape
    R = np.zeros([nl, nc])
    for i in range(-rad, rad+1):  # indice de ligne
        for j in range(-rad, rad+1):  # indice de colonne
            if i != 0:
                if i < 0:
                    tmp = np.concatenate([I[-i:, :], np.zeros([-i, nc])], axis=0)
                else:
                    tmp = np.concatenate([np.zeros([i, nc]), I[:-i, :]], axis=0)
            else:
                tmp = I
            if j != 0:
                if j < 0:
                    tmp = np.concatenate([tmp[:, -j:], np.zeros([nl, -j])], axis=1)
                else:
                    tmp = np.concatenate([np.zeros([nl, j]), tmp[:, :-j]], axis=1)

            idx = tmp > I
            R[idx] = R[idx]+1

    return R

def rank_filter_inf(I, rad):
    nl, nc = I.shape
    R = np.zeros([nl, nc])

    for i in range(-rad, rad+1):
        for j in range(-rad, rad+1):
            if i != 0:
                if i < 0:  # on decalle vers le haut de i lignes
                    tmp = np.concatenate([I[-i:, :], np.zeros([-i, nc])], axis=0)
                else:
                    tmp = np.concatenate([np.zeros([i, nc]), I[:-i, :]], axis=0)
            else:
                tmp = I
            if j != 0:
                if j < 0:
                    tmp = np.concatenate([tmp[:, -j:], np.zeros([nl, -j])], axis=1)
                else:
                    tmp = np.concatenate([np.zeros([nl, j]), tmp[:, :-j]], axis=1)

            idx = tmp < I
            R[idx] = R[idx]+1

    return R

def wrapData(I, u, v):
    """ Apply the [u,v] optical flow to the data I """
    col, row = I.shape[1], I.shape[0]
    X, Y = np.meshgrid(range(col), range(row))
    R = interp2(I, X+u, Y+v)

    return R
