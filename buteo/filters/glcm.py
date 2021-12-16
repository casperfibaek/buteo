# This code calculates gray-level invariant Haralick features according to
# [4] from one or more GLCMs calculated using e.g. MATLABs graycomatrix()
# function. The GLCMs do not have to be normalized, this is done by the
# function.

# Syntax:
# -------
# featureValues = GLCMFeaturesInvariant(GLCMs, features)

# GLCMs: an m-by-m-by-p array of GLCMs, where m is the dimension of each
# GLCM and p is the number of GLCMs in the array. Features are calculated
# for each of the p arrays.

# features: a string or cell array of strings, listing the features to
# calculate. If this is omitted, all features are calculated.

# GLCMFeaturesInvariant normalizes the GLCMs so the 'volume' of the GLCM is
# equal to 1. This is one step in making the Haralick features gray-level
# invariant.

# Features computed:
# ------------------
# Autocorrelation [2,4]
# Cluster Prominence [2,4]
# Cluster Shade [2,4]
# Contrast [1,4]
# Correlation [1,4]
# Difference average
# Difference entropy [1,4]
# Difference variance [1,4]
# Dissimilarity: [2,4]
# Energy [1,4]
# Entropy [2,4]
# Homogeneity: (Inverse Difference Moment) [1,2,4]
# np.information measure of correlation1 [1,4]
# np.informaiton measure of correlation2 [1,4]
# Inverse difference (Homogeneity in matlab): [3,4]
# Maximum correlation coefficient
# Maximum probability [2,4]
# Sum average [1,4]
# Sum entropy [1,4]
# Sum of sqaures: Variance [1,4]
# Sum variance [1,4]

# Example:
# --------
# First create GLCMs from a 2d image
# GLCMs = graycomatrix(image,'Offset',[0 1; -1 1;-1 0;-1 -1],'Symmetric',...
# true,'NumLevels',64,'GrayLimits',[0 255]);

# Sum the GLCMs of different directions to create a direction invariant
# # GLCM
# GLCM = sum(GLCMs,3)

# Calculate the invariant Haralick features
# features = GLCMFeaturesInvariant(GLCM)

# Calulate energy and entropy only
# features = GLCMFeaturesInvariant(GLCM,{'energy','entropy'})

import numpy as np
from scipy.sparse.linalg import eigs


def is_number(n):
    try:
        float(n)
        return True
    except ValueError:
        return False


def str_in_list(feature, list_of_features):
    if list_of_features == "all":
        return True

    if feature in list_of_features:
        return True

    return False


def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    return ind


def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0] * array_shape[1]] = -1
    rows = ind.astype("int") / array_shape[1]
    cols = ind % array_shape[1]
    return (rows, cols)


def GLCMFeaturesInvariant(
    glcm, features="all", homogeneityConstant=None, inverseDifferenceConstant=None
):
    if isinstance(features, list) and len(features) == 0:
        raise Exception("Not enough input arguments")
    elif isinstance(features, str) and features != "all":
        raise Exception("Invalid argument")
    else:
        if (glcm.shape[1 - 1] <= 1) or (glcm.shape[2 - 1] <= 1):
            raise Exception("The GLCM should be a 2-D or 3-D matrix.")
        else:
            if glcm.shape[1 - 1] != glcm.shape[2 - 1]:
                raise Exception(
                    "Each GLCM should be square with NumLevels rows and NumLevels cols"
                )

    checkHomogeneityConstant = lambda x=None: is_number(x) and x > 0 and x < np.inf
    checkInverseDifferenceConstant = (
        lambda x=None: is_number(x) and x > 0 and x < np.inf
    )

    if homogeneityConstant is None or not checkHomogeneityConstant(homogeneityConstant):
        homogeneityConstant = 1

    if inverseDifferenceConstant is None or not checkInverseDifferenceConstant(
        inverseDifferenceConstant
    ):
        inverseDifferenceConstant = 1

    # epsilon
    eps = 1e-10

    # Get size of GLCM
    nGrayLevels = glcm.shape[1 - 1]
    nglcm = glcm.shape[3 - 1]

    # Differentials
    dA = 1 / (nGrayLevels ** 2)
    dL = 1 / nGrayLevels

    dXplusY = 1 / (2 * nGrayLevels - 1)
    dXminusY = 1 / nGrayLevels
    dkdiag = 1 / nGrayLevels

    # Normalize the GLCMs
    glcm = glcm / np.sum(np.sum(glcm) * dA)
    # glcm = bsxfun(rdivide, glcm, sum(sum(glcm)) * dA)

    out = {}

    all_features = [
        "autoCorrelation",
        "clusterProminence",
        "clusterShade",
        "contrast",
        "correlation",
        "differenceAverage",
        "differenceEntropy",
        "differenceVariance",
        "dissimilarity",
        "different_entropy",
        "entropy",
        "homogeneity",
        "informationMeasureOfCorrelation1",
        "informationMeasureOfCorrelation2",
        "inverseDifference",
        "maximumCorrelationCoefficient",
        "maximumProbability",
        "sumAverage",
        "sumEntropy",
        "sumOfSquaresVariance",
        "sumVariance",
    ]

    for feat in all_features:
        if str_in_list(feat, features):
            out[feat] = np.zeros(nglcm)

    class Feature_class:
        def __init__(self):
            for key in out.keys():
                self.__dict__[key] = True

    use_features = Feature_class()

    glcmMean = np.zeros((nglcm, 1))
    uX = np.zeros((nglcm, 1))
    uY = np.zeros((nglcm, 1))
    sX = np.zeros((nglcm, 1))
    sY = np.zeros((nglcm, 1))

    # pX pY pXplusY pXminusY
    if (
        str_in_list("informationMeasureOfCorrelation1", features)
        or str_in_list("informationMeasureOfCorrelation2", features)
        or str_in_list("maximalCorrelationCoefficient", features)
    ):
        pX = np.zeros((nGrayLevels, nglcm))
        pY = np.zeros((nGrayLevels, nglcm))

    if (
        use_features.sumAverage
        or use_features.sumVariance
        or use_features.sumEntropy
        or use_features.sumVariance
    ):
        pXplusY = np.zeros(((nGrayLevels * 2 - 1), nglcm))

    if use_features.differenceEntropy or use_features.differenceVariance:
        pXminusY = np.zeros((nGrayLevels, nglcm))

    # HXY1 HXY2 HX HY
    if use_features.informationMeasureOfCorrelation1:
        HXY1 = np.zeros((nglcm, 1))
        HX = np.zeros((nglcm, 1))
        HY = np.zeros((nglcm, 1))

    if use_features.informationMeasureOfCorrelation2:
        HXY2 = np.zeros((nglcm, 1))

    # Create indices for vectorising code:
    sub = np.arange(1, nGrayLevels * nGrayLevels + 1)
    I, J = ind2sub(np.array([nGrayLevels, nGrayLevels]), sub)
    nI = I / nGrayLevels
    nJ = J / nGrayLevels
    if use_features.sumAverage or use_features.sumVariance or use_features.sumEntropy:
        sumLinInd = np.empty((1, 2 * nGrayLevels - 1), dtype=np.int)
        for i in np.arange(1, 2 * nGrayLevels - 1 + 1).reshape(-1):
            diagonal = i - nGrayLevels
            d = np.ones((1, nGrayLevels - np.abs(diagonal)))
            diag_ = np.diag(d, diagonal)
            diag_ud_ = np.flipud(diag_)
            sumLinInd[i] = diag_ud_[diag_ud_ != 0]

    if (
        use_features.differenceAverage
        or use_features.differenceVariance
        or use_features.differenceEntropy
    ):
        diffLinInd = np.empty((1, nGrayLevels), dtype=np.int)
        idx2 = np.arange(0, nGrayLevels - 1 + 1)
        for i in idx2.reshape(-1):
            diagonal = i
            d = np.ones((1, nGrayLevels - diagonal))
            if diagonal == 0:
                D = np.diag(d, diagonal)
                diffLinInd[i + 1] = D[D != 0]
            else:
                Dp = np.diag(d, diagonal)
                Dn = np.diag(d, -diagonal)
                Dp_Dn = Dp + Dn
                diffLinInd[i + 1] = Dp_Dn[Dp_Dn != 0]

    sumIndices = np.arange(2, 2 * nGrayLevels + 1)

    # Loop over all GLCMs
    for k in np.arange(1, nglcm + 1).reshape(-1):
        currentGLCM = glcm[:, :, k]
        glcmMean[k] = np.mean(currentGLCM)

        # For symmetric GLCMs, uX = uY
        uX[k] = np.sum(np.multiply(nI, currentGLCM(sub))) * dA
        uY[k] = np.sum(np.multiply(nJ, currentGLCM(sub))) * dA
        sX[k] = np.sum(np.multiply((nI - uX[k]) ** 2, currentGLCM(sub))) * dA
        sY[k] = np.sum(np.multiply((nJ - uY[k]) ** 2, currentGLCM(sub))) * dA

        if (
            use_features.sumAverage
            or use_features.sumVariance
            or use_features.sumEntropy
        ):
            for i in sumIndices.reshape(-1):
                pXplusY[i - 1, k] = np.sum(currentGLCM(sumLinInd[i - 1])) * dkdiag

        if (
            use_features.differenceAverage
            or use_features.differenceVariance
            or use_features.differenceEntropy
        ):
            idx2 = np.arange(0, nGrayLevels - 1 + 1)

            for i in idx2.reshape(-1):
                pXminusY[i + 1, k] = np.sum(currentGLCM(diffLinInd[i + 1])) * dkdiag

        if (
            use_features.informationMeasureOfCorrelation1
            or use_features.informationMeasureOfCorrelation2
            or use_features.maximalCorrelationCoefficient
        ):
            pX[:, k] = np.sum(currentGLCM, 2 - 1) * dL
            pY[:, k] = np.transpose(np.sum(currentGLCM, 1 - 1)) * dL

        if use_features.informationMeasureOfCorrelation1:
            HX[k] = -np.nansum(np.multiply(pX[:, k], np.log(pX[:, k]))) * dL
            HY[k] = -np.nansum(np.multiply(pY[:, k], np.log(pY[:, k]))) * dL

            HXY1[k] = (
                -np.nansum(
                    np.multiply(
                        np.transpose(currentGLCM(sub)),
                        np.log(np.multiply(pX[I, k], pY[J, k])),
                    )
                )
                * dA
            )

        if use_features.informationMeasureOfCorrelation2:
            HXY2[k] = (
                -np.nansum(
                    np.multiply(
                        np.multiply(pX[I, k], pY[J, k]),
                        np.log(np.multiply(pX[I, k], pY[J, k])),
                    )
                )
                * dA
            )

        # Haralick features:
        if use_features.energy:
            out["energy"][k] = np.sum(currentGLCM(sub) ** 2) * dA

        if use_features.contrast:
            out["contrast"][k] = (
                np.sum(np.multiply((nI - nJ) ** 2, currentGLCM(sub))) * dA
            )

        if use_features.autoCorrelation or use_features.correlation:
            autoCorrelation = (
                np.sum(np.multiply(np.multiply(nI, nJ), currentGLCM(sub))) * dA
            )

            if use_features.autoCorrelation:
                out["autoCorrelation"][k] = autoCorrelation

        if use_features.correlation:
            if sX[k] < eps or sY[k] < eps:
                out["correlation"][k] = np.amin(
                    np.amax((autoCorrelation - np.multiply(uX[k], uY[k])), -1), 1
                )
            else:
                out["correlation"][k] = (
                    autoCorrelation - np.multiply(uX[k], uY[k])
                ) / np.sqrt(np.multiply(sX[k], sY[k]))

        if use_features.sumOfSquaresVariance:
            out["sumOfSquaresVariance"][k] = (
                np.sum(np.multiply(currentGLCM(sub), ((nI - uX[k]) ** 2))) * dA
            )

        if use_features.homogeneity:
            out["homogeneity"][k] = (
                np.sum(currentGLCM(sub) / (1 + homogeneityConstant * (nI - nJ) ** 2))
                * dA
            )

        if use_features.sumAverage or use_features.sumVariance:
            sumAverage = (
                np.sum(
                    np.multiply(
                        np.transpose(((2 * (sumIndices - 1)) / (2 * nGrayLevels - 1))),
                        pXplusY[sumIndices - 1, k],
                    )
                )
                * dXplusY
            )

            if use_features.sumAverage:
                out["sumAverage"][k] = sumAverage

        if use_features.sumVariance:
            out["sumVariance"][k] = (
                np.sum(
                    np.multiply(
                        np.transpose(
                            (
                                ((2 * (sumIndices - 1)) / (2 * nGrayLevels - 1))
                                - sumAverage
                            )
                        )
                        ** 2,
                        pXplusY[sumIndices - 1, k],
                    )
                )
                * dXplusY
            )

        if use_features.sumEntropy:
            out["sumEntropy"][k] = (
                -np.nansum(
                    np.multiply(
                        pXplusY[sumIndices - 1, k], np.log(pXplusY[sumIndices - 1, k])
                    )
                )
                * dXplusY
            )

        if (
            use_features.entropy
            or use_features.informationMeasureOfCorrelation1
            or use_features.informationMeasureOfCorrelation2
        ):
            entropy = (
                -np.nansum(np.multiply(currentGLCM(sub), np.log(currentGLCM(sub)))) * dA
            )

            if use_features.entropy:
                out["entropy"][k] = entropy

        if use_features.differenceAverage or use_features.differenceVariance:
            differenceAverage = (
                np.sum(
                    np.multiply(
                        np.transpose(((idx2 + 1) / nGrayLevels)),
                        pXminusY[idx2 + 1, k],
                    )
                )
                * dXminusY
            )

            if use_features.differenceAverage:
                out["differenceAverage"][k] = differenceAverage

        if use_features.differenceVariance:
            out["differenceVariance"][k] = (
                np.sum(
                    np.multiply(
                        np.transpose(
                            (((idx2 + 1) / nGrayLevels) - differenceAverage) ** 2
                        ),
                        pXminusY[idx2 + 1, k],
                    )
                )
                * dXminusY
            )

        if use_features.differenceEntropy:
            out["differenceEntropy"][k] = (
                -np.nansum(
                    np.multiply(pXminusY[idx2 + 1, k], np.log(pXminusY[idx2 + 1, k]))
                )
                * dXminusY
            )

        if use_features.informationMeasureOfCorrelation1:
            np.infoMeasure1 = (entropy - HXY1(k)) / (np.amax(HX(k), HY(k)))
            out["informationMeasureOfCorrelation1"][k] = np.infoMeasure1

        if use_features.informationMeasureOfCorrelation2:
            np.infoMeasure2 = np.sqrt(1 - np.exp(-2 * (HXY2(k) - entropy)))
            out["informationMeasureOfCorrelation2"][k] = np.infoMeasure2

        if use_features.maximalCorrelationCoefficient:
            # Correct by eps if the matrix has columns or rows that sums to zero.
            P = currentGLCM

            # pX_ = pX(:,k)
            pX_ = pX[:, k]
            if np.any(pX_ < eps):
                pX_ = pX_ + eps
                pX_ = pX_ / (np.sum(pX_) * dL)

            # pY_ = pY(:,k)
            pY_ = pY[:, k]
            if np.any(pY_ < eps):
                pY_ = pY_ + eps
                pY_ = pY_ / (np.sum(pY_) * dL)

            # Compute the Markov matrix
            Q = np.zeros((P.shape, P.shape))
            for i in np.arange(1, nGrayLevels + 1).reshape(-1):
                # Pi = P(i,:)
                Pi = P[i, :]
                pXi = pX_(i)
                for j in np.arange(1, nGrayLevels + 1).reshape(-1):
                    # Pj = P(j,:)
                    Pj = P[j, :]
                    d = pXi * np.transpose(pY_)
                    if d < eps:
                        print(
                            "Division by zero in the maximalCorrelationCoefficient!\n"
                            % ()
                        )
                    Q[i, j] = dA * np.sum((np.multiply(Pi, Pj)) / d)

            # Compute the second largest eigenvalue
            if np.any(np.inf(Q)):
                e2 = np.nan
            else:
                try:
                    E = eigs(Q, 2)
                finally:
                    pass
                # There may be a near-zero imaginary component here
                if True and True:
                    e2 = E(2)
                else:
                    e2 = np.amin(real(E(1)), real(E(2)))
            out["maximalCorrelationCoefficient"][k] = e2

        if use_features.dissimilarity:
            dissimilarity = np.sum(np.multiply(np.abs(nI - nJ), currentGLCM(sub))) * dA
            out["dissimilarity"][k] = dissimilarity
        if use_features.clusterShade:
            out["clusterShade"][k] = (
                np.sum(np.multiply((nI + nJ - uX[k] - uY[k]) ** 3, currentGLCM(sub)))
                * dA
            )
        if use_features.clusterProminence:
            out["clusterProminence"][k] = (
                np.sum(np.multiply((nI + nJ - uX[k] - uY[k]) ** 4, currentGLCM(sub)))
                * dA
            )
        if use_features.maximumProbability:
            out["maximumProbability"][k] = np.amax(currentGLCM)
        if use_features.inverseDifference:
            out["inverseDifference"][k] = (
                np.sum(
                    currentGLCM(sub) / (1 + inverseDifferenceConstant * np.abs(nI - nJ))
                )
                * dA
            )

    return out


import sys

sys.path.append("../../")
from buteo.raster.io import raster_to_array
from skimage.feature import graycomatrix, greycoprops

folder = "C:/Users/caspe/Desktop/test_area/raster/"
raster = folder + "B12_20m.tif"
arr = raster_to_array(raster)
arr = np.rint((arr / arr.max()) * 256).astype("uint8")
glcm = graycomatrix(
    arr[:, :, 0],
    [1],
    [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    256,
    symmetric=False,
    normed=True,
)

bob = GLCMFeaturesInvariant(glcm)
import pdb

pdb.set_trace()
