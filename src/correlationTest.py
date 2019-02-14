import numpy as np
from scipy.stats import stats


testArr1 = np.array([2, 3, 4, 5, 4, 3, 2, 5, 1])
testArr2 = np.array([6, 2, 4, 6, 8, 9, 3, 2, 1])


add = np.multiply(testArr1, testArr2)


# Enhanced vegetation index v2
def evi2(B08, B04):
    return 2.4 * np.divide((B08 - B04), (B08 + B04 + 1.0))



_evi2 = evi2(testArr1, testArr2)
print(add)
print(_evi2)


spear = stats.spearmanr(_evi2, add)

# print(_evi2)
# print(_savi2)

# pearson = stats.pearsonr(_evi2, _savi2)
# spearman = stats.spearmanr(_evi2, _savi2)
# print(pearson)
# print(spearman)
print(spear)
