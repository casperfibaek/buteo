import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm


# https://en.wikipedia.org/wiki/Normal_distribution#Probability_density_function
# https://stackoverflow.com/questions/12412895/calculate-probability-in-normal-distribution-given-mean-std-in-python
values = np.random.normal(50, 1, 150)
values = np.append(values, [80, 30])

mean = values.mean()
std = values.std()

zscores = (values - mean) / std


# Turns zscores
def __cScale(zscore, sqrt=True, root=math.pi):
    cdf = 1 - abs((norm.cdf(zscore) - 0.5) / 0.5)
    if sqrt is True:
        return math.pow(cdf, 1 / root)
    else:
        return cdf


_cScale = np.vectorize(__cScale)


def cScale(arrOfZscores, sqrt=True, root=math.pi):
    return _cScale(arrOfZscores, sqrt=sqrt)


ar = np.linspace(0, 5, num=100)
ar2 = cScale(ar, sqrt=True, root=math.pi)

plot = plt.scatter(ar, ar2)
plt.show(plot)
# print(ar)
