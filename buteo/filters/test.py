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
    "np.information_measure_of_correlation1",
    "np.information_measure_of_correlation2",
    "inverseDifference",
    "maximumCorrelationCoefficient",
    "maximumProbability",
    "sumAverage",
    "sumEntropy",
    "sumOfSquaresVariance",
    "sumVariance",
]


class Feature_class:
    def __init__(self):
        for value in all_features:
            self.__dict__[value] = True


bob = Feature_class()

import pdb

pdb.set_trace()
