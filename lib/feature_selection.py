from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def variance_threshold_selector(data, threshold=0.2):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def chi_selector(X, y, k=2):
    kbest = SelectKBest(chi2, k=2).fit_transform(X, y)
    outcome = kbest.get_support()
    # for i in range(0, len(name)):
    #     if outcome[i]:
    #         print name[i] 
