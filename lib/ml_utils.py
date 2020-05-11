from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
import numpy as np

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    sets = {}
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                target_name = corr_matrix.columns[j]
                col_corr.add(f"{colname}:{target_name}")
                # if colname in dataset.columns:
                #     del dataset[colname]  # deleting the column from the dataset

    return col_corr

def best_param_random(X, y, n_iter=50, cv=3):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 4, 6]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    random_forest_tuned = RandomForestClassifier()

    rf_random = RandomizedSearchCV(
        estimator=random_forest_tuned,
        param_distributions=random_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    rf_random.fit(X, y)

    return rf_random.best_params_
