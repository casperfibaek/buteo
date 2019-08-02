import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import seaborn as sns


# in_csv = 'E:\\SATF\\phase_IV_urban-classification\\training_data\\training_data_cleaned_phase_I.csv'
# out_csv = 'E:\\SATF\\phase_IV_urban-classification\\training_data\\training_data_cleaned_phase_II.csv'
in_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selected.csv'
out_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selection_variance.csv'


data = pd.read_csv(in_csv)

X = data.drop(['DN', 'class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 89.36% acc
# random_forest_default = RandomForestClassifier(n_estimators=100)
# random_forest_default.fit(X_train, y_train)

# print('Before tuning')
# print(f"{'{0:.2f}'.format(random_forest_default.score(X_test, y_test) * 100)}% acc")


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]  # deleting the column from the dataset

correlation(X, 0.95)
X['DN'] = data['DN']
X['class'] = data['class']
X.to_csv(out_csv)

import pdb; pdb.set_trace()
exit()

# {'n_estimators': 1400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': False}
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

# estimator = RandomForestClassifier(
#     n_estimators=100,  # 1200
#     max_features='auto',
#     max_depth=30,  # 50
#     min_samples_split=2,  # 5 
#     min_samples_leaf=1,
#     bootstrap=False,
# )
estimator = RandomForestClassifier(n_estimators=100)
selector = RFECV(estimator, step=1, cv=3, verbose=2)
selector = selector.fit(X, y)

headers_masked = []
for nr, val in enumerate(selector.support_):
    if val is False:
        headers_masked.append(X.columns[nr])

masked = X.drop(headers_masked, axis=1)

masked['DN'] = data['DN']
masked['class'] = data['class']

masked.to_csv(out_csv)

# params = best_param_random(X, y, n_iter=50)

model = ExtraTreesClassifier(100)
model.fit(masked, y)

feat_importances = pd.Series(model.feature_importances_, index=masked.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

