from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump

# in_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selection_variance.csv'
in_csv = 'C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selected_standardized.csv'
out_model = 'C:\\Users\\CFI\\Desktop\\satf_training\\models\\RF_std.model'

# data = pd.read_csv(in_csv)
data = pd.read_csv(in_csv, index_col=0)

# import pdb; pdb.set_trace()
# exit()


# X = data.drop(['DN', 'class', 'Unnamed: 0.1'], axis=1)
X = data.drop(['DN', 'class'], axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# random_forest_default = RandomForestClassifier(n_estimators=100)
# random_forest_default.fit(X_train, y_train)

# print('Before tuning')
# print(f"{'{0:.2f}'.format(random_forest_default.score(X_test, y_test) * 100)}% acc")


def best_param_random(X, y, n_iter=100, cv=3):
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


# bp = best_param_random(X_train, y_train)
# {'n_estimators': 600, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 60, 'bootstrap': False}

# Create the parameter grid based on the results of random search
param_grid = {
    'n_estimators': [600],
    'min_samples_split': [3],
    'min_samples_leaf': [1],
    'max_features': ['sqrt'],
    'max_depth': [80, 90, 100],
    'bootstrap': [False],
}


def best_param(X, y, cv=3):
    random_forest_params = RandomForestClassifier()

    grid_search = GridSearchCV(
        estimator=random_forest_params,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X, y)

    return grid_search.best_params_


# bp = best_param(X, y)
# {'n_estimators': 600, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 90, 'bootstrap': False}

# import pdb; pdb.set_trace()
# exit()

random_forest_tuned = RandomForestClassifier(
    n_estimators=600,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=90,
    bootstrap=False,
)
random_forest_tuned.fit(X, y)

print('After tuning')
print(f"{'{0:.2f}'.format(random_forest_tuned.score(X_test, y_test) * 100)}% acc")

dump(random_forest_tuned, out_model)

# pred = random_forest_tuned.predict(X_test)
# labels = sorted(set(pred))
# matrix = confusion_matrix(pred, y_test, labels=labels)
# matrix = pd.DataFrame(matrix, index=labels, columns=labels)

# report = classification_report(pred.tolist(), y_test.tolist(), labels=labels)
# print(report)

# sn.set(font_scale=0.8)  # for label size
# sn.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g')  # font size
# plt.show()

import pdb; pdb.set_trace()
exit()
