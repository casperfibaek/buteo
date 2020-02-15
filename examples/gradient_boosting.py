import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from joblib import dump


df = pd.read_csv('C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selected_standardized.csv', index_col=0)

X = df.drop(['DN', 'class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# gb_classifier = GradientBoostingClassifier()
# gb_classifier.fit(X_train, y_train)

# print('Before tuning')
# print(f"{'{0:.2f}'.format(gb_classifier.score(X_test, y_test) * 100)}% acc")


# 88.03% before


# {'n_estimators': 1000, 'min_samples_split': 4, 'min_samples_leaf': 2, 'max_depth': 5, 'learning_rate': 0.05}
# {'n_estimators': 1200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 5, 'learning_rate': 0.025}
# {'n_estimators': 1800, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 5, 'learning_rate': 0.025}
def best_param(X, y):
    learning_rate = [0.025, 0.015]
    n_estimators = [1200, 1500, 1800]
    min_samples_split = [5]
    min_samples_leaf = [2]
    max_depth = [5]

    random_grid = {'learning_rate': learning_rate,
                   'n_estimators': n_estimators,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'max_depth': max_depth}

    estim = GradientBoostingClassifier()

    grid_search = RandomizedSearchCV(
        estimator=estim,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X, y)

    return grid_search.best_params_

# bp = best_param(X, y)


# logistic = 85.37% acc
# relu = 86.92% acc
# clf = GradientBoostingClassifier(
#     learning_rate=0.025,
#     n_estimators=2000,
#     min_samples_split=5,
#     min_samples_leaf=2,
#     max_depth=5,
#     verbose=10,
# )

# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# print(f"{'{0:.2f}'.format(accuracy_score(y_test, y_pred) * 100)}% acc")
# import pdb; pdb.set_trace()


clf = GradientBoostingClassifier(
    learning_rate=0.025,
    n_estimators=2000,
    min_samples_split=5,
    min_samples_leaf=2,
    max_depth=5,
    verbose=10,
)

clf.fit(X, y)

# pred = clf.predict(X_test)

# labels = sorted(set(pred))
# matrix = confusion_matrix(pred, y_test, labels=labels)
# matrix = pd.DataFrame(matrix, index=labels, columns=labels)

# report = classification_report(pred.tolist(), y_test.tolist(), labels=labels)
# print(report)

# sn.set(font_scale=0.8)  # for label size
# sn.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g')  # font size
# plt.show()

# import pdb; pdb.set_trace()
# exit()


dump(clf, 'C:\\Users\\CFI\\Desktop\\satf_training\\models\\training_data_GB_classifier.model')

import pdb; pdb.set_trace()
