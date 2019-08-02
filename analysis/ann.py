import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sn
import matplotlib.pyplot as plt
from joblib import dump


df = pd.read_csv('C:\\Users\\CFI\\Desktop\\satf_training\\training_data_feature_selected_standardized.csv', index_col=0)

X = df.drop(['DN', 'class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


def best_param(X, y):
    hidden_layer_sizes = [(1000, 1000)]
    activation = ['relu']
    alpha = [0.0001]
    solver = ['adam']
    max_iter = [1000]
    beta_1 = [0.9999]
    beta_2 = [0.7999]
    shuffle = [True]
    epsilon = [1e-1, 1e-2, 1e-3]

    random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
                   'activation': activation,
                   'alpha': alpha,
                   'max_iter': max_iter,
                   'beta_1': beta_1,
                   'beta_2': beta_2,
                   'shuffle': shuffle,
                   'epsilon': epsilon,
                   'solver': solver}

    estim = MLPClassifier()

    grid_search = GridSearchCV(
        estimator=estim,
        param_grid=random_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(X, y)

    return grid_search.best_params_

# bp = best_param(X, y)
# print(bp)
# import pdb; pdb.set_trace()


# logistic = 85.37% acc
# relu = 86.92% acc
clf = MLPClassifier(
    hidden_layer_sizes=(1000, 1000),
    alpha=0.0001,
    beta_1=0.9,
    beta_2=0.799,
    epsilon=0.1,
    solver='adam',
    activation='relu',
    shuffle=True,
    max_iter=1000,
    tol=0.000000001,
    verbose=10,
    random_state=42,
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"{'{0:.2f}'.format(accuracy_score(y_test, y_pred) * 100)}% acc")

clf = MLPClassifier(
    hidden_layer_sizes=(1000, 1000),
    alpha=0.0001,
    beta_1=0.9,
    beta_2=0.799,
    epsilon=0.1,
    solver='adam',
    activation='relu',
    shuffle=True,
    max_iter=1000,
    tol=0.000000001,
    # verbose=10,
    random_state=42,
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

labels = sorted(set(pred))
matrix = confusion_matrix(pred, y_test, labels=labels)
matrix = pd.DataFrame(matrix, index=labels, columns=labels)

report = classification_report(pred.tolist(), y_test.tolist(), labels=labels)
print(report)

sn.set(font_scale=0.8)  # for label size
sn.heatmap(matrix, annot=True, cmap="YlGnBu", fmt='g')  # font size
plt.show()

import pdb; pdb.set_trace()
exit()


dump(clf, 'C:\\Users\\CFI\\Desktop\\satf_training\\models\\training_data_MLP_classifier.model')

import pdb; pdb.set_trace()
