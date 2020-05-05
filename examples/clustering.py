import sqlite3
import pandas as pd
import hdbscan
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np

db_engine = create_engine('sqlite:///C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\classification.sqlite') 
db_path = 'C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\training_data.sqlite'
target_path = 'C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\target_data.sqlite'

cnx = sqlite3.connect(db_path)
cnx_target = sqlite3.connect(target_path)

df = pd.read_sql_query("SELECT mad_mean, cxb_mean, cxb_stdev, ndvi_mean, nl_mean, class FROM 'training_data';", cnx)
target = pd.read_sql_query("SELECT mad_mean, cxb_mean, cxb_stdev, ndvi_mean, nl_mean FROM 'target_data';", cnx_target)
dn = pd.read_sql_query("SELECT dn FROM 'target_data';", cnx_target)

X = df.drop(['class'], axis=1)
y = df['class']


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)

params = {'n_estimators': 800, 'min_samples_split': 6, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': True}

# model = ExtraTreesClassifier(100)
test_model = RandomForestClassifier(n_estimators=800, min_samples_split=6, min_samples_leaf=4, max_features='sqrt', max_depth=50)
test_model.fit(X_train, y_train)

y_pred = test_model.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

model = RandomForestClassifier(n_estimators=800, min_samples_split=6, min_samples_leaf=4, max_features='sqrt', max_depth=50)
model.fit(X, y)

pred = model.predict(target)
dn['class'] = pred

dn.to_sql('classification', db_engine)


# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(10).plot(kind='barh')
# plt.show()

# selector = RFECV(estimator, step=1, cv=3, verbose=2)
# selector = selector.fit(X, y)

# headers_masked = []
# for nr, val in enumerate(selector.support_):
#     if val is False:
#         headers_masked.append(X.columns[nr])

# masked = X.drop(headers_masked, axis=1)


