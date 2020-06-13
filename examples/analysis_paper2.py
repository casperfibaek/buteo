import sys; sys.path.append('..'); sys.path.append('../lib/')
import sqlite3
import pandas as pd
import numpy as np
import tensorflow
import keras
import matplotlib.pyplot as plt
import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from lib.ml_utils import best_param_random


db_engine = create_engine('sqlite:///C:\\Users\\caspe\\Desktop\\analysis_p2\\classification\\classification.sqlite') 
db_path = 'C:\\Users\\caspe\\Desktop\\analysis_p2\\classification\\training_data.sqlite'
target_path = 'C:\\Users\\caspe\\Desktop\\analysis_p2\\classification\\target_data.sqlite'

cnx = sqlite3.connect(db_path)
cnx_t = sqlite3.connect(target_path)

class_col = 'class'
seed = 42

df = pd.read_sql_query(f"SELECT DN, mad_mean, b4_mean, b8_mean, cxbb_mean, cxbb_min, cxbb_max,_ndvi_mean, _ndvi_stdev, nl_min, coh_mean, {class_col} FROM 'training_data';", cnx)
# df = pd.read_sql_query(f"SELECT DN, mad_mean, mad_stdev, mad_min, mad_max, b4_mean, b4_stdev, b4_min, b4_max, b8_mean, b8_stdev, b8_min, b8_max, cxbb_mean, cxbb_stdev, cxbb_min, cxbb_max, _ndvi_mean, _ndvi_stdev, _ndvi_min, _ndvi_max, nl_min, nl_max, nl_mean, ipq, z_perimeter, z_area, slo_mean, slo_stdev, slo_min, slo_max, bs_mean, bs_stdev, bs_min, bs_max, coh_mean, coh_stdev, coh_min, coh_max, {class_col} FROM 'training_data';", cnx)
# target = pd.read_sql_query(f"SELECT DN, mad_mean, mad_min, b8_mean, cxbb_mean, cxbb_min, ndvi_mean, ndvi_stdev, ndvi_max, nl_min, coh_mean, coh_stdev, coh_max FROM 'target_data';", cnx_t)
# target = target.fillna(0)
# df = df.dropna()

# df_pred = pd.DataFrame(target['dn'])
# target = target.drop(['dn'], axis=1)

X = df.drop([class_col], axis=1)
X = X.astype('float32')

y = df[class_col]
y = y.astype('int64')


class_count = y.nunique()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
dn = pd.DataFrame(X_test['dn'])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train.drop(['dn'], axis=1)
X_test = X_test.drop(['dn'], axis=1)
X = X.drop(['dn'], axis=1)

# sm = SMOTE(sampling_strategy='auto', k_neighbors=8, random_state=seed)
# X, y = sm.fit_sample(X, y)

# import statsmodels.api as sm
# import matplotlib.pyplot as plt

# corr = X.corr()
# sm.graphics.plot_corr(corr, xnames=list(corr.columns), normcolor=True)
# plt.show()

# import pdb; pdb.set_trace()
# exit()

sm = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=seed)
X_train, y_train = sm.fit_sample(X_train, y_train)


# params = best_param_random(X, y)

# {'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': 30, 'bootstrap': False}

# # estimator = RandomForestClassifier(n_estimators=2000, min_samples_split=2, min_samples_leaf=4, max_features='log2', max_depth=30, bootstrap=False)
# estimator = ExtraTreesClassifier(100)
# selector = RFECV(estimator, step=1, cv=3, verbose=2)

# selector = selector.fit(X, y)
# estimator.fit(X, y)

# support = selector.support_
# importance = estimator.feature_importances_

# import pdb; pdb.set_trace()

# exit()

# TensorBoard callback.
logdir = "C:/Users/caspe/Desktop/yellow/log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X.shape[1]
# n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.25))
model.add(Dense(class_count, activation='softmax'))

# # define model
# model = Sequential()
# model.add(Dense(50, activation='tanh', kernel_initializer='he_normal', input_shape=(n_features,)))
# model.add(Dropout(0.2))
# model.add(Dense(25, activation='tanh', kernel_initializer='he_normal'))
# model.add(Dropout(0.2))
# model.add(Dense(25, activation='tanh', kernel_initializer='he_normal'))
# model.add(Dropout(0.2))
# model.add(Dense(5, activation='tanh', kernel_initializer='he_normal'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))

# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# compile the model
model.compile(optimizer='adam', AMSGrad=True, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=300, verbose=1, batch_size=256, callbacks=tensorboard_callback)
# model.fit(X, y, epochs=1000, verbose=1, batch_size=1024)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: %.3f' % acc)

# y_pred = model.predict(target)
# y_prop = np.max(y_pred, axis=1)
# y_pred = np.argmax(y_pred, axis=1)

y_pred = model.predict(X_test)
y_prop = np.max(y_pred, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# import pdb; pdb.set_trace()

# df_pred['y_pred'] = y_pred
# df_pred['y_prop'] = y_prop
dn['y_pred'] = y_pred
dn['y_prop'] = y_prop

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# df_pred.to_sql('predicted', con=db_engine)
import pdb; pdb.set_trace()



# model = RandomForestClassifier(n_estimators=800, min_samples_split=6, min_samples_leaf=4, max_features='sqrt', max_depth=50)
# model.fit(X, y)

# pred = model.predict(target)
# dn['class'] = pred

# dn.to_sql('classification', db_engine)


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


