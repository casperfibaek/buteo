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
import pdb

db_engine = create_engine('sqlite:///C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\classification.sqlite') 
db_path = 'C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\training_data.sqlite'
# target_path = 'C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\target_data.sqlite'

cnx = sqlite3.connect(db_path)
# cnx_t = sqlite3.connect(target_path)

class_col = 'class'
seed = 42

df = pd.read_sql_query(f"SELECT DN, mad_mean, b4_mean, b4_stdev, b8_mean, b8_stdev, cxb_mean, cxb_stdev, cxbb_mean, cxbb_stdev, ndvi_mean, ndvi_stdev, nl_mean, {class_col} FROM 'training_data';", cnx)
# target = pd.read_sql_query("SELECT DN, mad_mean, b4_mean, b4_stdev, b8_mean, b8_stdev, cxb_mean, cxb_stdev, cxbb_mean, cxbb_stdev, ndvi_mean, ndvi_stdev, nl_mean FROM 'target_data';", cnx_t)

# target = target.fillna(0)
df = df.dropna()

# df_pred = pd.DataFrame(target['dn'])
# target = target.drop(['dn'], axis=1)

X = df.drop([class_col], axis=1)
X = X.astype('float32')

y = df[class_col]

# import pdb; pdb.set_trace()

class_count = y.nunique()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
dn = pd.DataFrame(X_test['dn'])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train.drop(['dn'], axis=1)
X_test = X_test.drop(['dn'], axis=1)
X = X.drop(['dn'], axis=1)

# sm = SMOTE(sampling_strategy='auto', k_neighbors=16, random_state=seed)
# X, y = sm.fit_sample(X, y)

sm = SMOTE(sampling_strategy='auto', k_neighbors=16, random_state=seed)
X_train, y_train = sm.fit_sample(X_train, y_train)

# pdb.set_trace()

# TensorBoard callback.
logdir = "C:/Users/caspe/Desktop/yellow/log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
# n_features = X.shape[1]
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(100, activation='sigmoid', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dropout(0.2))
model.add(Dense(50, activation='sigmoid', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='sigmoid', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(class_count, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, verbose=1, batch_size=512, callbacks=tensorboard_callback)
# model.fit(X, y, epochs=1000, verbose=1)

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

import pdb; pdb.set_trace()
# df_pred.to_sql('predicted', con=db_engine)



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


