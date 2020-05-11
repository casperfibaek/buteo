import sqlite3
import pandas as pd
import numpy as np
import tensorflow
import keras
import matplotlib.pyplot as plt
import datetime
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
import pdb

db_engine = create_engine('sqlite:///C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\kmeans.sqlite') 
db_path = 'C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\training_data_kmeans.sqlite'
target_path = 'C:\\Users\\caspe\\Desktop\\Ghana_data\\classification\\target_kmeans.sqlite'

cnx = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT dn, cxbb_mean, cxbb_stdev, nl_mean FROM 'training_data_kmeans';", cnx)

dn = pd.DataFrame(df['dn'])
X = df.drop(['dn'], axis=1)

kmeans = KMeans(n_clusters=25, random_state=42).fit_predict(X)
dn['kmeans'] = kmeans

dn.to_sql('kmeans', con=db_engine)

pdb.set_trace()

