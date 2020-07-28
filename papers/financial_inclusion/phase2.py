import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.neural_network import BernoulliRBM
folder = '/mnt/c/Users/caspe/Desktop/Analysis/Phase2/vector/'

db_connection = sqlite3.connect(folder + 'features_perm_lights.sqlite')

df = pd.read_sql_query("SELECT * FROM features_perm_lights LIMIT -1;", db_connection)

no_fid = df.loc[:, df.columns != 'ogc_fid0']
no_fid = no_fid.loc[:, no_fid.columns != 'ogc_fid']
no_fid = no_fid.loc[:, no_fid.columns != 'fid']
no_fid = no_fid.loc[:, no_fid.columns != 'dn']
no_fid = no_fid.loc[:, no_fid.columns != 'DN']
no_fid = no_fid.loc[:, no_fid.columns != 'GEOMETRY']
no_fid = no_fid.loc[:, no_fid.columns != 'area']
no_fid = no_fid.loc[:, no_fid.columns != 'area_std']
no_fid = no_fid.loc[:, no_fid.columns != 'perimeter']
no_fid = no_fid.loc[:, no_fid.columns != 'perimeter_std']
no_fid = no_fid.loc[:, no_fid.columns != 'ipq']
no_fid = no_fid.fillna(0)

print('read shp')
rdf = pd.DataFrame()
# rdf['ogc_fid'] = df['ogc_fid']
rdf['dn'] = df['dn']
# rdf['spectr'] = SpectralClustering().fit_predict(no_fid)
rdf['kmeans'] = KMeans(n_clusters=25, n_jobs=-1, max_iter=1000, tol=0.00001).fit_predict(no_fid)
# print(DBSCAN(eps=2, min_samples=100).fit_predict(no_fid))
# import pdb; pdb.set_trace()
print('added columns')


out_connection = sqlite3.connect(folder + 'features_perm_lights_clusters.sqlite')
rdf.to_sql(name='features_perm_lights_clusters', con=out_connection, index=False)
