import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.neural_network import BernoulliRBM
folder = '/mnt/c/Users/caspe/Desktop/Analysis/Phase2/vector/'

db_connection = sqlite3.connect(folder + 'phase2_features_reduced.sqlite')

df = pd.read_sql_query("SELECT * FROM phase2_features_reduced LIMIT 100;", db_connection)

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

import pdb; pdb.set_trace()
print('read shp')
rdf = pd.DataFrame()
rdf['ogc_fid'] = df['ogc_fid']
# rdf['spectr'] = SpectralClustering().fit_predict(no_fid)
rdf['bernou'] = BernoulliRBM(verbose=True).fit_predict(no_fid)
import pdb; pdb.set_trace()
rdf['kmeans'] = KMeans(n_clusters=16, n_jobs=-1).fit_predict(no_fid)
# print(DBSCAN(eps=2, min_samples=100).fit_predict(no_fid))
# import pdb; pdb.set_trace()
print('added columns')


out_connection = sqlite3.connect(folder + 'phase2_features_reduced_clusters.sqlite')
rdf.to_sql(name='phase2_features_reduced_clusters', con=out_connection, index=False)
