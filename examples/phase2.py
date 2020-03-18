import sqlite3
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

folder = 'C:/Users/caspe/Desktop/Analysis/Phase2/'

# db_connection = sqlite3.connect(folder + 'subset_zonal.sqlite')

limit = 100
df = gpd.read_file(folder + 'subset_zonal.sqlite')
# df = pd.read_sql_query("select 'tbl_name' from sqlite_master where type = 'table'", db_connection)
import pdb; pdb.set_trace()
# df = pd.read_sql_query(f"SELECT * FROM subset_zonal LIMIT {limit};", db_connection)


geometry = df['WKT_GEOMETRY'].values
fids = df['ogc_fid'].values

del df['ogc_fid']
del df['WKT_GEOMETRY']

dbscan = DBSCAN(n_jobs=-1, eps=2, min_samples=10, verbose=True).fit_predict(df)
aggclu = AgglomerativeClustering(n_clusters=10, verbose=True).fit_predict(df)
kmeans = KMeans(n_clusters=10, n_jobs=-1, verbose=True).fit_predict(df)

df = None

rdc = pd.DataFrame()

# dfc['clu_optic'] = optic
rdc['ogc_fid'] = fids
rdc['WKT_GEOMETRY'] = geometry
rdc['clu_dbsca'] = dbscan
rdc['clu_aggcl'] = dbscan
rdc['clu_kmean'] = dbscan

rdc.to_csv(folder + 'clusters.csv', index=False)
