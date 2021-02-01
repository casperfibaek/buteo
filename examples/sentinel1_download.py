import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
from sen1mosaic.download import search, download, connectToAPI

project_area = '../geometry/studyArea100mBuffer.gpkg'
project_geom = gpd.read_file(project_area)
project_geom_wgs = project_geom.to_crs('EPSG:4326')

tiles = '../geometry/sentinel2_tiles_world.shp'
tiles_geom = gpd.read_file(tiles)
tiles_dest = tiles_geom.to_crs(project_geom.crs)

data = []
data_bounds = []
for index_g, geom in project_geom_wgs.iterrows():
    for index_t, tile in tiles_geom.iterrows():
        if geom['geometry'].intersects(tile['geometry']):
            data.append(tile['Name'])
            data_bounds.append(list(tile['geometry'].bounds))

api_connection = connectToAPI('casperfibaek', 'Goldfish12')

base = "/home/cfi/data/sentinel1_paper2/"

for index, tile in enumerate(data):
    # import pdb; pdb.set_trace()
    sdf_grd = search(data_bounds[index], api_connection, start='20191115', end='20191215', producttype='GRD', direction='ASCENDING')
    download(sdf_grd, api_connection, base + 'ascending/')
    sdf_slc = search(data_bounds[index], api_connection, start='20200315', end='20200331', producttype='SLC', direction='ASCENDING')
    download(sdf_slc, api_connection, base + 'ascending')
    sdf_grd = search(data_bounds[index], api_connection, start='20191115', end='20191215', producttype='GRD', direction='DESCENDING')
    download(sdf_grd, api_connection, base + 'descending/')
    sdf_slc = search(data_bounds[index], api_connection, start='20200315', end='20200331', producttype='SLC', direction='DESCENDING')
    download(sdf_slc, api_connection, base + 'descending')
