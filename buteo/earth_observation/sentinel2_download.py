import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
from sen2mosaic.download import search, download, connectToAPI

folder = "C:/Users/caspe/Desktop/sne_mosaic/"

project_area = 'dk_wgs84.gpkg'
project_geom = gpd.read_file(folder + project_area)

tiles = '../geometry/sentinel2_tiles_world.shp'
tiles_geom = gpd.read_file(tiles)
tiles_dest = tiles_geom.to_crs(project_geom.crs)

data = []
data_bounds = []
for index_g, geom in project_geom.iterrows():
    for index_t, tile in tiles_geom.iterrows():
        if geom['geometry'].intersects(tile['geometry']):
            data.append(tile['Name'])
            data_bounds.append(list(tile['geometry'].bounds))

api_connection = connectToAPI('casperfibaek2', 'Goldfish12')

for tile in data:
    sdf = search(tile, level='2A', start='20210125', end='20210210', maxcloud=60, minsize=100.0)
    print(f"Found {len(sdf)} images of {tile}" )
    try:
        download(sdf, folder)
    except:
        print(f"No images of tile: {str(tile)}")

print("Finished downloading all images..")