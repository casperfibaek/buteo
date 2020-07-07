import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
from sen2mosaic.download import search, download, connectToAPI


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

for tile in data:
    cloud_search = 5
    min_images = 10
    downloaded = False
    sdf = None

    while cloud_search <= 100:
        sdf = search(tile, level='2A', start='20190901', end='20191101', maxcloud=cloud_search, minsize=100.0)
        
        if len(sdf) >= min_images:
            print(f"Found {len(sdf)} images of {tile} @ {cloud_search}% cloud cover - downloading.." )
            download(sdf, '/home/cfi/data/sentinel2_paper2/')
            downloaded = True
            break

        print(f'Found too few images of {tile} @ {cloud_search}% cloud cover - skipping..')
        cloud_search += 5
    
    if downloaded == False:
        download(sdf, '/home/cfi/data/sentinel2_paper2/')

print("Finished downloading all images..")