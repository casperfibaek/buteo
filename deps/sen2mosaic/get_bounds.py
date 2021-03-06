import geopandas as gpd

def get_bounds(input_geom):
    # tiles = '../../geometry/sentinel2_tiles_world.shp'
    tiles = 'C:/Users/caspe/Desktop/yellow/geometry/sentinel2_tiles_world.shp'
    tiles_geom = gpd.read_file(tiles)
    in_geom = gpd.read_file(input_geom)

    bounds = []
    for ig, geom in in_geom.iterrows():
        for it, tile in tiles_geom.iterrows():
            if geom['geometry'].intersects(tile['geometry']):
                bounds.append({ "name": tile['Name'], "bounds": list(tile['geometry'].bounds) })

    return bounds