import os
import sys; sys.path.append("../")
import pandas as pd
import math
import tqdm
import requests
import numpy as np
import buteo as beo
from osgeo import gdal


class GlobalMercator(object):

    def __init__(self, tileSize=256):
        "Initialize the TMS Global Mercator pyramid"
        self.tileSize = tileSize
        self.initialResolution = 2 * math.pi * 6378137 / self.tileSize
        # 156543.03392804062 for tileSize 256 pixels
        self.originShift = 2 * math.pi * 6378137 / 2.0
        # 20037508.342789244

    def LatLonToMeters(self, lat, lon):
        "Converts given lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913"

        mx = lon * self.originShift / 180.0
        my = math.log( math.tan((90 + lat) * math.pi / 360.0 )) / (math.pi / 180.0)

        my = my * self.originShift / 180.0
        return mx, my

    def MetersToLatLon(self, mx, my):
        "Converts XY point from Spherical Mercator EPSG:900913 to lat/lon in WGS84 Datum"

        lon = (mx / self.originShift) * 180.0
        lat = (my / self.originShift) * 180.0

        lat = 180 / math.pi * (2 * math.atan( math.exp( lat * math.pi / 180.0)) - math.pi / 2.0)
        return lat, lon

    def PixelsToMeters(self, px, py, zoom):
        "Converts pixel coordinates in given zoom level of pyramid to EPSG:900913"

        res = self.Resolution( zoom )
        mx = px * res - self.originShift
        my = py * res - self.originShift
        return mx, my
        
    def MetersToPixels(self, mx, my, zoom):
        "Converts EPSG:900913 to pyramid pixel coordinates in given zoom level"
                
        res = self.Resolution( zoom )
        px = (mx + self.originShift) / res
        py = (my + self.originShift) / res
        return px, py
    
    def PixelsToTile(self, px, py):
        "Returns a tile covering region in given pixel coordinates"

        tx = int( math.ceil( px / float(self.tileSize) ) - 1 )
        ty = int( math.ceil( py / float(self.tileSize) ) - 1 )
        return tx, ty

    def PixelsToRaster(self, px, py, zoom):
        "Move the origin of pixel coordinates to top-left corner"
        
        mapSize = self.tileSize << zoom
        return px, mapSize - py
        
    def MetersToTile(self, mx, my, zoom):
        "Returns tile for given mercator coordinates"
        
        px, py = self.MetersToPixels(mx, my, zoom)
        return self.PixelsToTile(px, py)

    def TileBounds(self, tx, ty, zoom):
        "Returns bounds of the given tile in EPSG:900913 coordinates"
        
        minx, miny = self.PixelsToMeters( tx*self.tileSize, ty*self.tileSize, zoom )
        maxx, maxy = self.PixelsToMeters( (tx+1)*self.tileSize, (ty+1)*self.tileSize, zoom )
        return ( minx, miny, maxx, maxy )

    def TileLatLonBounds(self, tx, ty, zoom ):
        "Returns bounds of the given tile in latutude/longitude using WGS84 datum"

        bounds = self.TileBounds(tx, ty, zoom)
        minLat, minLon = self.MetersToLatLon(bounds[0], bounds[1])
        maxLat, maxLon = self.MetersToLatLon(bounds[2], bounds[3])
         
        return ( minLat, minLon, maxLat, maxLon )
        
    def Resolution(self, zoom):
        "Resolution (meters/pixel) for given zoom level (measured at Equator)"
        
        # return (2 * math.pi * 6378137) / (self.tileSize * 2**zoom)
        return self.initialResolution / (2**zoom)
        
    def ZoomForPixelSize(self, pixelSize):
        "Maximal scaledown zoom of the pyramid closest to the pixelSize."
        
        for i in range(30):
            if pixelSize > self.Resolution(i):
                return i-1 if i!=0 else 0 # We don't want to scale up

    def GoogleTile(self, tx, ty, zoom):
        "Converts TMS tile coordinates to Google Tile coordinates"
        
        # coordinate origin is moved from bottom-left to top-left corner of the extent
        return tx, (2**zoom - 1) - ty
    
    def GoogleToTMSTile(self, tx, ty, zoom):
        "Converts Google Tile coordinates to TMS tile coordinates"
        
        # coordinate origin is moved from top-left to bottom-left corner of the extent
        # The opposite of this: return tx, (2**zoom - 1) - ty
        return tx, (2**zoom - 1) - ty

    def QuadTree(self, tx, ty, zoom):
        "Converts TMS tile coordinates to Microsoft QuadTree"
        
        quadKey = ""
        ty = (2**zoom - 1) - ty
        for i in range(zoom, 0, -1):
            digit = 0
            mask = 1 << (i-1)
            if (tx & mask) != 0:
                digit += 1
            if (ty & mask) != 0:
                digit += 2
            quadKey += str(digit)
            
        return quadKey


def latlon_to_tilexy(lat: float, lon: float, z: int):
    """
    Convert lat/lon to tile x/y

    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    z : int
        Zoom level

    Returns
    -------
    tuple(list, tuple)
        List of tiles, and tuple of x/y
    """
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    
    x0 = round(x)
    y0 = round(y)

    x1 = x0 - 1 if x < x0 else x0 - 1
    y1 = y0 - 1 if y < y0 else y0 - 1

    squares = [
        (min(x0, x1), min(y0, y1), z),
        (min(x0, x1), max(y0, y1), z),
        (max(x0, x1), max(y0, y1), z),
        (max(x0, x1), min(y0, y1), z)
    ]

    if round(x) == max(x0, x1):
        x = 0.5 + ((x - max(x0, x1)) / 2.0)
    else:
        x = (x - min(x0, x1)) / 2.0

    if round(y) == max(y0, y1):
        y = 0.5 + ((y - max(y0, y1)) / 2.0)
    else:
        y = (y - min(y0, y1)) / 2.0

    return squares, (x, y)


def process_latlng(latlng, FOLDER_OUT):
    convertor = GlobalMercator(tileSize=256)

    for BING in [True, False]:
        for px_size in [0.5, 2.0, 5.0]:
            fid, x, y = latlng

            z = convertor.ZoomForPixelSize(px_size)

            source = 0 if BING else 1

            outname = f"i{int(fid)}_x{round(x, 6)}_y{round(y, 6)}_z{z}_s{source}.tif"

            if os.path.isfile(os.path.join(FOLDER_OUT, outname)):
                continue

            squares, (og_x, og_y) = latlon_to_tilexy(y, x, z)

            total_image = np.zeros((3, 512, 512), dtype=np.uint8)

            pixel_sizes = []
            x_min, y_min, x_max, y_max = None, None, None, None

            skip = False
            for i, s in enumerate(squares):
                if skip:
                    continue
                x_tile, y_tile, z_tile = s

                tms_x_tile, tms_y_tile = convertor.GoogleToTMSTile(x_tile, y_tile, z_tile)
                minx, miny, maxx, maxy = convertor.TileBounds(tms_x_tile, tms_y_tile, z_tile)

                if x_min is None:
                    x_min, y_min, x_max, y_max = minx, miny, maxx, maxy
                else:
                    x_min = min(x_min, minx)
                    y_min = min(y_min, miny)
                    x_max = max(x_max, maxx)
                    y_max = max(y_max, maxy)

                pixel_sizes.append((maxx - minx) / 256)
                pixel_sizes.append((maxy - miny) / 256)

                if BING:
                    q = convertor.QuadTree(tms_x_tile, tms_y_tile, z_tile)
                    url = f"https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
                    tmp_path = os.path.abspath(os.path.join("./tmp", f"{x_tile}_{y_tile}_{z_tile}.jpeg"))
                else:
                    url = f"https://mt1.google.com/vt/lyrs=s&x={x_tile}&y={y_tile}&z={z_tile}"
                    tmp_path = os.path.abspath(os.path.join("./tmp", f"{x_tile}_{y_tile}_{z_tile}.png"))

                response = requests.get(url)

                # if respose is not ok, skip
                if response.status_code != 200:
                    skip = True
                    continue

                with open(f'{tmp_path}', 'wb') as file:
                    file.write(response.content)

                try:
                    # Read the raster band as numpy array
                    array = gdal.Open(tmp_path).ReadAsArray()

                    if i == 0:
                        total_image[:, 0:256, 0:256] = array
                    elif i == 3:
                        total_image[:, 0:256, 256:512] = array
                    elif i == 2:
                        total_image[:, 256:512, 256:512] = array
                    elif i == 1:
                        total_image[:, 256:512, 0:256] = array
                    
                except:
                    skip = True
                finally:
                    os.remove(tmp_path)

            if skip:
                continue

            total_image = total_image.transpose(1, 2, 0)

            raster_tmp = beo.raster_create_from_array(
                total_image,
                out_path=None,
                pixel_size=np.array(pixel_sizes).mean(),
                x_min=x_min,
                y_max=y_max,
            )

            label_y = max(0, round(og_y * 512) - 128)
            label_x = max(0, round(og_x * 512) - 128)

            beo.array_to_raster(
                beo.raster_to_array(raster_tmp, pixel_offsets=[label_x, label_y, 256, 256]),
                out_path=os.path.join(FOLDER_OUT, outname),
                reference=raster_tmp,
                pixel_offsets=[label_x, label_y, 256, 256],
            )

            beo.delete_dataset_if_in_memory(raster_tmp)

    return True


if __name__ == "__main__":
    import multiprocessing as mp
    from tqdm import tqdm

    FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/"
    FOLDER_OUT = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/scraped_schools_south-america-test/"
    MAX_IMAGES = 10000
    PROCESSES = 8

    csv = pd.read_csv(os.path.join(FOLDER, "south-america_schools_osm_sampled_10k.csv"), encoding="latin-1")

    np.random.seed(42)

    latlng = csv[["fid", "lon", "lat"]].to_numpy()
    latlng = latlng[np.random.permutation(len(latlng))]
    latlng = latlng[:MAX_IMAGES]

    bar = tqdm(total=len(latlng))

    def update_progress_bar(_result):
        bar.update()

    pool = mp.Pool(8)
    for i in range(len(latlng)):
        pool.apply_async(process_latlng, args=(latlng[i], FOLDER_OUT), callback=update_progress_bar)

    pool.close()
    pool.join()
    bar.close()

    print("Done!")
    print("Finding straglers...")

    for c in tqdm(latlng, total=len(latlng)):
        process_latlng(c, FOLDER_OUT)
