#%%
import ee
ee.Initialize()
#%%
image = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20160701T155827_20160701T155852_011957_0126F0_E543')

test = ee.Image('USGS/SRTMGL1_003')
# Get a download URL for an image.

study_area = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "MINX": 804123.2211340084,
        "MINY": 7343073.665317697,
        "MAXX": 864410.690564305,
        "MAXY": 7410611.773716191,
        "CNTX": 834266.9558491567,
        "CNTY": 7376842.719516944,
        "AREA": 4071701645.454233,
        "PERIM": 255651.15565758012,
        "HEIGHT": 67538.10839849338,
        "WIDTH": 60287.46943029668
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [
              21.727211420427153,
              66.06119136391337
            ],
            [
              23.046035433261228,
              65.99764795269222
            ],
            [
              23.241327459580265,
              66.59743936639583
            ],
            [
              21.891167266431697,
              66.66281400552248
            ],
            [
              21.727211420427153,
              66.06119136391337
            ]
          ]
        ]
      }
    }
  ]
}

geom = ee.Geometry(study_area)
#%%
import ipygee as ui

Map = ui.Map()

Map.show()








# %%
Map.show()
# %%
