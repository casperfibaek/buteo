import ee

#### must be run first/if no account is registered ###
### pairs with google account ### 

# ee.Authenticate()

ee.Initialize()
#%%
my_image = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20160701T155827_20160701T155852_011957_0126F0_E543')

# Get a download URL for an image.

aoi = [
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

#### sentinel 1 / 2 images have bands at different float values (16/32/64) ###
### these bands must either be removed or converted in order to export successfully ###

my_image = my_image.toUint16()

my_geometry = ee.Geometry.Polygon(aoi)


### exports to google drive associated with google account used to log in ###

task = ee.batch.Export.image.toDrive(image=my_image,  # an ee.Image object.
                                     region=my_geometry,  # an ee.Geometry object.
                                     description='mock_export',
                                     folder='export_test',
                                     fileNamePrefix='mock_export',
                                     scale=1000,
                                     crs='EPSG:4326')

task.start()
# %%
task.status()



