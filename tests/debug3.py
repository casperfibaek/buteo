import os
import sys; sys.path.append("../")
import buteo as beo

BANDS = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
MOD2DIC = {
    'Forest Closed Acquatic': 'Wetlands',
    'Forest Open Acquatic': 'Wetlands',
    'River': 'Natural Waterbodies',
    'Tree Orchard Large': 'Cultivated and Managed Terrestrial Area(s)',
    'Tree Orchard Small': 'Cultivated and Managed Terrestrial Area(s)',
    'Herbaceous Crops Small': 'Cultivated and Managed Terrestrial Area(s)',
    'Forest Closed': 'Natural vegetation',
    'Forest Open': 'Natural vegetation',
    'Shrubs Open': 'Natural vegetation',
    'Grasslands Open': 'Natural vegetation',
    'Tree Plantations Small': 'Cultivated and Managed Terrestrial Area(s)',
    'Shrubs Closed': 'Natural vegetation',
    'Grasslands Closed': 'Natural vegetation',
    'Agriculture flooded - Graminoid Small': 'Acquatic agriculture',
    'Shrubs Open Acquatic': 'Wetlands',
    'Grasslands Open Acquatic': 'Wetlands',
    'Shrubs Closed Acquatic': 'Wetlands',
    'Grasslands Closed Acquatic': 'Wetlands',
    'Urban - Built Up': 'Urban',
    'Urban - Not Built Up': 'Urban',
    'Standing Artificial Waterbodies': 'Artificial Waterbodies',
    'Lake': 'Natural Waterbodies',
    'Bare': 'Bare',
    'Tree Plantations Large': 'Cultivated and Managed Terrestrial Area(s)',
}

CLASSES = {
    'Acquatic agriculture': 0,
    'Artificial Waterbodies': 1,
    'Bare': 2,
    'Cultivated and Managed Terrestrial Area(s)': 3,
    'Natural Waterbodies': 4,
    'Natural vegetation': 5,
    'Urban': 6,
    'Wetlands': 7,
}

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/test_data/"

GROUND_TRUTH = os.path.join(FOLDER, "WAF_04_lc_b.shp")
REF_31PCN = {'path': os.path.join(FOLDER, "31PCN", "S2A", "B04.tif"), 'name': '31PCN' }
REF_31PDN = {'path': os.path.join(FOLDER, "31PDN", "S2A", "B04.tif"), 'name': '31PDN' }
REF_31PDP = {'path': os.path.join(FOLDER, "31PDP", "S2A", "B04.tif"), 'name': '31PDP' }

if __name__ == '__main__':
    ground_truth = beo.vector_copy(GROUND_TRUTH)
    attribute_header, attribute_table = beo.vector_get_attribute_table(ground_truth, include_fids=True)

    update_header = ["class", "fid"]
    update_attributes = []

    idx_bname = attribute_header.index('b_name')
    idx_fid = attribute_header.index('fid')
    for feature in attribute_table:
        b_name = feature[idx_bname].strip()
        fid = feature[idx_fid]

        simple_name = MOD2DIC[b_name]
        class_id = CLASSES[simple_name]
        update_attributes.append([class_id, fid])

    beo.vector_add_field(ground_truth, "class", "integer")
    beo.vector_set_attribute_table(ground_truth, update_header, update_attributes, match='fid')

    for ref in [REF_31PCN, REF_31PDN, REF_31PDP]:
        beo.vector_rasterize(
            ground_truth,
            pixel_size=ref['path'],
            extent=ref['path'],
            projection=ref['path'],
            fill_value=100,
            attribute='class',
            out_path=os.path.join(FOLDER, f"{ref['name']}_groundtruth.tif"),
        )
