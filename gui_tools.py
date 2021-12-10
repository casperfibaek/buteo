import datetime
from buteo.raster.borders import add_border_to_raster
from buteo.raster.align import align_rasters
from buteo.earth_observation.download_sentinel import download_s2_tile


tools = {
    "Add Borders": {
        "description": "Add borders to an image",
        "function_path": add_border_to_raster,
        "parameters": [
            {
                "input_raster": {
                    "type": "file_browse",
                    "tooltip": "The raster to which the borders will be added",
                }
            },
            {
                "out_path": {
                    "type": "file_save",
                    # "default_extension": "tif",
                }
            },
            {"border_size": {"type": "number", "default": 1}},
            {
                "border_size_unit_px": {
                    "type": "boolean",
                    "default": True,
                    "tooltip": "Border size in pixels (True) or in map units (False)",
                }
            },
            {"border_value": {"type": "number", "default": 0}},
            {"overwrite": {"type": "boolean", "default": True}},
        ],
    },
    "Align Rasters": {
        "description": "Align rasters using a reference raster",
        "function_path": align_rasters,
        "parameters": [
            {
                "input_rasters": {
                    "type": "file_browse_multiple",
                    "tooltip": "The rasters which will be aligned",
                }
            },
            {
                "out_path": {
                    "type": "folder_save",
                }
            },
            {
                "master": {
                    "type": "file_browse",
                    "tooltip": "The master_raster to which the other rasters will be aligned.",
                }
            },
            {
                "postfix": {
                    "type": "string",
                    "default": "_aligned",
                    "tooltip": "The postfix to be added to the output rasters.",
                }
            },
        ],
    },
    "Download Sentinel 2": {
        "description": "Download Sentinel 2 data",
        "function_path": download_s2_tile,
        "parameters": [
            {
                "scihub_username": {
                    "type": "string",
                    "tooltip": "The username to use for the Sentinel Hub API.",
                    "default": "casperfibaek",
                }
            },
            {
                "scihub_password": {
                    "type": "string",
                    "tooltip": "The password to use for the Sentinel Hub API.",
                    "default": "Goldfish12",
                }
            },
            {
                "onda_username": {
                    "type": "string",
                    "tooltip": "The username to use for the Onda API.",
                    "default": "cfi@niras.dk",
                }
            },
            {
                "onda_password": {
                    "type": "string",
                    "tooltip": "The password to use for the Onda API.",
                    "default": "Goldfish12!@",
                }
            },
            {
                "out_path": {
                    "type": "folder_save",
                    "tooltip": "The folder where the downloaded data will be saved.",
                },
            },
            {
                "aoi_vector": {
                    "type": "file_browse",
                    "tooltip": "The vector file containing the area of interest.",
                },
            },
            {
                "start_date": {
                    "type": "date_year",
                    "tooltip": "The start date of the data to be downloaded.",
                    "default_date": "days_ago_14",
                },
            },
            {
                "end_date": {
                    "type": "date_year",
                    "tooltip": "The end date of the data to be downloaded.",
                    "default_date": "today",
                },
            },
            {
                "cloud_cover": {
                    "type": "number",
                    "tooltip": "The maximum cloud cover allowed for the data to be downloaded.",
                    "default": 20,
                },
            },
            {
                "tile_id": {
                    "type": "string",
                    "tooltip": "Optional. Specify a tile ID to download only one tile.",
                    "default": "",
                }
            },
        ],
    },
}

# scihub_username,
# scihub_password,
# onda_username,
# onda_password,
# destination,
# aoi_vector,
# date_start="20200601",
# date_end="20210101",
# clouds=10,
# min_size=100,
