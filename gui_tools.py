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
                    "display_name": "Input Raster",
                    "type": "file_browse",
                    "tooltip": "The raster to which the borders will be added",
                }
            },
            {
                "out_path": {
                    "display_name": "Output Raster",
                    "type": "file_save",
                    "tooltip": "The path to the output raster",
                    "default_extension": "tif",
                }
            },
            {
                "border_size": {
                    "display_name": "Border Size",
                    "type": "number",
                    "default": 100,
                    "tooltip": "The size of the border to add",
                }
            },
            {
                "border_size_unit": {
                    "display_name": "Border Size Unit",
                    "type": "radio",
                    "tooltip": "Border size is specified in  either pixels or map units.",
                    "options": [
                        {
                            "label": "Pixels",
                            "key": "pixels",
                            "value": "px",
                            "default": True,
                        },
                        {"label": "Map units", "key": "map_units", "value": "m"},
                    ],
                }
            },
            {
                "border_value": {
                    "display_name": "Border Value",
                    "type": "number",
                    "default": 0,
                    "tooltip": "The pixel values of the border",
                }
            },
            {
                "overwrite": {
                    "display_name": "Overwrite",
                    "type": "boolean",
                    "default": True,
                    "tooltip": "Overwrite existing files",
                }
            },
        ],
    },
    "Align Rasters": {
        "description": "Align rasters using a reference raster",
        "function_path": align_rasters,
        "parameters": [
            {
                "input_rasters": {
                    "display_name": "Input Rasters",
                    "type": "file_browse_multiple",
                    "tooltip": "The rasters which will be aligned",
                }
            },
            {
                "out_path": {
                    "display_name": "Output Raster",
                    "type": "folder_save",
                    "tooltip": "The folder where the aligned rasters will be saved",
                }
            },
            {
                "master": {
                    "display_name": "Master Raster",
                    "type": "file_browse",
                    "tooltip": "The master_raster to which the other rasters will be aligned.",
                }
            },
            {
                "postfix": {
                    "display_name": "Postfix",
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
                    "display_name": "SciHub Username",
                    "type": "string",
                    "tooltip": "The username to use for the Sentinel Hub API.",
                    "default": "casperfibaek",
                }
            },
            {
                "scihub_password": {
                    "display_name": "SciHub Password",
                    "type": "password",
                    "tooltip": "The password to use for the Sentinel Hub API.",
                    "default": "Goldfish12",
                }
            },
            {
                "onda_username": {
                    "display_name": "Onda Username",
                    "type": "string",
                    "tooltip": "The username to use for the Onda API.",
                    "default": "cfi@niras.dk",
                }
            },
            {
                "onda_password": {
                    "display_name": "Onda Password",
                    "type": "password",
                    "tooltip": "The password to use for the Onda API.",
                    "default": "Goldfish12!@",
                }
            },
            {
                "out_path": {
                    "display_name": "Output Folder",
                    "type": "folder_save",
                    "tooltip": "The folder where the downloaded data will be saved.",
                },
            },
            {
                "aoi_vector": {
                    "display_name": "AOI Vector",
                    "type": "file_browse",
                    "tooltip": "The vector file containing the area of interest.",
                },
            },
            {
                "start_date": {
                    "display_name": "Start Date",
                    "type": "date_year",
                    "tooltip": "The start date of the data to be downloaded.",
                    "default_date": "days_ago_14",
                },
            },
            {
                "end_date": {
                    "display_name": "End Date",
                    "type": "date_year",
                    "tooltip": "The end date of the data to be downloaded.",
                    "default_date": "today",
                },
            },
            {
                "cloud_cover": {
                    "display_name": "Cloud Cover",
                    "type": "number",
                    "tooltip": "The maximum cloud cover allowed for the data to be downloaded.",
                    "default": 20,
                },
            },
            {
                "tile_id": {
                    "display_name": "Tile ID",
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
