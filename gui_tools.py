from buteo.raster.borders import add_border_to_raster
from buteo.raster.align import align_rasters
from buteo.raster.clip import clip_raster
from buteo.earth_observation.download_sentinel import download_s2_tile
from buteo.earth_observation.s2_mosaic import mosaic_tile_s2


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
    "Clip Rasters": {
        "description": "Clips rasters using a reference",
        "function_path": clip_raster,
        "parameters": [
            {
                "raster": {
                    "display_name": "Input Rasters",
                    "type": "file_browse_multiple",
                    "tooltip": "The rasters to be clipped",
                }
            },
            {
                "clip_geom": {
                    "display_name": "Clip Reference",
                    "type": "file_browse",
                    "tool_tip": "The reference to clip the raster. Can be a vector or another raster.",
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
                "resample_alg": {
                    "display_name": "Resample Algorithm",
                    "type": "dropdown",
                    "tooltip": "The resampling algorithm to use when resampling the raster.",
                    "options": [
                        {
                            "label": "Nearest",
                            "value": "nearest",
                            "default": True,
                        },
                        {
                            "label": "Bilinear",
                            "value": "bilinear",
                        },
                        {
                            "label": "Average",
                            "value": "average",
                        },
                        {
                            "label": "Sum",
                            "value": "sum",
                        },
                    ],
                }
            },
            {
                "crop_to_geom": {
                    "display_name": "Crop to Geometry",
                    "type": "boolean",
                    "default": True,
                    "tooltip": "If true, the raster will be cropped to the geometry of the clip_geom.",
                }
            },
            {
                "adjust_bbox": {
                    "display_name": "Adj. Bounding Box",
                    "type": "boolean",
                    "default": False,
                    "tooltip": "if true, the bounding box of the raster will be adjusted to the extent of the reference.",
                }
            },
            {
                "all_touch": {
                    "display_name": "All Touch",
                    "type": "boolean",
                    "default": False,
                    "tooltip": "If true, all the pixels touching the reference will be included.",
                }
            },
            {
                "postfix": {
                    "display_name": "Postfix",
                    "type": "string",
                    "default": "_clipped",
                    "tooltip": "The postfix to be added to the output rasters.",
                }
            },
            {
                "prefix": {
                    "display_name": "Prefix",
                    "type": "string",
                    "default": "",
                    "tooltip": "The prefix to be added to the output rasters.",
                }
            },
        ],
    },
    "Sentinel 2 - Download": {
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
                "producttype": {
                    "display_name": "Process. Level",
                    "type": "radio",
                    "tooltip": "The processing level of the data to be downloaded.",
                    "options": [
                        {
                            "label": "Level 2A",
                            "key": "level2",
                            "value": "S2MSI2A",
                            "default": True,
                        },
                        {
                            "label": "Level 1C",
                            "key": "level1",
                            "value": "S2MSI1C",
                        },
                    ],
                }
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
    "Sentinel 2 - Mosaic": {
        "description": "Mosaics tiles of Sentinel 2 data",
        "function_path": mosaic_tile_s2,
        "parameters": [
            {
                "s2_files_folder": {
                    "display_name": "S2 Files Folder",
                    "type": "folder_save",
                    "tooltip": "The folder containing the Sentinel 2 files.",
                }
            },
            {
                "out_path": {
                    "display_name": "Output Folder",
                    "type": "folder_save",
                    "tooltip": "The folder where the mosaic tiles will be saved.",
                }
            },
            {
                "ideal_date": {
                    "display_name": "Ideal Date",
                    "type": "date_year",
                    "tooltip": "The start date of the data to be downloaded.",
                    "tooltip": "The ideal central date for the mosaic.",
                    "default_date": "days_ago_14",
                },
            },
            {
                "max_time_delta": {
                    "display_name": "Time Delta (days)",
                    "type": "number",
                    "tooltip": "The maximum time delta in days allowed for included rasters.",
                    "default": 60,
                },
            },
            {
                "max_images": {
                    "display_name": "Maximum Images",
                    "type": "number",
                    "tooltip": "The maximum included images in a mosaic.",
                    "default": 6,
                },
            },
        ],
    },
}
