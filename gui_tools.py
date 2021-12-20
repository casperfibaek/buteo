from buteo.raster.borders import add_border_to_raster
from buteo.raster.align import align_rasters
from buteo.raster.clip import clip_raster
from buteo.earth_observation.download_sentinel import download_s2_tile
from buteo.earth_observation.s2_mosaic import mosaic_tile_s2

# from buteo.machine_learning.patch_extraction import predict_raster
from buteo.filters.norm_rasters import norm_rasters


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
                    "display_name": "AOI Reference",
                    "type": "file_browse",
                    "tooltip": "The reference file containing the area of interest. Can be both a vector or raster.",
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
                    "type": "slider",
                    "tooltip": "The maximum cloud cover allowed for the data to be downloaded.",
                    "default": 20,
                    "min_value": 0,
                    "max_value": 100,
                    "step": 1,
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
                "s2_files": {
                    "display_name": "S2 Files",
                    "type": "file_browse_multiple",
                    "tooltip": "The Sentinel 2 files to be mosaicked (.zip or .safe).",
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
                "tmp_folder": {
                    "display_name": "Temporary Folder",
                    "type": "folder_save",
                    "tooltip": "The folder where temporary files will be saved.",
                }
            },
            {
                "ideal_date": {
                    "keyword": True,
                    "display_name": "Ideal Date",
                    "type": "date_year",
                    "tooltip": "The ideal central date for the mosaic.",
                    "default_date": "days_ago_14",
                },
            },
            {
                "max_time_delta": {
                    "keyword": True,
                    "display_name": "Time Delta (days)",
                    "type": "number",
                    "tooltip": "The maximum time delta in days allowed for included rasters.",
                    "default": 60,
                },
            },
            {
                "max_images": {
                    "keyword": True,
                    "display_name": "Maximum Images",
                    "type": "number",
                    "tooltip": "The maximum included images in a mosaic.",
                    "default": 6,
                },
            },
            {
                "clean_tmp_folder": {
                    "keyword": True,
                    "display_name": "Clean Temp. Folder",
                    "type": "boolean",
                    "tooltip": "Clean the temporary folder after the mosaic is done.",
                    "default": False,
                },
            },
        ],
    },
    # "Predict Raster": {
    #     "description": "Applies a deep learning model to a raster",
    #     "function_path": predict_raster,
    #     "parameters": [
    #         {
    #             "raster_list": {
    #                 "display_name": "Raster Files",
    #                 "type": "file_browse_multiple",
    #                 "tooltip": "The input rasters for the model to use.",
    #             }
    #         },
    #         {
    #             "tile_size": {
    #                 "keyword": True,
    #                 "display_name": "Tile Sizes",
    #                 "type": "string",
    #                 "tooltip": "tile_sizes in the format: '32, 32, 16'",
    #                 "default": "32, 32, 16",
    #             }
    #         },
    #         {
    #             "output_tile_size": {
    #                 "keyword": True,
    #                 "display_name": "Output Tile Size",
    #                 "type": "number",
    #                 "tooltip": "The tilesize of the output prediction raster.",
    #                 "default": 32,
    #             }
    #         },
    #         {
    #             "output_channels": {
    #                 "keyword": True,
    #                 "display_name": "Output Channels",
    #                 "type": "number",
    #                 "tooltip": "The number of channels in the output prediction raster.",
    #                 "default": 1,
    #             },
    #         },
    #         {
    #             "model_path": {
    #                 "keyword": True,
    #                 "display_name": "Path to Model",
    #                 "type": "file_browse",
    #                 "tooltip": "The deep learning model to use. (Tensorflow formats)",
    #             },
    #         },
    #         {
    #             "reference_raster": {
    #                 "keyword": True,
    #                 "display_name": "Reference Array",
    #                 "type": "file_browse",
    #                 "tooltip": "A reference raster for the output.",
    #             },
    #         },
    #         {
    #             "out_path": {
    #                 "keyword": True,
    #                 "display_name": "Out Path",
    #                 "type": "file_save",
    #                 "tooltip": "Where to save the output prediction raster.",
    #             },
    #         },
    #         {
    #             "offsets": {
    #                 "keyword": True,
    #                 "display_name": "Use Offsets",
    #                 "type": "boolean",
    #                 "tooltip": "Should offsets be used when making the prediction?",
    #                 "default": True,
    #             },
    #         },
    #         {
    #             "batch_size": {
    #                 "keyword": True,
    #                 "display_name": "Batch Size",
    #                 "type": "number",
    #                 "tooltip": "The batch size to use when making the prediction.",
    #                 "default": 64,
    #             },
    #         },
    #         {
    #             "method": {
    #                 "keyword": True,
    #                 "display_name": "Merge Method",
    #                 "type": "radio",
    #                 "tooltip": "Where to save the output prediction raster.",
    #                 "options": [
    #                     {
    #                         "label": "Median",
    #                         "key": "median",
    #                         "value": "median",
    #                         "default": True,
    #                     },
    #                     {
    #                         "label": "Olympic",
    #                         "key": "olympic",
    #                         "value": "olympic",
    #                     },
    #                 ],
    #             },
    #         },
    #         {
    #             "scale_to_sum": {
    #                 "keyword": True,
    #                 "display_name": "Scale to sum",
    #                 "type": "boolean",
    #                 "tooltip": "Should the output be scaled to the sum predictions?",
    #                 "default": False,
    #             },
    #         },
    #     ],
    # },
    "Normalise Rasters": {
        "description": "Normalises and splits rasters into bands",
        "function_path": norm_rasters,
        "parameters": [
            {
                "in_rasters": {
                    "display_name": "Raster Files",
                    "type": "file_browse_multiple",
                    "tooltip": "The raster files to be normalised.",
                }
            },
            {
                "out_folder": {
                    "display_name": "Output Folder",
                    "type": "folder_save",
                    "tooltip": "The folder where the normalised rasters will be saved.",
                }
            },
            {
                "method": {
                    "keyword": True,
                    "display_name": "Method",
                    "type": "dropdown",
                    "tooltip": "The normalisation method to use.",
                    "options": [
                        {
                            "label": "Normalise",
                            "value": "normalise",
                            "default": True,
                        },
                        {
                            "label": "Standardise",
                            "value": "standardise",
                        },
                        {
                            "label": "Median Absolute Deviation",
                            "value": "median_absolute_deviation",
                        },
                        {
                            "label": "Robust 2 - 98",
                            "value": "robust_98",
                        },
                        {
                            "label": "Robust Quantile",
                            "value": "robust_quantile",
                        },
                        {
                            "label": "Range",
                            "value": "range",
                        },
                    ],
                }
            },
            {
                "min_target": {
                    "keyword": True,
                    "display_name": "Target Minumum",
                    "type": "number",
                    "tooltip": "METHOD: Range.: The new minumum value for the target raster.",
                    "default": 0,
                },
            },
            {
                "max_target": {
                    "keyword": True,
                    "display_name": "Target Maximim",
                    "type": "number",
                    "tooltip": "METHOD: Range.: The new maximum value for the target raster.",
                    "default": 10000,
                },
            },
            {
                "min_og": {
                    "keyword": True,
                    "display_name": "Original Minumum",
                    "type": "number",
                    "tooltip": "METHOD: Range.: The minumum of the original raster. if -9999 the nanmin() is used.",
                    "default": -9999,
                },
            },
            {
                "max_og": {
                    "keyword": True,
                    "display_name": "Original Maximim",
                    "type": "number",
                    "tooltip": "METHOD: Range.: The maximum of the original raster. if -9999 the nanmax() is used.",
                    "default": -9999,
                },
            },
            {
                "truncate": {
                    "keyword": True,
                    "display_name": "Truncate",
                    "type": "boolean",
                    "tooltip": "Should the raster be truncated between the target min and max?",
                    "default": True,
                },
            },
            {
                "split_bands": {
                    "keyword": True,
                    "display_name": "Split Bands",
                    "type": "boolean",
                    "tooltip": "Should the bands of the rasters be split in the output?",
                    "default": False,
                },
            },
            {
                "prefix": {
                    "keyword": True,
                    "display_name": "Prefix",
                    "type": "string",
                    "default": "",
                    "tooltip": "The prefix to be added to the output rasters.",
                }
            },
            {
                "postfix": {
                    "keyword": True,
                    "display_name": "Postfix",
                    "type": "string",
                    "default": "",
                    "tooltip": "The postfix to be added to the output rasters.",
                }
            },
        ],
    },
}
