import PySimpleGUI as sg
from buteo.raster.borders import add_border_to_raster
from buteo.raster.align import align_rasters

tools = {
    "Add Borders": {
        "description": "Add borders to an image",
        "function_path": add_border_to_raster,
        "parameters": [
            {
                "input_raster": {
                    "optional": False,
                    "type": "file_browse",
                    "tooltip": "The raster to which the borders will be added",
                }
            },
            {
                "out_path": {
                    "optional": False,
                    "type": "file_save",
                    "default_extension": ".tif",
                }
            },
            {"border_size": {"optional": True, "type": "number", "default": 1}},
            {"border_value": {"optional": True, "type": "number", "default": 0}},
            {"overwrite": {"optional": True, "type": "boolean", "default": True}},
        ],
    },
    "Align Rasters": {
        "description": "Align rasters using a reference raster",
        "function_path": align_rasters,
        "parameters": [
            {
                "input_rasters": {
                    "optional": False,
                    "type": "file_browse_multiple",
                    "tooltip": "The rasters which will be aligned",
                }
            },
            {
                "out_path": {
                    "optional": False,
                    "type": "folder_save",
                }
            },
            {
                "master": {
                    "optional": False,
                    "type": "file_browse",
                    "tooltip": "The master_raster to which the other rasters will be aligned.",
                }
            },
            {
                "postfix": {
                    "optional": True,
                    "type": "string",
                    "default": "_aligned",
                    "tooltip": "The postfix to be added to the output rasters.",
                }
            },
        ],
    },
}


def functions():
    return tools.keys()


def get_function_description(function_name):
    return tools[function_name]["description"]


def layout_from_name(name):
    if name not in tools:
        raise Exception("Tool not found")

    layout = [[sg.Text(name)]]

    for parameter in tools[name]["parameters"]:
        parameter_name = list(parameter.keys())[0]
        parameter_type = parameter[parameter_name]["type"]

        if "default" in parameter[parameter_name]:
            default = parameter[parameter_name]["default"]
        else:
            default = False

        if "tooltip" in parameter[parameter_name]:
            tooltip = parameter[parameter_name]["tooltip"]
        else:
            tooltip = None

        if "default_extension" in parameter[parameter_name]:
            default_extension = parameter[parameter_name]["default_extension"]
        else:
            default_extension = ""

        param_input = None
        path_input = None
        if parameter_type == "file_browse":
            param_input = sg.FileBrowse(
                key=parameter_name,
                enable_events=True,
                tooltip=tooltip,
                target=parameter_name + "_path",
            )
            path_input = sg.Text(key=parameter_name + "_path", enable_events=True)
        elif parameter_type == "file_browse_multiple":
            param_input = sg.FilesBrowse(
                key=parameter_name,
                enable_events=True,
                tooltip=tooltip,
                target=parameter_name + "_path",
            )
            path_input = sg.Text(key=parameter_name + "_path", enable_events=True)
        elif parameter_type == "file_save":
            param_input = sg.SaveAs(
                key=parameter_name,
                enable_events=True,
                tooltip=tooltip,
                target=parameter_name + "_path",
                default_extension=default_extension,
            )
            path_input = sg.Text(key=parameter_name + "_path", enable_events=True)
        elif parameter_type == "folder_save":
            param_input = sg.FolderBrowse(
                key=parameter_name,
                enable_events=True,
                tooltip=tooltip,
                target=parameter_name + "_path",
            )
            path_input = sg.Text(key=parameter_name + "_path", enable_events=True)
        elif parameter_type == "number" or parameter_type == "string":
            param_input = sg.InputText(
                key=parameter_name,
                enable_events=True,
                default_text=default,
                tooltip=tooltip,
            )
        elif parameter_type == "boolean":
            param_input = sg.Checkbox(
                "True/False",
                key=parameter_name,
                enable_events=True,
                default=default,
                tooltip=tooltip,
            )

        if param_input is not None:
            param_text = sg.Text(parameter_name, justification="right")
            param_inputs = [param_input]

            if path_input is not None:
                param_inputs = [param_input, path_input]

            append = [
                sg.Column(
                    [
                        [param_text],
                    ],
                    size=(100, 36),
                    pad=0,
                    justification="right",
                    element_justification="right",
                    vertical_alignment="right",
                ),
                sg.Column(
                    [
                        param_inputs,
                    ],
                    size=(200, 36),
                    pad=0,
                    element_justification="left",
                    vertical_alignment="left",
                    justification="left",
                ),
            ]

            layout.append(append)

    layout.append([sg.Button("Run")])
    layout.append([sg.Text("Progress:")])
    layout.append([sg.ProgressBar(1, orientation="h", size=(20, 20), key="-PROGRESS-")])

    return (layout, tools[name]["function_path"])
