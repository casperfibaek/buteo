import os
import datetime
import PySimpleGUIQt as sg
from gui_tools import tools


def get_list_of_functions():
    return [i for i in tools.keys()]


def get_function_description(function_name):
    return tools[function_name]["description"]


def add_slash_to_end(path):
    if path[-1] != "/":
        path += "/"
    return path


def get_today_date():
    today = datetime.date.today()
    return (today.month, today.day, today.year)


def get_days_ago(days_ago):
    today = datetime.date.today()
    delta = datetime.timedelta(days_ago)

    fortnight = today - delta
    return (fortnight.month, fortnight.day, fortnight.year)


def parse_date(date_str):
    if date_str == "today":
        return get_today_date()

    elif "days_ago" in date_str:
        days = int(date_str.split("_")[2])
        return get_days_ago(days)
    else:
        return get_today_date()


def validate_type(input_type, input_value, name):
    valid = False
    cast = None
    message = ""
    if input_type == "file_browse":
        if isinstance(input_value, str) and os.path.isfile(input_value):
            valid = True
            cast = input_value
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid file."

    elif input_type == "file_browse_multiple":
        if (
            isinstance(input_value, str)
            and len(input_value) > 0
            and len(input_value.split(";")) > 0
            and all(os.path.isfile(i) for i in input_value.split(";"))
        ):
            valid = True
            cast = input_value.split(";")
        else:
            valid = False
            message = f"{name}: {input_value} contains invalid files."

    elif input_type == "folder_save":
        if isinstance(input_value, str) and os.path.isdir(input_value):
            valid = True
            cast = add_slash_to_end(input_value)
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid folder."

    elif input_type == "file_save":
        if isinstance(input_value, str) and os.path.isdir(os.path.dirname(input_value)):
            valid = True
            cast = input_value
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid save destination."

    elif input_type == "number":
        if (
            isinstance(input_value, int)
            or isinstance(input_value, float)
            or isinstance(input_value, str)
        ):
            try:
                cast = float(input_value)
                if cast.is_integer():
                    cast = int(cast)

                valid = True
            except:
                valid = False
                message = f"{name}: {input_value} is not a valid number."
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid number."

    elif input_type == "boolean":
        if isinstance(input_value, bool):
            valid = True
            cast = input_value
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid boolean."

    elif input_type == "string":
        if isinstance(input_value, str):
            valid = True
            cast = input_value
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid string."
    elif input_type == "date_year":
        if (
            isinstance(input_value, str)
            and len(input_value) >= 6
            and len(input_value) <= 8
        ):
            valid = True
            cast = input_value
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid date (yyyymmdd)."

    return valid, cast, message


def get_param_name(tool_params):
    for key in tool_params.keys():
        return key


default_texts = [
    "Choose a file..",
    "Choose files..",
    "Target file..",
    "Target folder..",
]


def validate_input(tool_name, parameters):
    ret_obj = {
        "valid": [],
        "cast": [],
        "message": [],
        "names": [],
    }

    for tool_params in tools[tool_name]["parameters"]:
        name = get_param_name(tool_params)
        input_type = tool_params[name]["type"]
        input_value = parameters[name]

        if input_value in default_texts:
            valid = False
            cast = None
            message = f"No file/folder selected for {name}."
        else:
            valid, cast, message = validate_type(input_type, input_value, name)

        ret_obj["valid"].append(valid)
        ret_obj["cast"].append(cast)
        ret_obj["message"].append(message)
        ret_obj["names"].append(name)

    return ret_obj


def layout_from_name(name):
    if name not in tools:
        raise Exception("Tool not found")

    layout = []

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
            default_extension = "*"

        if "default_date" in parameter[parameter_name]:
            default_date = parse_date(parameter[parameter_name]["default_date"])
        else:
            default_date = get_today_date()

        default_date_str = datetime.datetime(
            default_date[2], default_date[0], default_date[1]
        ).strftime("%Y%m%d")

        button_size = (16, 1.2)
        param_input = None
        path_input = None
        if parameter_type == "file_browse":
            param_input = sg.FileBrowse(
                key=parameter_name + "_picker",
                enable_events=True,
                size=button_size,
                tooltip=tooltip,
                target=parameter_name,
            )
            path_input = sg.In(
                default_text="Choose a file..",
                key=parameter_name,
                enable_events=True,
                disabled=True,
            )
        elif parameter_type == "file_browse_multiple":
            param_input = sg.FilesBrowse(
                key=parameter_name + "_picker",
                enable_events=True,
                size=button_size,
                tooltip=tooltip,
                target=parameter_name,
            )
            path_input = sg.In(
                default_text="Choose files..",
                key=parameter_name,
                enable_events=True,
                disabled=True,
            )
        elif parameter_type == "file_save":
            param_input = sg.SaveAs(
                key=parameter_name + "_picker",
                enable_events=True,
                size=button_size,
                tooltip=tooltip,
                target=parameter_name,
                file_types=((default_extension, default_extension),),
            )
            path_input = sg.In(
                default_text="Target file..",
                key=parameter_name,
                enable_events=True,
                disabled=True,
            )
        elif parameter_type == "folder_save":
            param_input = sg.FolderBrowse(
                key=parameter_name + "_picker",
                enable_events=True,
                size=button_size,
                tooltip=tooltip,
                target=parameter_name,
            )
            path_input = sg.In(
                default_text="Target folder..",
                key=parameter_name,
                enable_events=True,
                disabled=True,
            )
        elif parameter_type == "number" or parameter_type == "string":
            param_input = sg.InputText(
                key=parameter_name,
                enable_events=True,
                default_text=default,
                tooltip=tooltip,
                background_color="#f1f1f1",
            )
        elif parameter_type == "boolean":
            param_input = sg.Checkbox(
                "True/False",
                key=parameter_name,
                enable_events=True,
                default=default,
                tooltip=tooltip,
                pad=((0, 0), (0, 10)),
            )
        elif parameter_type == "date_year":
            param_input = sg.CalendarButton(
                "Date",
                key=parameter_name + "_picker",
                border_width=0,
                tooltip=tooltip,
                target=parameter_name,
                size=button_size,
                default_date_m_d_y=default_date,
            )
            path_input = sg.Input(
                default_date_str,
                key=parameter_name,
                enable_events=True,
                visible=True,
                disabled=True,
            )

        if param_input is not None:
            param_text = sg.Text(
                parameter_name,
                tooltip=tooltip,
                key=parameter_name + "_text",
                background_color=sg.theme_background_color(),
                size=(26, button_size[1]),
                pad=((0, 0), (0, 0)),
                justification="right",
            )
            param_inputs = [param_input]

            if path_input is not None:
                if parameter_type in [
                    "date_year",
                    "file_browse",
                    "file_browse_multiple",
                    "folder_save",
                    "file_save",
                ]:
                    param_inputs = [path_input, param_input]
                else:
                    param_inputs = [param_input, path_input]

            append = [
                sg.Column(
                    [
                        [param_text],
                    ],
                    size=(120, 36),
                    pad=((0, 0), (0, 0)),
                    element_justification="r",
                ),
                sg.Column(
                    [
                        param_inputs,
                    ],
                    size=(260, 36),
                    pad=((0, 0), (0, 0)),
                ),
            ]

            layout.append(append)

    layout.append(
        [
            sg.Column(
                [
                    [
                        sg.Text("", size=(18, 1)),
                        sg.Button("Run", size=button_size),
                        sg.Exit(
                            "Exit",
                            size=button_size,
                            button_color=(sg.theme_background_color(), "firebrick"),
                        ),
                    ]
                ],
            )
        ]
    )

    layout.append(
        [
            sg.Text(
                "Progress:",
                key="-PROGRESS-TEXT-",
                pad=((0, 0), (10, 0)),
            ),
        ]
    )

    layout.append(
        [
            sg.ProgressBar(
                1,
                orientation="h",
                size=(None, 20),
                key="-PROGRESS-",
                pad=((0, 0), (0, 10)),
            )
        ]
    )

    # layout.append(
    #     [
    #         sg.Output(
    #             pad=((0, 0), (10, 10)),
    #             size_px=(None, 200),
    #             background_color="#f1f1f1",
    #         )
    #     ]
    # )

    layout = [
        [
            sg.Column(
                layout,
                size=(800, None),
                # pad=((0, 0), (0, 200)),
                scrollable=True,
                element_justification="left",
            )
        ]
    ]

    return (layout, tools[name]["function_path"])
