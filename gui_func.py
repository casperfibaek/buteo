import os
import inspect
import datetime
import PySimpleGUIQt as sg
from PySimpleGUIQt.PySimpleGUIQt import (
    BUTTON_TYPE_BROWSE_FILE,
    BUTTON_TYPE_BROWSE_FILES,
    BUTTON_TYPE_SAVEAS_FILE,
    BUTTON_TYPE_BROWSE_FOLDER,
)
from gui_tools import tools


def get_list_of_functions():
    return [i for i in tools.keys()]


def get_first_key(dictionary):
    for key in dictionary.keys():
        return key


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


def get_param_name(tool_params):
    for key in tool_params.keys():
        return key


def get_default_date(target_param, window):
    date_str = window[target_param].Get()

    return (int(date_str[0:4]), int(date_str[4:6]), int(date_str[6:8]))


def validate_type(input_type, input_value, name, tool_name):
    valid = False
    keyword = False
    cast = None
    message = ""

    func_keywords = inspect.getfullargspec(tools[tool_name]["function_path"]).args

    func_params = tools[tool_name]["parameters"]
    definition = False
    for func_param in func_params:
        if name in func_param.keys():
            definition = func_param[name]
            break

    if definition is not False and "keyword" in definition.keys():
        keyword = definition["keyword"]

        if keyword and name not in func_keywords:
            raise Exception(
                f"{tool_name} has a keyword {name} that the function {tool_name} does not take."
            )

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

    elif input_type == "radio":
        if definition:
            options = definition["options"]
            for button in input_value:
                if True in button.values():
                    button_key = get_first_key(button)[len(name) + 1 :]
                    break

            for option in options:
                if option["key"] == button_key:
                    cast = option["value"]
                    valid = True
                    break

        else:
            valid = False
            message = f"{name}: {input_value} is defined poorly in tools.."

    elif input_type == "dropdown":
        if definition:
            options = definition["options"]

        for option in options:
            if option["label"] == input_value:
                cast = option["value"]
                valid = True
                break

    elif input_type == "slider":
        try:
            cast = float(input_value)
            if cast.is_integer():
                cast = int(cast)

            if "min_value" in definition.keys() and cast < definition["min_value"]:
                valid = False
                message = f"{name}: {input_value} is below the minimum value."
            elif "max_value" in definition.keys() and cast > definition["max_value"]:
                valid = False
                message = f"{name}: {input_value} is above the maximum value."
            else:
                valid = True
        except:
            valid = False
            message = f"{name}: {input_value} is not a valid number."

    elif input_type == "string" or "password":
        if isinstance(input_value, str):
            valid = True
            cast = input_value
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid string."
    elif input_type == "date_year":
        if isinstance(input_value, str) and len(input_value) == 8:
            valid = True
            cast = input_value
        else:
            valid = False
            message = f"{name}: {input_value} is not a valid date (yyyymmdd)."
    else:
        valid = False
        message = f"{name}: {input_value} is not a valid type."

    return valid, cast, message, keyword


default_texts = [
    "Choose a file..",
    "Choose files..",
    "Target file..",
    "Target folder..",
]


def validate_input(tool_name, parameters):
    ret_obj = {
        "valid": [],
        "cast_args": [],
        "cast_kwargs": {},
        "message": [],
        "names": [],
    }

    for tool_params in tools[tool_name]["parameters"]:
        name = get_param_name(tool_params)
        input_type = tool_params[name]["type"]

        if input_type == "radio":
            input_value = []
            for key in parameters.keys():
                if name == key[: len(name)]:
                    input_value.append({key: parameters[key]})
        else:
            input_value = parameters[name]

        if isinstance(input_value, str) and input_value in default_texts:
            valid = False
            cast = None
            message = f"No file/folder selected for {name}."
            keyword = False
        else:
            valid, cast, message, keyword = validate_type(
                input_type,
                input_value,
                name,
                tool_name,
            )

        ret_obj["valid"].append(valid)
        ret_obj["message"].append(message)
        ret_obj["names"].append(name)

        if keyword:
            ret_obj["cast_kwargs"][name] = cast
        else:
            ret_obj["cast_args"].append(cast)

    return ret_obj


def layout_from_name(name):
    if name not in tools:
        raise Exception("Tool not found")

    layout = []

    radio_group_id = 1
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

        if "display_name" in parameter[parameter_name]:
            display_name = parameter[parameter_name]["display_name"]
        else:
            display_name = parameter_name

        default_date_str = datetime.datetime(
            default_date[2], default_date[0], default_date[1]
        ).strftime("%Y%m%d")

        input_pad = ((0, 10), (0, 0))
        button_size = (16, 1.2)
        input_size = (54, 1.2)
        text_size = (24, 1.2)
        param_input = None
        path_input = None
        justification = "center"
        if parameter_type == "file_browse":
            param_input = sg.Button(
                "Browse",
                button_type=BUTTON_TYPE_BROWSE_FILE,
                key=parameter_name + "_picker",
                border_width=0,
                enable_events=True,
                size=button_size,
                tooltip=tooltip,
                target=parameter_name,
            )
            path_input = sg.In(
                default_text=default_texts[0],
                key=parameter_name,
                justification=justification,
                enable_events=True,
                disabled=False,
                tooltip=tooltip,
                size=(input_size[0] - button_size[0] - 1, input_size[1]),
                pad=input_pad,
            )
        elif parameter_type == "file_browse_multiple":
            param_input = sg.Button(
                "Browse",
                button_type=BUTTON_TYPE_BROWSE_FILES,
                key=parameter_name + "_picker",
                enable_events=True,
                border_width=0,
                size=button_size,
                tooltip=tooltip,
                target=parameter_name,
            )
            path_input = sg.In(
                default_text=default_texts[1],
                key=parameter_name,
                enable_events=True,
                disabled=False,
                tooltip=tooltip,
                justification=justification,
                size=(input_size[0] - button_size[0] - 1, input_size[1]),
                pad=input_pad,
            )
        elif parameter_type == "file_save":
            param_input = sg.Button(
                "Save As",
                button_type=BUTTON_TYPE_SAVEAS_FILE,
                border_width=0,
                key=parameter_name + "_picker",
                enable_events=True,
                size=button_size,
                tooltip=tooltip,
                target=parameter_name,
                file_types=((default_extension, default_extension),),
            )
            path_input = sg.In(
                default_text=default_texts[2],
                key=parameter_name,
                enable_events=True,
                disabled=False,
                tooltip=tooltip,
                justification=justification,
                size=(input_size[0] - button_size[0] - 1, input_size[1]),
                pad=input_pad,
            )
        elif parameter_type == "folder_save":
            param_input = sg.Button(
                "Browse",
                button_type=BUTTON_TYPE_BROWSE_FOLDER,
                border_width=0,
                key=parameter_name + "_picker",
                enable_events=True,
                size=button_size,
                tooltip=tooltip,
                target=parameter_name,
            )
            path_input = sg.In(
                default_text=default_texts[3],
                key=parameter_name,
                enable_events=True,
                disabled=False,
                tooltip=tooltip,
                justification=justification,
                size=(input_size[0] - button_size[0] - 1, input_size[1]),
                pad=input_pad,
            )
        elif (
            parameter_type == "number"
            or parameter_type == "string"
            or parameter_type == "password"
        ):
            param_input = sg.InputText(
                key=parameter_name,
                enable_events=True,
                password_char="*" if parameter_type == "password" else "",
                default_text=default,
                tooltip=tooltip,
                background_color="#f1f1f1",
                size=input_size,
                pad=input_pad,
            )
        elif parameter_type == "boolean":
            param_input = sg.Checkbox(
                "",
                key=parameter_name,
                enable_events=True,
                default=default,
                tooltip=tooltip,
                pad=((0, 0), (8, 0)),
            )
        elif parameter_type == "slider":
            param_args = parameter[parameter_name].keys()
            min_value = (
                parameter[parameter_name]["min_value"]
                if "min_value" in param_args
                else 0
            )
            max_value = (
                parameter[parameter_name]["max_value"]
                if "max_value" in param_args
                else 100
            )
            default_value = (
                parameter[parameter_name]["default"] if "default" in param_args else 50
            )
            step = parameter[parameter_name]["step"] if "step" in param_args else 1

            if default < min_value or default > max_value:
                default = min_value

            param_input = sg.Slider(
                range=(min_value, max_value),
                orientation="h",
                default_value=default_value,
                enable_events=True,
                tick_interval=step,
                key="slider_" + parameter_name,
                tooltip=tooltip,
                size_px=(360, 38),
                pad=input_pad,
            )

            path_input = sg.In(
                default_text=default,
                key=parameter_name,
                enable_events=True,
                disabled=False,
                tooltip=tooltip,
                size=button_size,
                pad=input_pad,
                justification=justification,
            )
        elif parameter_type == "dropdown":
            param_options = parameter[parameter_name]["options"]

            labels = []
            selected = None
            for idx, option in enumerate(param_options):
                labels.append(option["label"])
                if "default" in option.keys() and option["default"] == True:
                    selected = option["label"]

            param_input = sg.Combo(
                labels,
                default_value=selected,
                key=parameter_name,
                metadata=option["value"],
                background_color="#f1f1f1",
                readonly=True,
                enable_events=True,
                visible_items=10,
                tooltip=tooltip,
                size=(input_size[0] - button_size[0] - 1, input_size[1]),
            )
        elif parameter_type == "radio":
            param_options = parameter[parameter_name]["options"]
            param_input = []
            for idx, option in enumerate(param_options):
                if "default" in option.keys() and option["default"] == True:
                    selected = True
                else:
                    selected = False

                left_pad = 0 if idx == 0 else 16

                param_input.append(
                    sg.Radio(
                        option["label"],
                        radio_group_id,
                        default=selected,
                        key=parameter_name + "_" + option["key"],
                        metadata=option["value"],
                        tooltip=tooltip,
                        pad=((left_pad, 0), (0, 0)),
                    )
                )
            radio_group_id += 1
        elif parameter_type == "date_year":
            param_input = sg.Button(
                "Date",
                key="date_picker_" + parameter_name,
                button_type=sg.BUTTON_TYPE_READ_FORM,
                enable_events=True,
                tooltip=tooltip,
                bind_return_key=True,
                border_width=0,
                size=button_size,
            )
            path_input = sg.Input(
                default_date_str,
                key=parameter_name,
                enable_events=True,
                tooltip=tooltip,
                visible=True,
                disabled=False,
                justification=justification,
                size=(input_size[0] - button_size[0] - 1, input_size[1]),
                pad=input_pad,
            )

        if param_input is not None:
            param_text = sg.Text(
                display_name,
                tooltip=tooltip,
                key=parameter_name + "_text",
                background_color=sg.theme_background_color(),
                size=text_size,
                pad=((0, 0), (0, 0)),
                margins=(0, 0, 4, 0),
                justification="right",
            )

            if not isinstance(param_input, list):
                param_inputs = [param_input]
            else:
                param_inputs = param_input

            if parameter_type != "radio" and path_input is not None:
                if parameter_type in [
                    "date_year",
                    "file_browse",
                    "file_browse_multiple",
                    "folder_save",
                    "file_save",
                    "slider",
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
                        sg.Text("", size=(26, button_size[1])),
                        sg.Button(
                            "Run",
                            size=button_size,
                            key="-RUN-BUTTON-",
                            visible=True,
                            border_width=0,
                        ),
                        sg.Text("", size=(1, button_size[1])),
                        sg.Button(
                            "Exit",
                            size=button_size,
                            button_color=(sg.theme_background_color(), "#B22222"),
                            key="-EXIT-BUTTON-",
                            border_width=0,
                        ),
                    ]
                ],
            )
        ]
    )

    layout.append(
        [
            sg.Text("", size=(36, None)),
            sg.Text(
                "Progress:",
                key="-PROGRESS-TEXT-",
                pad=((20, 100), (0, 0)),
            ),
            sg.Column(
                [
                    [
                        sg.ProgressBar(
                            1,
                            orientation="h",
                            key="-PROGRESS-",
                            pad=((0, 24), (0, 0)),
                            size=(input_size[0] - 4, 36),
                        ),
                        sg.Button(
                            "Cancel",
                            key="-CANCEL-BUTTON-",
                            button_color=(sg.theme_background_color(), "#d7a824"),
                            border_width=0,
                            size=button_size,
                            pad=((10, 0), (0, 0)),
                        ),
                    ],
                ],
                pad=((10, 10), (0, 0)),
                size=(520, 36),
            ),
        ]
    )

    # layout.append(
    #     [
    #         sg.Output(
    #             pad=((0, 0), (10, 10)),
    #             size_px=(None, 200),
    #             background_color="#f1f1f1",
    #         ),
    #     ]
    # )

    layout = [
        [
            sg.Column(
                layout,
                size=(900, None),
                scrollable=True,
                element_justification="left",
                pad=((0, 0), (0, 0)),
            ),
            sg.Button("-THREAD-", visible=False),
        ]
    ]

    return (layout, tools[name]["function_path"])
