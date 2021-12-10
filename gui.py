import PySimpleGUIQt as sg
from PySimpleGUIQt.PySimpleGUIQt import (
    POPUP_BUTTONS_NO_BUTTONS,
)
from gui_elements.globe_icon import globe_icon
from gui_base import home_layout
from gui_func import (
    get_list_of_functions,
    layout_from_name,
    get_function_description,
    validate_input,
    get_param_name,
)


window_func = None


def validate_inputs(function_name, values_func):
    validation = validate_input(function_name, values_func)

    for idx, input_param_name in enumerate(validation["names"]):
        input_param_key = input_param_name + "_text"

        if validation["valid"][idx] == False:
            window_func[input_param_key].update(background_color="red")
        else:
            window_func[input_param_key].update(
                background_color=sg.theme_background_color()
            )

    return validation


def open_function(function_name):
    layout, buteo_function = layout_from_name(function_name)

    global window_func
    window_func = sg.Window(
        function_name,
        layout,
        resizable=True,
        size=(800, 1080),
        finalize=True,
        icon=globe_icon,
        element_justification="center",
        border_depth=0,
    )

    progress_bar = window_func["-PROGRESS-"]
    progress_bar.UpdateBar(0, 100)
    print("Opening function:", function_name)

    while True:
        event_func, values_func = window_func.read()

        if event_func == "Exit" or event_func == sg.WIN_CLOSED:
            break
        elif event_func == "Run":
            progress_bar.UpdateBar(10, 100)
            window_func["-PROGRESS-TEXT-"].update("Running..")

            try:
                validation = validate_inputs(function_name, values_func)

                if False in validation["valid"]:
                    sg.popup(
                        "\n".join(validation["message"]),
                        title="Error",
                        keep_on_top=True,
                        no_titlebar=False,
                        grab_anywhere=True,
                        button_type=POPUP_BUTTONS_NO_BUTTONS,
                        non_blocking=True,
                    )
                    progress_bar.UpdateBar(0, 100)
                else:
                    args = validation["cast"]
                    buteo_function(*args)

                    window_func["-PROGRESS-TEXT-"].update("Completed")
                    progress_bar.UpdateBar(100, 100)

            except Exception as e:
                progress_bar.UpdateBar(0, 100)
                window_func["-PROGRESS-TEXT-"].update("Progress:")
                sg.Popup("Error", str(e))

        elif event_func == "-OPERATION DONE-":
            window_func["-PROGRESS-TEXT-"].update("Completed.")
            progress_bar.UpdateBar(100, 100)
        else:
            validate_inputs(function_name, values_func)

    window_func.close()


def select_function(function_name):
    description = get_function_description(function_name)

    inp = window["-DESC-"]
    inp.update(value=description)


sg.theme("Reddit")

available_functions = get_list_of_functions()

window = sg.Window(
    "Buteo Toolbox",
    home_layout(available_functions),
    resizable=True,
    auto_size_buttons=True,
    size=(800, 600),
    finalize=True,
    icon=globe_icon,
    element_justification="center",
)

select_function(available_functions[0])

while True:
    event, values = window.read()
    print(event, values)

    if event == "Exit" or event == sg.WIN_CLOSED or event is None:
        break
    elif event == "-BUTTON1-" or event == "-FUNC-LIST-DOUBLE-CLICK-":
        if isinstance(values["-FUNC-LIST-"], list) and len(values["-FUNC-LIST-"]) != 0:
            function_name = values["-FUNC-LIST-"][0]
            open_function(function_name)
    elif event == "-FUNC-LIST-":
        if isinstance(values[event], list) and len(values[event]) != 0:
            select_function(values[event][0])


if window_func is not None:
    try:
        window_func.close()
    except Exception:
        pass

window.close()

# pyinstaller -wF --noconfirm --clean --noconsole --icon=./gui_elements/globe_icon.ico gui.py
