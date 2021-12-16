import os
import ctypes
import platform
import threading
import PySimpleGUIQt as sg
from PySimpleGUIQt.PySimpleGUIQt import (
    POPUP_BUTTONS_NO_BUTTONS,
)
from gui_elements.globe_icon import globe_icon
from gui_elements.date_picker import popup_get_date
from gui_elements.gui_base import home_layout
from gui_func import (
    get_list_of_functions,
    layout_from_name,
    get_function_description,
    validate_input,
    get_default_date,
)


KEY_UP_QT = "special 16777235"
KEY_DOWN_QT = "special 16777237"
KEY_ENTER_QT = "special 16777220"
DEFAULT_MARGINS = (0, 0)
DEFAULT_ELEMENT_PADDING = (0, 0)
DEFAULT_FONT = ("Helvetica", 10)
DEFAULT_TEXT_JUSTIFICATION = "left"
DEFAULT_BORDER_WIDTH = 0

# os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

global thread
global thread_message


def validate_inputs(function_name, values_func, window):
    validation = validate_input(function_name, values_func)

    for idx, input_param_name in enumerate(validation["names"]):
        input_param_key = input_param_name + "_text"

        if validation["valid"][idx] == False:
            window[input_param_key].update(background_color="#cfa79d")
        else:
            window[input_param_key].update(background_color=sg.theme_background_color())

    return validation


def open_function(function_name):
    global thread, thread_message
    layout, buteo_function = layout_from_name(function_name)

    window_func = sg.Window(
        function_name,
        layout,
        resizable=True,
        size=(900, 1100),
        finalize=True,
        icon=globe_icon,
        element_justification="center",
        border_depth=0,
        element_padding=(0, 0),
    )

    progress_bar = window_func["-PROGRESS-"]
    progress_bar.UpdateBar(0, 100)
    print("Opening function:", function_name)

    thread = None
    run_clicked = False
    while True:
        event_func, values_func = window_func.read()

        if (
            event_func == "-EXIT-BUTTON-"
            or event_func == sg.WIN_CLOSED
            or event_func is None
        ):
            break
        elif event_func == "-RUN-BUTTON-":
            run_clicked = True

            try:
                validation = validate_inputs(function_name, values_func, window_func)

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
                    args = validation["cast_args"]
                    kwargs = validation["cast_kwargs"]

                    def long_operation_thread(window):
                        global thread_message

                        buteo_return = None
                        try:
                            buteo_return = buteo_function(*args, **kwargs)
                            thread_message = buteo_return
                        except Exception as e:
                            thread_message = ("Error", e)
                        window["-THREAD-"].click()

                        return buteo_return

                    progress_bar.UpdateBar(10, 100)
                    window_func["-PROGRESS-TEXT-"].update("Running..")
                    window_func["-RUN-BUTTON-"].update(
                        button_color=(sg.theme_element_text_color(), "#999999")
                    )

                    thread = threading.Thread(
                        target=long_operation_thread,
                        args=(window_func,),
                        daemon=True,
                    )
                    thread.start()

            except Exception as e:
                progress_bar.UpdateBar(0, 100)
                window_func["-PROGRESS-TEXT-"].update("Progress:")
                window_func["-RUN-BUTTON-"].update(button_color=sg.theme_button_color())

                sg.Popup("Error", str(e))

        elif event_func == "-THREAD-":
            try:
                thread.join(timeout=0)
                print(thread_message)
            except:
                print("Error joining thread")

            if isinstance(thread_message, list) and thread_message[0] == "Error":
                sg.Popup("Error", str(thread_message[1]))
                window_func["-PROGRESS-TEXT-"].update("Progress:")
                progress_bar.UpdateBar(0, 100)
            else:
                window_func["-PROGRESS-TEXT-"].update("Completed.")
                progress_bar.UpdateBar(100, 100)

            window_func["-RUN-BUTTON-"].update(button_color=sg.theme_button_color())
        elif (
            isinstance(event_func, str)
            and len(event_func) > 12
            and event_func[:12] == "date_picker_"
        ):
            target_param = event_func[12:]
            try:
                default_date = get_default_date(target_param, window_func)
                date = popup_get_date(
                    icon=globe_icon,
                    start_year=default_date[0],
                    start_mon=default_date[1],
                    start_day=default_date[2],
                )

                if date is not None:
                    window_func[event_func[12:]].update(value=date)
                    if run_clicked:
                        validate_inputs(function_name, values_func, window_func)
            except Exception as e:
                sg.Popup("Error", str(e))

        elif (
            isinstance(event_func, str)
            and len(event_func) > len("slider_")
            and event_func[: len("slider_")] == "slider_"
        ):
            target_param = event_func[len("slider_") :]
            window_func[target_param].update(value=values_func[event_func])
            if run_clicked:
                validate_inputs(function_name, values_func, window_func)
        else:
            if run_clicked:
                validate_inputs(function_name, values_func, window_func)

    window_func.close()


def select_function(function_name, window):
    description = get_function_description(function_name)
    window["-DESC-"].update(value=description)


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
    return_keyboard_events=True,
    border_depth=0,
)

select_function(available_functions[0], window)
current_selection = 0
max_selection = len(available_functions) - 1

list_not_clicked = True
ignore_list_update = False
while True:
    event, values = window.read()

    update_func = True
    if event == "Exit" or event == sg.WIN_CLOSED or event is None:
        break
    elif (
        event == "-BUTTON1-"
        or event == "-FUNC-LIST-DOUBLE-CLICK-"
        or event == KEY_ENTER_QT
    ):
        if isinstance(values["-FUNC-LIST-"], list) and len(values["-FUNC-LIST-"]) != 0:
            function_name = values["-FUNC-LIST-"][0]
            open_function(function_name)
    elif event == "-FUNC-LIST-":
        if ignore_list_update:
            ignore_list_update = False
            continue

        list_not_clicked = False
        current_selection = available_functions.index(values[event][0])
        select_function(available_functions[current_selection], window)
    elif event == KEY_DOWN_QT and list_not_clicked:
        if current_selection < max_selection:
            ignore_list_update = True
            current_selection += 1
            select_function(available_functions[current_selection], window)
            window["-FUNC-LIST-"].update(set_to_index=current_selection)
    elif event == KEY_UP_QT and list_not_clicked:
        if current_selection > 0:
            ignore_list_update = True
            current_selection -= 1
            select_function(available_functions[current_selection], window)
            window["-FUNC-LIST-"].update(set_to_index=current_selection)


window.close()

# pyinstaller -wF --noconfirm --clean --noconsole --icon=./gui_elements/globe_icon.ico gui.py
