import threading
import time
import PySimpleGUIQt as sg
from PySimpleGUIQt.PySimpleGUIQt import (
    POPUP_BUTTONS_NO_BUTTONS,
)
from gui_elements.globe_icon import globe_icon
from gui_elements.date_picker import popup_get_date
from gui_base import home_layout
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
        size=(1000, 1000),
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
    while True:
        event_func, values_func = window_func.read()
        print(event_func, values_func)

        if event_func == "Exit" or event_func == sg.WIN_CLOSED or event_func is None:
            break
        elif event_func == "Run":
            progress_bar.UpdateBar(10, 100)
            window_func["-PROGRESS-TEXT-"].update("Running..")

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
                    args = validation["cast"]

                    def long_operation_thread(window):
                        global thread_message
                        buteo_return = buteo_function(*args)
                        thread_message = buteo_return
                        window["-THREAD-"].click()

                        return buteo_return

                thread = threading.Thread(
                    target=long_operation_thread,
                    args=(window_func,),
                    daemon=True,
                )
                thread.start()

            except Exception as e:
                progress_bar.UpdateBar(0, 100)
                window_func["-PROGRESS-TEXT-"].update("Progress:")
                sg.Popup("Error", str(e))

        elif event_func == "-THREAD-":
            window_func["-PROGRESS-TEXT-"].update("Completed.")
            progress_bar.UpdateBar(100, 100)
            print(thread_message)
            thread.join(timeout=0)
            thread_message
        elif (
            isinstance(event_func, str)
            and len(event_func) > 8
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
                    validate_inputs(function_name, values_func, window_func)
            except Exception as e:
                sg.Popup("Error", str(e))

        else:
            validate_inputs(function_name, values_func, window_func)

    window_func.close()


def select_function(function_name, window):
    description = get_function_description(function_name)

    # window["-FUNC-LIST-"].update(default_values=[function_name])
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
