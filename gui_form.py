import threading
import PySimpleGUIQt as sg
from PySimpleGUIQt import POPUP_BUTTONS_NO_BUTTONS, WIN_CLOSED
from gui_elements.globe_icon import globe_icon
from gui_elements.date_picker import popup_get_date
from gui_func import (
    layout_from_name,
    validate_input,
    get_default_date,
    update_inputs,
)


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
    layout, buteo_function, listeners = layout_from_name(function_name)

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
            or event_func == WIN_CLOSED
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
            update_inputs(
                event_func, values_func, window_func, listeners, function_name
            )
            if run_clicked:
                validate_inputs(function_name, values_func, window_func)

    window_func.close()
