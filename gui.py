import PySimpleGUIQt as sg
from gui_elements.globe_icon import globe_icon
from gui_elements.gui_base import home_layout
from gui_form import open_function
from gui_func import (
    get_list_of_functions,
    get_function_description,
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
