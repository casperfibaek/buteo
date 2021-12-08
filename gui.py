import PySimpleGUI as sg
from gui_elements.globe_icon import globe_icon
from gui_elements.base_layout import home_layout
from gui_tools import functions, layout_from_name, get_function_description
from buteo.utils import keys_to_args


sg.theme("Reddit")

window = sg.Window(
    "Buteo Toolbox",
    home_layout(functions()),
    resizable=False,
    size=(500, 400),
    finalize=True,
    element_padding=0,
    icon=globe_icon,
)


def open_function(function_name):
    layout, buteo_function = layout_from_name(function_name)

    window_func = sg.Window(
        function_name,
        layout,
        resizable=True,
        size=(500, 400),
        finalize=True,
        icon=globe_icon,
    )
    progress_bar = window_func["-PROGRESS-"]
    progress_bar.UpdateBar(0, 100)

    while True:
        event_func, values_func = window_func.read()

        if event_func == "Exit" or event_func == sg.WIN_CLOSED:
            break
        elif event_func == "Run":
            progress_bar.UpdateBar(0, 100)
            try:
                buteo_function(*keys_to_args(values_func))
                progress_bar.UpdateBar(100, 100)
            except Exception as e:
                progress_bar.UpdateBar(0, 100)
                sg.Popup("Error", str(e))

    window_func.close()


def select_function(function_name):
    description = get_function_description(function_name)

    inp = window["-DESC-"]
    inp.update(value=description)


while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    elif event == "-BUTTON1-":
        if isinstance(values["-FUNC-LIST-"], list) and len(values["-FUNC-LIST-"]) != 0:
            open_function(values["-FUNC-LIST-"][0])
    elif event == "-FUNC-LIST-":
        select_function(values[event][0])

window.close()

# pyinstaller -wF --noconfirm --clean --noconsole --icon=./gui_elements/globe_icon.ico gui.py
