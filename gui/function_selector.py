import sys

sys.path.append("../")

import PySimpleGUI as sg
from globe_icon import globe_icon

from buteo.raster.resample import resample

sg.theme("Reddit")

# GLOBALS
called = 0
description = "HELLO HOW ARE YOU?"

functions = [
    "Align Rasters",
    "Clip Rasters",
    "Reproject Rasters",
    "Clip Vectors",
    "----",
    "Download Sentinel 1",
    "Download Sentinel 2",
    "Preprocess Sentinel 1",
    "Preprocess Sentinel 2",
    "Mosaic Sentinel 1",
    "Mosaic Sentinel 2",
    "----",
    "Patch Extraction",
]

# ------ Menu Definition ------ #
menu_def = [
    ["&File", ["E&xit"]],
    [
        "&Options",
        ["Paths", "Defaults"],
    ],
    [
        "&Help",
        ["Documentation", "About"],
    ],
]

right_click_menu = ["Unused", ["Right", "!&Click", "&Menu", "E&xit", "Properties"]]


col1 = sg.Column(
    [
        [
            sg.Listbox(
                [str(i) for i in functions],
                key="-FUNC-LIST-",
                size=(24, 22),
                enable_events=True,
                # highlight_background_color="#f0f0f0",
                p=10,
            )
        ]
    ],
    size=(200, 400),
    # scrollable=True,
    vertical_scroll_only=True,
    pad=0,
)

col2 = sg.Column(
    [
        [
            sg.Frame(
                "Description",
                [
                    [
                        sg.Multiline(
                            description,
                            size=(30, 16),
                            key="-DESC-",
                            disabled=True,
                            no_scrollbar=True,
                            border_width=None,
                            background_color=sg.theme_background_color(),
                        )
                    ],
                    [
                        sg.Button(
                            "Open Function",
                            key="-BUTTON1-",
                        )
                    ],
                ],
                size=(260, 320),
            )
        ]
    ],
    size=[300, 400],
    vertical_scroll_only=True,
    justification="left",
    element_justification="right",
    pad=0,
)

# The final layout is a simple one
layout = [
    [
        sg.Menu(
            menu_def,
            tearoff=False,
            pad=(200, 1),
            background_color="white",
        )
    ],
    [[col1, col2]],
]

window = sg.Window(
    "Buteo Toolbox",
    layout,
    resizable=False,
    size=(500, 400),
    finalize=True,
    icon=globe_icon,
)


def callback_function1():
    sg.popup("In Callback Function 1")
    print("In the callback function 1")
    resample("", "", "", method="bilinear")


def callback_function2(function_name):
    global description
    global called
    description = f"You called {function_name}.\nI AM FINE, THANKS!\nThis function was called {called} times.\nNeat, huh?! HERE IS A FEW MORE LINES JUST TO PUSH THE BOTTOM TEXT A BIT LOWER..."
    called += 1
    print("In the callback function 2")

    inp = window["-DESC-"]
    inp.update(value=description)


while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    elif event == "-BUTTON1-":
        callback_function1()  # call the "Callback" function
    elif event == "-FUNC-LIST-":
        callback_function2(values[event][0])  # call the "Callback" function

window.close()

# pyinstaller -wF --clean --icon=globe_icon.ico function_selector.py
