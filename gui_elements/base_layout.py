import PySimpleGUI as sg


def home_layout(functions):
    description = "Select a function to run."

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

    col1 = sg.Column(
        [
            [
                sg.Listbox(
                    [str(i) for i in functions],
                    key="-FUNC-LIST-",
                    size=(24, 20),
                    pad=0,
                    enable_events=True,
                )
            ]
        ],
        size=(200, 370),
        pad=0,
    )

    col2 = sg.Column(
        [
            [
                sg.Multiline(
                    description,
                    size=(38, 21),
                    key="-DESC-",
                    disabled=True,
                    no_scrollbar=True,
                    border_width=None,
                    background_color=sg.theme_background_color(),
                    pad=0,
                )
            ],
            [
                sg.Button(
                    "Open Function",
                    key="-BUTTON1-",
                )
            ],
        ],
        size=[300, 370],
        vertical_scroll_only=True,
        element_justification="right",
        pad=0,
    )

    return [
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
