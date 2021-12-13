import PySimpleGUIQt as sg


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
                    size_px=(300, None),
                    pad=((0, 0), (0, 0)),
                    enable_events=True,
                    default_values=[functions[0]],
                )
            ]
        ],
        size=(300, None),
        pad=((0, 0), (0, 0)),
    )

    col2 = sg.Column(
        [
            [
                sg.Multiline(
                    description,
                    size_px=(None, None),
                    key="-DESC-",
                    disabled=True,
                    background_color="#f1f1f1",
                    pad=((0, 0), (0, 0)),
                )
            ],
            [
                sg.Button(
                    "Open Function",
                    key="-BUTTON1-",
                    size_px=(500, 60),
                    pad=((0, 0), (10, 0)),
                    bind_return_key=True,
                    border_width=0,
                )
            ],
        ],
        size=(500, None),
        element_justification="left",
        pad=((0, 0), (0, 0)),
    )

    base_layout = [
        sg.Column(
            [[col1, col2]],
            size=(920, None),
            pad=((0, 0), (0, 0)),
            scrollable=True,
            element_justification="left",
        )
    ]

    return [
        [
            sg.Menu(
                menu_def,
                tearoff=False,
            )
        ],
        base_layout,
    ]
