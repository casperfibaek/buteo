from itertools import product


def step_function(func, *args, grid=None, outfile=False, outfile_arg='outfile', outfile_prefix='step_', outfile_suffix='.shp'):
    assert(test_grid is not None)

    call_grid = []
    for key, val in grid.items():
        call = []

        for x in val:
            call.append((key, x))

        call_grid.append(call)

    calls = list(product(*call_grid))

    print(f'Running {len(calls)} calls!')

    for call in calls:
        calls_obj = {}
        for y in call:
            calls_obj[y[0]] = y[1]

        if outfile is True:
            outfile_name = outfile_prefix
            for key, val in calls_obj.items():
                outfile_name += f"{key}={str(val).replace('.', '-')}_"
            if len(calls_obj.items()) > 0:
                outfile_name = outfile_name[:-1]
            outfile_name += outfile_suffix

            if isinstance(outfile_arg, str):
                calls_obj[outfile_arg] = outfile_name
            elif isinstance(outfile_arg, int):
                edited_args = list(args)
                edited_args.insert(outfile_arg, outfile_name)
                edited_args = tuple(edited_args)

            func(*edited_args, **calls_obj)
        else:
            func(*args, **calls_obj)
