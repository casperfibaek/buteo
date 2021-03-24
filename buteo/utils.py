import os
import sys
import time


def progress(count, total, name='Processing'):
    sys.stdout.flush()
    bar_len = os.get_terminal_size().columns - 24
    filled_len = int(round(bar_len * count / float(total)))
    display_name = name[:10] + '..: '

    bar = '█' * filled_len + '.' * (bar_len - filled_len)

    percents = round(100.0 * count / float(total), 1)

    if count == total:
        sys.stdout.write(f"{display_name}[{bar}] {percents} %\r")
        sys.stdout.flush()
        print("")
        return None
    else:
        sys.stdout.write(f"{display_name}[{bar}] {percents} %\r")
        sys.stdout.flush()

    return None


def timing(before):
    after = time.time()
    dif = after - before
    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = "{0:.2f}".format(dif % 60)
    print(f"Processing took: {hours}h {minutes}m {seconds}s")


def path_to_ext(path):
    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    ext = basesplit[1]

    return ext


def path_to_name(path, with_ext=False):
    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    name = basesplit[0]

    if with_ext:
        return basename
    
    return name


def file_exists(path):
    return os.path.exists(path)


def remove_if_overwrite(path, overwrite):
    if path is not None:
        exists = file_exists(path)
        if exists and overwrite:
            os.remove(path)
        elif exists:
            raise Exception(f"File: {path} already exists and overwrite is False.")


def get_size(start_path='.', rough=True):
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    if rough is True:
        return total_size >> 20
    else:
        return total_size


def divide_steps(total, step):
    steps = []
    remainder = total % step
    divided = int(total / step)
    for _ in range(step):
        if remainder > 0:
            steps.append(divided + 1)
            remainder -= 1
        else:
            steps.append(divided)

    return steps


def divide_into_steps(arr, steps_length):
    steps = divide_steps(len(arr), steps_length)

    ret_arr = []
    last = 0
    count = 0
    for x in steps:
        count += 1
        if count > len(arr):
            continue
        ret_arr.append(arr[last:x + last])
        last += x

    return ret_arr


def step_ranges(steps):
    start_stop = []
    last = 0
    for num in range(0, len(steps)):
        step_size = steps[num]
        id = num + 1

        start_stop.append({
            "id": id,
            "start": last,
            "stop": last + step_size,
        })

        last += step_size

    return start_stop

