import os
import sys


def progress(count, total):
    bar_len = os.get_terminal_size().columns - 10
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '█' * filled_len + '.' * (bar_len - filled_len)

    sys.stdout.write(f"[{bar}] {percents} %\r")
    sys.stdout.flush()

    return None


def progress_callback(complete, message, unknown):
    return progress(complete, 1)


def progress_callback_quiet(complete, message, unknown):
    return None
