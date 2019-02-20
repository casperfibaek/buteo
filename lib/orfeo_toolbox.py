import subprocess
import os
import sys
import time
from progress import progress

otb_folder = os.path.abspath('../OTB/bin/')


def executeOtb(command, name, quiet=False):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    try:
        before = time.time()
        for line in iter(process.stdout.readline, ''):
            if 'FATAL' in line:
                raise RuntimeError(line)
            elif 'CRITICAL' in line:
                raise RuntimeError(line)
            elif 'WARNING' in line:
                continue
            elif quiet is False:
                if 'INFO' in line:
                    continue
            try:
                strip = line.strip()
                if len(strip) != 0:
                    part = strip.rsplit(':', 1)[1]
                    percent = int(part.split('%')[0])
                    progress(percent, 100, name)
            except:
                if len(line.strip()) != 0:
                    raise RuntimeError(line) from None
    except:
        raise RuntimeError('Critical failure while performing Orfeo-Toolbox action.')

    print(f'{name} completed in {round(time.time() - before, 2)}s.')


def pansharpen(in_pan, in_xs, out, options=None, out_datatype=None, quiet=False):
    ''' Pansharpen an image using the attributes
        of another image. Beware that the two images
        should be of the same size and position. '''

    cli = os.path.join(otb_folder, 'otbcli_Pansharpening.bat')

    ''' *******************************************************
        Parse the input and create CLI string
    ******************************************************* '''
    methods = ['rcs', 'lmvm', 'bayes']

    if options is None:
        options = {
            'method': 'lmvm',
            'method.lmvm.radiusx': 3,
            'method.lmvm.radiusy': 3,
        }

    if options['method'] not in methods:
        raise AttributeError('Selected method is not available.')

    if options['method'] == 'lmvm':
        if 'method.lmvm.radiusx' not in options:
            options['method.lmvm.radiusx'] = 3
        if 'method.lmvm.radiusx' not in options:
            options['method.lmvm.radiusx'] = 3
    if options['method'] == 'bayes':
        if 'method.bayes.lamda' not in options:
            options['method.bayes.lamda'] = 0.9999
        if 'method.bayes.s' not in options:
            options['method.bayes.s'] = 1

    if out_datatype is None:
        out_datatype = ''

    cli_args = [cli, '-inp', os.path.abspath(in_pan), '-inxs', os.path.abspath(in_xs), '-out', os.path.abspath(out), out_datatype]

    for key, value in options.items():
        cli_args.append('-' + str(key))
        cli_args.append(str(value))

    cli_string = ' '.join(cli_args)

    ''' *******************************************************
        Make CLI request and handle responses
    ******************************************************* '''

    executeOtb(cli_string, name='Pansharpening')

    return out
