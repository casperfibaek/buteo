import multiprocessing
import queue
import psutil
import signal
import subprocess



### Functions to enable command line interface with multiprocessing, allowing for KeyboardInterrupt etc.

## Used in sen2mosaic and sen1mosaic preprocessing steps

def _do_work(job_queue, partial_func, counter=None):
    """
    Processes jobs from  the multiprocessing queue until all jobs are finished
    Adapted from: https://github.com/ikreymer/cdx-index-client
    
    Args:
        job_queue: multiprocessing.Queue() object
        counter: multiprocessing.Value() object
    """
           
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while not job_queue.empty():
        try:
            job = job_queue.get_nowait()
            
            partial_func(job)

            num_done = 0
            with counter.get_lock():
                counter.value += 1
                num_done = counter.value
                
        except queue.Empty:
            pass

        except KeyboardInterrupt:
            break

        except Exception:
            if not job:
                raise


def runWorkers(partial_func, n_processes, jobs):
    """
    This script is a queuing system that respects KeyboardInterrupt.
    Adapted from: https://github.com/ikreymer/cdx-index-client
    Which in turn was adapted from: http://bryceboe.com/2012/02/14/python-multiprocessing-pool-and-keyboardinterrupt-revisited/
    
    Args:
        partial_func: Function to be run (established with functools.partial())
        n_processes: Number of parallel processes
        jobs: List of individual inputs for partial_func (e.g. a list of Sentinel-2 images)
    """  
    
    # Queue up all jobs
    job_queue = multiprocessing.Queue()
    counter = multiprocessing.Value('i', 0)
    
    for job in jobs:
        job_queue.put(job)
    
    workers = []
    
    for i in range(0, n_processes):
        
        tmp = multiprocessing.Process(target=_do_work, args=(job_queue, partial_func, counter))
        tmp.daemon = True
        tmp.start()
        workers.append(tmp)

    try:
        
        for worker in workers:
            worker.join()
            
    except KeyboardInterrupt:
        for worker in workers:
            print('Keyboard interrupt (ctrl-c) detected. Exiting all processes.')
            # This is an impolite way to kill sen2cor, but it otherwise does not listen.
            parent = psutil.Process(worker.pid)
            children = parent.children(recursive=True)
            parent.send_signal(signal.SIGKILL)
            for process in children:
                process.send_signal(signal.SIGKILL)
            worker.terminate()
            worker.join()
            
        raise


def runCommand(command, verbose = False):
    """
    Function to run command line tool run a 'command' and tidily capture KeyboardInterrupt.
    Idea from: https://stackoverflow.com/questions/38487972/target-keyboardinterrupt-to-subprocess

    Args:
        command: A list containing a command for subprocess.Popen().
        verbose: Set True to print command progress
    """
    
    try:
        p = None

        # Register handler to pass keyboard interrupt to the subprocess
        def handler(sig, frame):
            if p:
                p.send_signal(signal.SIGINT)
            else:
                raise KeyboardInterrupt
                
        signal.signal(signal.SIGINT, handler)
        
        p = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        
        # Optionally print progress, skipping out with KeyboardInterrupt
        if verbose:
            while True:
                stdout_line = p.stdout.readline().decode('utf-8')
                if stdout_line == '' and p.poll() is not None:
                    break
                if stdout_line:
                    print(stdout_line)
        
        text = p.communicate()[0]
                
        if p.wait():
            raise Exception('Command failed: %s'%' '.join(command))
        
    finally:
        # Reset handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    return text.decode('utf-8').split('/n')

