import multiprocessing
import os
import multiprocessing

def setup_env():

    multiprocessing.set_start_method('fork',force=True)

    for key,value in (("task_id","0"),("job_id","0"),("num_tasks","1")):

        if not key in os.environ:
            os.environ[key] = value
