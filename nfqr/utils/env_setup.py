import multiprocessing
import os


def setup_env():

    multiprocessing.set_start_method("fork", force=True)

    for key, value in (("task_id", "0"), ("job_id", "0"), ("num_tasks", "1")):

        if key not in os.environ:
            os.environ[key] = value
