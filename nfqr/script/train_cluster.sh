#!/usr/bin/env bash
# $ -binding linear:4  # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
# $ -N running       # set consistent base name for output and error file (allows for easy deletion alias)
# $ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
# $ -cwd        # change working directory (to current)
# $ -V          # provide environment variables
# $ -o /home/dechentf/MA/pipe_out/$JOB_ID/
# $ -e /home/dechentf/MA/pipe_out/$JOB_ID/
# $ -t 1-1

export CUDA=1

source /home/dechentf/MA/ma-env-cu113/bin/activate

/home/dechentf/MA/ma-env-cu113/bin/python3 train.py $1