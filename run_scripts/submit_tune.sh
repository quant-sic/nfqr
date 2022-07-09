#!/usr/bin/env bash
#$ -l cuda=2 
# $ -l h="!"node34"&!"node30"&!"node47"&!"node44"
#$ -binding linear:4  # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N running       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables
#$ -o /home/dechentf/MA/nfqr/pipe_out/$JOB_ID/
#$ -e /home/dechentf/MA/nfqr/pipe_out/$JOB_ID/


export num_tasks=1
export task_id=0
export job_id=$JOB_ID
export CUDA=1

# echo environment
env

# activate python environment
source /home/dechentf/MA/nfqr/nfqr-env/bin/activate

# execute script
/home/dechentf/MA/nfqr/nfqr-env/bin/python3 /home/dechentf/MA/nfqr/nfqr/script/tune.py --exp_dir $1