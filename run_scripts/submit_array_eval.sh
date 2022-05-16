#!/usr/bin/env bash
# $ -l cuda=1 
# $ -l h="!"node34"&!"node30"&!"node47"&!"node44"&!"node03
#$ -binding linear:4  # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)
#$ -N running       # set consistent base name for output and error file (allows for easy deletion alias)
#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)
#$ -cwd        # change working directory (to current)
#$ -V          # provide environment variables
#$ -o /home/dechentf/MA/nfqr/pipe_out/$JOB_ID/
#$ -e /home/dechentf/MA/nfqr/pipe_out/$JOB_ID/
#$ -t 1-15


export num_tasks=$SGE_TASK_LAST
export task_id=$((SGE_TASK_ID - 1))
export job_id=$JOB_ID
export CUDA=1

source /home/dechentf/MA/nfqr/nfqr-env/bin/activate

echo /home/dechentf/MA/nfqr/nfqr-env/bin/python3 /home/dechentf/MA/nfqr/nfqr/script/eval.py --exp_dir $1 --observables $2

/home/dechentf/MA/nfqr/nfqr-env/bin/python3 /home/dechentf/MA/nfqr/nfqr/script/eval.py --exp_dir $1 --observables $2 --n_iter $3