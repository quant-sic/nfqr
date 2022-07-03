#!/home/dechentf/MA/nfqr/nfqr-env/bin/python3

import subprocess
import os
from argparse import ArgumentParser
from pathlib import Path
import json

from nfqr.globals import TMP_DIR,EXPERIMENTS_DIR
from nfqr.train.config import LitModelConfig
from nfqr.mcmc.config import MCMCConfig

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--script",choice=["train","eval","hmc"], type=Path)
    parser.add_argument("--exp_name",type=bool,default=True)
    parser.add_argument("--cuda",type=bool,default=True)

    args = parser.parse_args()

    script_head_lines = ["#!/usr/bin/env bash",
                                "#$ -binding linear:4  # request 4 cpus (8 with Hyperthreading) (some recommend 4 per GPU)",
                                "#$ -N running       # set consistent base name for output and error file (allows for easy deletion alias)",
                                "#$ -q all.q    # don't fill the qlogin queue (can some add why and when to use?)",
                                "#$ -cwd        # change working directory (to current)",
                                "#$ -V          # provide environment variables",
                                "#$ -o /home/dechentf/MA/nfqr/pipe_out/$JOB_ID/",
                                "#$ -e /home/dechentf/MA/nfqr/pipe_out/$JOB_ID/"
                        ]
    
    if args.cuda:
        script_head_lines += ["#$ -l cuda=1"],


    if args.script=="train":
        script_execution_line = f"/home/dechentf/MA/nfqr/nfqr-env/bin/python3 /home/dechentf/MA/nfqr/nfqr/script/train.py --exp_dir {args.exp_name}"
        num_pars= LitModelConfig.get_num_tasks(EXPERIMENTS_DIR/args.exp_name)

    if args.script=="eval":
        script_execution_line = f"/home/dechentf/MA/nfqr/nfqr-env/bin/python3 /home/dechentf/MA/nfqr/nfqr/script/eval.py --exp_dir {args.exp_name}"
        with json.open(EXPERIMENTS_DIR/args.exp_name) as buff:
            num_pars = len(json.load(buff)["n_iter"])

    if args.script=="hmc":
        script_execution_line = f"/home/dechentf/MA/nfqr/nfqr-env/bin/python3 /home/dechentf/MA/nfqr/nfqr/script/run_mcmc.py --exp_dir {args.exp_name}"
        num_pars = MCMCConfig.get_num_tasks(EXPERIMENTS_DIR/args.exp_name)


    script_head_lines+=[f"#$ -t 1-{num_pars}"]

    script_name = TMP_DIR/(args.exp_name+".sh")
    with open(script_name,"w") as bash_script:
        bash_script.writelines(script_head_lines)

        bash_script.writelines(["export num_tasks=$SGE_TASK_LAST",
                                "export task_id=$((SGE_TASK_ID - 1))",
                                "export job_id=$JOB_ID",
                                "export CUDA=1",
                                "env",
                                "source /home/dechentf/MA/nfqr/nfqr-env/bin/activate"
                                ])

        bash_script.writelines([script_execution_line])

    
    subprocess.run(f"qsub {script_name}",shell=True)
    os.remove(script_name)

        
    