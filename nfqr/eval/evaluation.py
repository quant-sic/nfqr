import os
import shutil
from typing import Dict, List, Literal, Optional, Union

import torch
from pydantic import root_validator, validator

from nfqr.config import BaseConfig
from nfqr.globals import TMP_DIR
from nfqr.mcmc.nmcmc import NeuralMCMC
from nfqr.nip import NeuralImportanceSampler, calc_ess_q_from_unnormalized_log_weights
from nfqr.target_systems import OBSERVABLE_REGISTRY


class EvalConfig(BaseConfig):

    _name: str = "eval_config"

    n_iter: List[int]=1
    batch_size: List[int]=10000

    methods:List[Literal["nip","nmcmc"]] = ["nip","nmcmc"]
    observables: List[OBSERVABLE_REGISTRY.enum] = ["Chi_t"]

    @validator("observables","methods", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v

    @validator("n_iter","batch_size", pre=True)
    @classmethod
    def int_to_list(cls, v):
        if isinstance(v, int):
            return [v]
        if not isinstance(v,(list,int)):
            raise ValueError("n_iter and batch_size must be int or list of ints")

        return v

    @root_validator
    @classmethod
    def match_n_iter_and_batch_size(cls,values):
        lengths = set([len(values["n_iter"]),len(values["batch_size"])])

        if not len(lengths)==1:
            if 1 in lengths:
                for key in ("n_iter","batch_size"):
                    if len(values[key]) == 1:
                        values[key] = values[key]*max(lengths)
        
        return values

class EvalResult(BaseConfig):

    _name: str = "eval_result"

    observables: List[OBSERVABLE_REGISTRY.enum]

    n_samples: Union[int,List[int]]

    nip: Optional[
        Union[
            List[Dict[OBSERVABLE_REGISTRY.enum, Dict[str, float]]],
            Dict[OBSERVABLE_REGISTRY.enum, Dict[str, float]]
            ]
        ]
    nmcmc: Optional[
        Union[
            List[Dict[Union[str, OBSERVABLE_REGISTRY.enum], Union[float, Dict[str, float]]]]
            ,Dict[Union[str, OBSERVABLE_REGISTRY.enum], Union[float, Dict[str, float]]]
            ]
        ]

    exact_sus: Optional[float]

    @validator("observables", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v


def get_tmp_path_from_name_and_environ(name):

    tmp_path = TMP_DIR / "{}/{}/{}".format(
        os.environ["job_id"], os.environ["task_id"], name
    )
    return tmp_path


def estimate_ess_nip(model, target, batch_size, n_iter):

    model.eval()

    rec_tmp = get_tmp_path_from_name_and_environ("estimate_nip")

    nip_sampler = NeuralImportanceSampler(
        model=model,
        target=target,
        n_iter=n_iter,
        batch_size=batch_size,
        observables=[],
        out_dir=rec_tmp,
    )

    with torch.no_grad():

        nip_sampler.run()
        ess_q = calc_ess_q_from_unnormalized_log_weights(
            nip_sampler.unnormalized_log_weights
        )

    shutil.rmtree(rec_tmp)

    return ess_q


def estimate_obs_nip(model, target, observables, batch_size, n_iter):

    model.eval()

    rec_tmp = get_tmp_path_from_name_and_environ("estimate_obs_nip")

    nip_sampler = NeuralImportanceSampler(
        model=model,
        target=target,
        n_iter=n_iter,
        batch_size=batch_size,
        observables=observables,
        out_dir=rec_tmp,
    )

    with torch.no_grad():

        nip_sampler.run()
        stats = nip_sampler.get_stats()

    shutil.rmtree(rec_tmp)

    return stats


def estimate_nmcmc_acc_rate(model, target, trove_size, n_steps):

    rec_tmp = get_tmp_path_from_name_and_environ("estimate_nmcmc_acc_rate")

    nmcmc = NeuralMCMC(
        model=model,
        target=target,
        trove_size=trove_size,
        n_steps=n_steps,
        observables=[],
        out_dir=rec_tmp,
    )
    nmcmc.run()

    shutil.rmtree(rec_tmp)

    return nmcmc.acceptance_rate


def estimate_obs_nmcmc(model, observables, target, trove_size, n_steps):

    rec_tmp = get_tmp_path_from_name_and_environ("estimate_obs_nmcmc")

    nmcmc = NeuralMCMC(
        model=model,
        target=target,
        trove_size=trove_size,
        n_steps=n_steps,
        observables=observables,
        out_dir=rec_tmp,
    )

    with torch.no_grad():

        nmcmc.run()
        stats = nmcmc.get_stats()

    shutil.rmtree(rec_tmp)

    return stats
