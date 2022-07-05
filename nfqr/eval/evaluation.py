import os
import shutil
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import torch
from pydantic import root_validator, validator

from nfqr.config import BaseConfig
from nfqr.data import ConditionConfig, MCMCConfig, PSampler, TrajectorySamplerConfig
from nfqr.globals import TMP_DIR
from nfqr.mcmc.initial_config import InitialConfigSamplerConfig
from nfqr.mcmc.nmcmc import NeuralMCMC
from nfqr.nip import (
    NeuralImportanceSampler,
    calc_ess_p_from_unnormalized_log_weights,
    calc_ess_q_from_unnormalized_log_weights,
)
from nfqr.target_systems import OBSERVABLE_REGISTRY
from nfqr.target_systems.rotor import RotorTrajectorySamplerConfig
from nfqr.utils import create_logger
from ray import tune

logger = create_logger(__name__)


class EvalConfig(BaseConfig):

    _name: str = "eval_config"

    n_iter: List[int] = 1
    batch_size: List[int] = 10000

    methods: List[Literal["nip", "nmcmc"]] = ["nip", "nmcmc"]
    observables: List[OBSERVABLE_REGISTRY.enum] = ["Chi_t"]

    @validator("observables", "methods", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v

    @validator("n_iter", "batch_size", pre=True)
    @classmethod
    def int_to_list(cls, v):
        if isinstance(v, int):
            return [v]
        if not isinstance(v, (list, int)):
            raise ValueError("n_iter and batch_size must be int or list of ints")

        return v

    @root_validator
    @classmethod
    def match_n_iter_and_batch_size(cls, values):
        lengths = set([len(values["n_iter"]), len(values["batch_size"])])

        if not len(lengths) == 1:
            if 1 in lengths:
                for key in ("n_iter", "batch_size"):
                    if len(values[key]) == 1:
                        values[key] = values[key] * max(lengths)

        return values


ObsStats = Dict[Union[str, OBSERVABLE_REGISTRY.enum], Dict[str, float]]
EvalStats = Union[
    List[Dict[str, Union[ObsStats, float, int]]], Dict[str, Union[ObsStats, float, int]]
]


class EvalResult(BaseConfig):

    _name: str = "eval_result"

    observables: List[OBSERVABLE_REGISTRY.enum]

    n_samples: Union[int, List[int]]

    nip: Optional[EvalStats]
    nmcmc: Optional[EvalStats]

    exact_sus: Optional[float]

    @validator("observables", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v


def get_tmp_path_from_name_and_environ(name):

    task_dir  = TMP_DIR / "{}/{}".format(os.environ["job_id"],os.environ["task_id"])

    if "tune" in os.environ and os.environ["tune"]=="ray":
        task_dir = task_dir/tune.get_trial_id()

    tmp_path = task_dir / "{}".format(name)

    return tmp_path


def estimate_ess_p_nip(
    model,
    data_sampler,
    target,
    batch_size,
    n_iter,
    cut_quantiles=([0, 1], [0.05, 1], [0.1, 1]),
):

    model.eval()

    rec_tmp = get_tmp_path_from_name_and_environ("estimate_nip_ess_p")

    nip_sampler = NeuralImportanceSampler(
        model=model,
        target=target,
        n_iter=n_iter,
        batch_size=batch_size,
        observables=[],
        out_dir=rec_tmp,
        mode="p",
        sampler=data_sampler,
    )

    with torch.no_grad():

        nip_sampler.run()
        ess_p_dict = {}

        for _cut_quantiles in cut_quantiles:
            ess_p_dict[
                f"{_cut_quantiles[0]}-{_cut_quantiles[1]}"
            ] = calc_ess_p_from_unnormalized_log_weights(
                nip_sampler.unnormalized_log_weights, cut_quantiles=_cut_quantiles
            )

    shutil.rmtree(rec_tmp)

    return ess_p_dict


def estimate_ess_q_nip(model, target, batch_size, n_iter):

    model.eval()

    rec_tmp = get_tmp_path_from_name_and_environ("estimate_nip_ess_q")

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


def get_ess_p_sampler(dim, action_config, batch_size):

    mcmc_sampler_config = TrajectorySamplerConfig(
        trajectory_sampler_config=MCMCConfig(
            mcmc_alg="cluster",
            mcmc_type="wolff",
            observables="Chi_t",
            n_steps=1,
            dim=dim,
            action_config=action_config,
            n_burnin_steps=25000,
            n_traj_steps=3,
            out_dir=Path("./"),
            initial_config_sampler_config=InitialConfigSamplerConfig(
                trajectory_sampler_config=RotorTrajectorySamplerConfig(
                    dim=dim, traj_type="hot"
                )
            ),n_replicas=1
        ),
        condition_config=ConditionConfig(),
        batch_size=5000,
    )

    p_sampler = PSampler(
        trajectory_sampler_configs=[mcmc_sampler_config],
        batch_size=batch_size,
        elements_per_dataset=250000,
        subset_distribution=[1.0],
        num_workers=1,
        shuffle=True,
    )

    return p_sampler
