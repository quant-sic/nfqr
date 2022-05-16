import os
import shutil
from typing import Dict, List, Optional, Union

import torch
from pydantic import validator

from nfqr.config import BaseConfig
from nfqr.globals import TEMP_DIR
from nfqr.mcmc import get_mcmc_statistics
from nfqr.mcmc.nmcmc import NeuralMCMC
from nfqr.nip import get_impsamp_statistics
from nfqr.nip.nip import NeuralImportanceSampler, calc_ess_q
from nfqr.target_systems import OBSERVABLE_REGISTRY
from nfqr.target_systems.observable import ObservableRecorder


class EvalConfig(BaseConfig):

    _name: str = "eval_result"

    observables: List[OBSERVABLE_REGISTRY.enum]

    n_samples:int

    nip: Optional[Dict[OBSERVABLE_REGISTRY.enum, Dict[str, float]]]
    nmcmc: Optional[
        Dict[Union[str, OBSERVABLE_REGISTRY.enum], Union[float, Dict[str, float]]]
    ]

    exact_sus: Optional[float]

    @validator("observables", pre=True)
    @classmethod
    def str_to_list(cls, v):
        if isinstance(v, str):
            return v.split(",")

        return v


def estimate_ess_nip(model, target, batch_size, n_iter):

    model.eval()

    rec_tmp = TEMP_DIR / "{}/{}/estimate_nip".format(
        os.environ["job_id"], os.environ["task_id"]
    )
    # to avoid erroneously adding onto existing
    if rec_tmp.is_file():
        shutil.rmtree(rec_tmp)

    rec = ObservableRecorder(
        observables={},
        save_dir_path=rec_tmp,
        stats_function=get_impsamp_statistics,
    )

    nip_sampler = NeuralImportanceSampler(
        model=model,
        target=target,
        n_iter=n_iter,
        batch_size=batch_size,
        observables_rec=rec,
    )

    with torch.no_grad():

        nip_sampler.run()
        unnormalized_imp_weights = rec.load_imp_weights()
        ess_q = calc_ess_q(unnormalized_weights=unnormalized_imp_weights)

    shutil.rmtree(rec_tmp)

    return ess_q


def estimate_obs_nip(model, target, observables, batch_size, n_iter):

    model.eval()

    rec_tmp = TEMP_DIR / "{}/{}/estimate_obs_nip".format(
        os.environ["job_id"], os.environ["task_id"]
    )
    if rec_tmp.is_file():
        shutil.rmtree(rec_tmp)

    rec = ObservableRecorder(
        observables=observables,
        save_dir_path=rec_tmp,
        stats_function=get_impsamp_statistics,
    )

    nip_sampler = NeuralImportanceSampler(
        model=model,
        target=target,
        n_iter=n_iter,
        batch_size=batch_size,
        observables_rec=rec,
    )

    with torch.no_grad():

        nip_sampler.run()
        stats = rec.aggregate()

    shutil.rmtree(rec_tmp)

    return stats


def estimate_nmcmc_acc_rate(model, target, trove_size, n_steps):

    rec_tmp = TEMP_DIR / "{}/{}/estimate_nmcmc_acc_rate".format(
        os.environ["job_id"], os.environ["task_id"]
    )
    if rec_tmp.is_file():
        shutil.rmtree(rec_tmp)

    rec = ObservableRecorder(
        observables={},
        save_dir_path=rec_tmp,
        stats_function=get_impsamp_statistics,
    )

    nmcmc = NeuralMCMC(
        model=model,
        target=target,
        trove_size=trove_size,
        n_steps=n_steps,
        observables_rec=rec,
    )
    nmcmc.run()

    shutil.rmtree(rec_tmp)

    return nmcmc.acceptance_ratio


def estimate_obs_nmcmc(model, observables, target, trove_size, n_steps):

    rec_tmp = TEMP_DIR / "{}/{}/estimate_obs_nmcmc".format(
        os.environ["job_id"], os.environ["task_id"]
    )
    if rec_tmp.is_file():
        shutil.rmtree(rec_tmp)

    rec = ObservableRecorder(
        observables=observables,
        save_dir_path=rec_tmp,
        stats_function=get_mcmc_statistics,
    )

    nmcmc = NeuralMCMC(
        model=model,
        target=target,
        trove_size=trove_size,
        n_steps=n_steps,
        observables_rec=rec,
    )

    with torch.no_grad():

        nmcmc.run()
        stats = rec.aggregate()
        stats["acc_rate"] = nmcmc.acceptance_ratio

    shutil.rmtree(rec_tmp)

    return stats
