import shutil

import torch

from nfqr.globals import TEMP_DIR
from nfqr.mcmc.nmcmc import NeuralMCMC
from nfqr.nip.nip import NeuralImportanceSampler, calc_ess_q
from nfqr.stats import get_impsamp_statistics, get_mcmc_statistics
from nfqr.target_systems.observable import ObservableRecorder


def estimate_ess_nip(model, target, batch_size, n_iter):

    model.eval()

    nip_sampler = NeuralImportanceSampler(
        model=model, target=target, n_iter=n_iter, batch_size=batch_size
    )
    rec_tmp = TEMP_DIR / "estimate_nip"
    # to avoid erroneously adding onto existing
    if rec_tmp.is_file():
        shutil.rmtree(rec_tmp)

    rec = ObservableRecorder(
        observables={},
        sampler=nip_sampler,
        save_dir_path=rec_tmp,
        stats_function=get_impsamp_statistics,
    )

    with torch.no_grad():

        rec.record_sampler()
        unnormalized_imp_weights = rec.load_imp_weights()
        ess_q = calc_ess_q(unnormalized_weights=unnormalized_imp_weights)

    shutil.rmtree(rec_tmp)

    return ess_q


def estimate_obs_nip(model, target, observables, batch_size, n_iter):

    model.eval()

    nip_sampler = NeuralImportanceSampler(
        model=model, target=target, n_iter=n_iter, batch_size=batch_size
    )
    rec_tmp = TEMP_DIR / "estimate_obs_nip"
    if rec_tmp.is_file():
        shutil.rmtree(rec_tmp)

    rec = ObservableRecorder(
        observables=observables,
        sampler=nip_sampler,
        save_dir_path=rec_tmp,
        stats_function=get_impsamp_statistics,
    )

    with torch.no_grad():

        rec.record_sampler()
        stats = rec.aggregate()

    shutil.rmtree(rec_tmp)

    return stats


def estimate_nmcmc_acc_rate(model, target, trove_size, n_steps):

    nmcmc = NeuralMCMC(
        model=model, target=target, trove_size=trove_size, n_steps=n_steps
    )
    nmcmc.run_entire_chain()

    return nmcmc.acceptance_ratio


def estimate_obs_nmcmc(model, observables, target, trove_size, n_steps):

    nmcmc = NeuralMCMC(
        model=model, target=target, trove_size=trove_size, n_steps=n_steps
    )

    rec_tmp = TEMP_DIR / "estimate_obs_nmcmc"

    if rec_tmp.is_file():
        shutil.rmtree(rec_tmp)

    rec = ObservableRecorder(
        observables=observables,
        sampler=nmcmc,
        save_dir_path=rec_tmp,
        stats_function=get_mcmc_statistics,
    )

    with torch.no_grad():

        rec.record_sampler()
        stats = rec.aggregate()
        stats["acc_rate"] = nmcmc.acceptance_ratio

    shutil.rmtree(rec_tmp)

    return stats
