import shutil

import torch

from nfqr.config import TEMP_DIR
from nfqr.nip.nip import NeuralImportanceSampler
from nfqr.stats import get_impsamp_statistics
from nfqr.target_systems.observable import ObservableRecorder


def estimate_ess_nip(model, target, batch_size, n_iter):

    nip_sampler = NeuralImportanceSampler(
        model=model, target=target, n_iter=n_iter, batch_size=batch_size
    )
    rec_tmp = TEMP_DIR / "estimate_nip"
    rec = ObservableRecorder(
        observables={},
        sampler=nip_sampler,
        save_dir_path=rec_tmp,
        stats_function=get_impsamp_statistics,
    )

    with torch.no_grad():

        rec.record_sampler()
        unnormalized_imp_weights = rec.load_imp_weights()
        imp_weights = unnormalized_imp_weights / unnormalized_imp_weights.mean()

        ess_q = 1 / (imp_weights**2).mean()

    shutil.rmtree(rec_tmp)

    return ess_q
