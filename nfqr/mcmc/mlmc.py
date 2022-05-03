import os
import subprocess

from nfqr.mcmc.config import MLMC_PATH


def setup(system="linux"):
    if not (MLMC_PATH / "mlmcpathintegral").is_dir():
        subprocess.run(
            f"git clone https://bitbucket.org/em459/mlmcpathintegral.git {MLMC_PATH}/mlmcpathintegral",
            shell=True,
        )

    subprocess.run(
        f"cp {MLMC_PATH}/mlmcpathintegral/local_{system}.mk {MLMC_PATH}/mlmcpathintegral/local.mk",
        shell=True,
    )

    subprocess.run(f"make -C {MLMC_PATH}/mlmcpathintegral/ -f Makefile all", shell=True)


def print_par_file(
    lat_shape, T_final, mom_inertia, n_burnin_steps, n_steps, dt_hmc, mlmc_dir=MLMC_PATH
):

    with open(
        os.path.join(mlmc_dir / "mlmcpathintegral", "parameters_qm_template.in"), "r"
    ) as rest_pars:
        original_pars = rest_pars.readlines()
        new_pars = original_pars.copy()

    exchange_list = [
        f"M_lat = {lat_shape}\n",
        f"T_final = {T_final}\n",
        f"n_autocorr_window = {100}\n",
        f"n_min_samples_qoi = {1000}\n",
        f"m0 = {mom_inertia}\n",
        "renormalisation = 'none'\n",
        f"n_burnin = {n_burnin_steps}\n",
        f"n_samples = {n_steps}\n",
        "epsilon = 1.0E-1\n",
        # "sampler = 'HMC'\n",
        "nt = 100\n",
        f"dt = {dt_hmc}\n",
        "n_burnin = 1000\n",
        "n_rep = 1\n",
    ]

    for idx, o_line in enumerate(original_pars):
        for value in exchange_list:
            if value.split("=")[0] + "=" in o_line:
                new_pars[idx] = value

    par_file_path = mlmc_dir / "parameters.in"

    with open(par_file_path, "w") as pars_in:
        pars_in.writelines(new_pars)

    return par_file_path


def run_mlmc(mlmc_path=MLMC_PATH):

    print_par_file()

    subprocess.run(f"{mlmc_path}/build/driver_qm {mlmc_path}/parameters.in")

"    read file
"