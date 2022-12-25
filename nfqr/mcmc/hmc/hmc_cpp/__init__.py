from torch.utils import cpp_extension

from nfqr.globals import REPO_ROOT

hmc_cpp = cpp_extension.load(
    name="hmc_cpp",
    sources=[*(REPO_ROOT / "nfqr/mcmc/hmc/hmc_cpp").glob("*.cpp")],
    extra_include_paths=[
        *((REPO_ROOT / "nfqr/mcmc/hmc/hmc_cpp").glob(".h")),
        "/usr/include/eigen-3.4.0",
        "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3",
        "/home/dechentf/eigen-3.4.0",
    ],
    verbose=True,
)

