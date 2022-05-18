import itertools
import math
from timeit import timeit

import torch

from nfqr.mcmc.hmc.hmc_cpp import hmc_cpp
from nfqr.target_systems.rotor.rotor import QuantumRotor, TopologicalSusceptibility


def test_qr_action_single_config():

    betas = [0.5 * i for i in range(1, 10)]
    sizes = [10 * i for i in range(1, 10)]

    for size, beta in itertools.product(sizes, betas):

        cpp_qr = hmc_cpp.QR(beta)
        py_qr = QuantumRotor(beta=beta)

        config = torch.rand(size, dtype=torch.float64) * 2 * math.pi

        assert torch.allclose(
            torch.tensor(cpp_qr.evaluate_single_config(config), dtype=torch.float64),
            py_qr.evaluate(config),
            atol=1e-8,
        )


def test_force_single_config():

    betas = [0.5 * i for i in range(1, 10)]
    sizes = [10 * i for i in range(1, 10)]

    for size, beta in itertools.product(sizes, betas):

        cpp_qr = hmc_cpp.QR(beta)
        py_qr = QuantumRotor(beta=beta)

        config = torch.rand(size, dtype=torch.float64) * 2 * math.pi

        assert torch.allclose(
            cpp_qr.force_single_config(config),
            py_qr.force(config),
            atol=1e-8,
        )


def test_sus_single_config():

    sizes = [10 * i for i in range(1, 10)]

    for size in itertools.product(sizes):

        cpp_sus = hmc_cpp.TopologicalSusceptibility()
        py_sus = TopologicalSusceptibility()

        config = torch.rand(size, dtype=torch.float64) * 2 * math.pi

        assert torch.allclose(
            torch.tensor(cpp_sus.evaluate_single_config(config), dtype=torch.float64),
            py_sus.evaluate(config),
            atol=1e-8,
        )


def test_qr_action_batch():

    betas = [0.5 * i for i in range(1, 10)]
    sizes = [10 * i for i in range(1, 10)]
    batch_sizes = [10**i for i in range(4)]

    for size, beta, batch_size in itertools.product(sizes, betas, batch_sizes):

        cpp_qr = hmc_cpp.QR(beta)
        py_qr = QuantumRotor(beta=beta)

        config = torch.rand(batch_size, size, dtype=torch.float64) * 2 * math.pi

        assert torch.allclose(
            cpp_qr.evaluate_batch(config),
            py_qr.evaluate(config),
            atol=1e-8,
        )


def test_force_batch():

    betas = [0.5 * i for i in range(1, 10)]
    sizes = [10 * i for i in range(1, 10)]
    batch_sizes = [10**i for i in range(4)]

    for size, beta, batch_size in itertools.product(sizes, betas, batch_sizes):

        cpp_qr = hmc_cpp.QR(beta)
        py_qr = QuantumRotor(beta=beta)

        config = torch.rand(batch_size, size, dtype=torch.float64) * 2 * math.pi

        assert torch.allclose(
            cpp_qr.force_batch(config),
            py_qr.force(config),
            atol=1e-8,
        )


def test_sus_batch():

    sizes = [10 * i for i in range(1, 10)]
    batch_sizes = [10**i for i in range(4)]

    for size, batch_size in itertools.product(sizes, batch_sizes):

        cpp_sus = hmc_cpp.TopologicalSusceptibility()
        py_sus = TopologicalSusceptibility()

        config = torch.rand(batch_size, size, dtype=torch.float64) * 2 * math.pi

        assert torch.allclose(
            cpp_sus.evaluate_batch(config),
            py_sus.evaluate(config),
            atol=1e-8,
        )


#%%
from nfqr.mcmc.hmc.test_cpp_compatibility import (
    test_force_batch,
    test_force_single_config,
    test_qr_action_batch,
    test_qr_action_single_config,
    test_sus_batch,
    test_sus_single_config,
)

test_qr_action_single_config()
test_qr_action_batch()
test_force_batch()
test_force_single_config()
test_sus_single_config()
test_sus_batch()
# %%

import math
import timeit

import torch

from nfqr.mcmc.hmc.hmc import hmc_cpp

cpp_qr = hmc_cpp.QR(1.0)

for size in [20, 40, 60, 80, 100]:
    config = torch.rand(size, dtype=torch.float64) * 2 * math.pi
    print(
        timeit.timeit(lambda: cpp_qr.force_single_config(config), number=10000) / 10000
    )

for size in [20, 40, 60, 80, 100]:
    config = torch.rand(size, dtype=torch.float64) * 2 * math.pi
    print(
        timeit.timeit(lambda: cpp_qr.evaluate_single_config(config), number=10000)
        / 10000
    )

# %%
