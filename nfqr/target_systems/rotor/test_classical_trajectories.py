import itertools

import numpy as np
from classical_trajectories import DiscreteClassicalRotorTrajectorySampler
from observable import TopologicalCharge


def test_discrete_classical_trajectory_k_values():

    for dim, k in itertools.product(
        [5, 10, 25], [-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10]
    ):
        sampler = DiscreteClassicalRotorTrajectorySampler(
            dim=[dim], batch_size=10, k=[k]
        )

        try:
            trajectories = sampler.sample(device="cpu")
        except ValueError:
            continue

        charges = TopologicalCharge().evaluate(trajectories)

        assert abs(k) <= int(dim / 2)
        assert all(np.allclose(charge, k) for charge in charges), charges


#%%
test_discrete_classical_trajectory_k_values()
