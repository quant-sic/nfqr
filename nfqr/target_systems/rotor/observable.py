import math

import numpy as np
import torch
from scipy import integrate as sint
from scipy import special as sp_special

from nfqr.registry import StrRegistry
from nfqr.target_systems.observable import Observable

ROTOR_OBSERVABLE_REGISTRY = StrRegistry("qr")


@ROTOR_OBSERVABLE_REGISTRY.register("Q")
class TopologicalCharge(Observable):
    def __init__(self, diffs=False):
        super().__init__()
        self.diffs = diffs

    @classmethod
    def use_diffs(cls):
        return cls(diffs=True)

    def evaluate(self, config):
        if not self.diffs:
            _config = torch.roll(config, shifts=1, dims=-1) - config
        else:
            _config = config

        return ((_config + math.pi) % (2 * math.pi) - math.pi).sum(dim=-1) / (
            2 * math.pi
        )


ROTOR_OBSERVABLE_REGISTRY.register("Q_diffs", TopologicalCharge.use_diffs)


@ROTOR_OBSERVABLE_REGISTRY.register("Chi_t")
class TopologicalSusceptibility(Observable):
    def __init__(self, diffs=False):
        super().__init__()

        self.charge = TopologicalCharge(diffs=diffs)

    @classmethod
    def use_diffs(cls):
        return cls(diffs=True)

    def evaluate(self, config):

        charge = self.charge.evaluate(config)
        susceptibility = charge**2
        return susceptibility


ROTOR_OBSERVABLE_REGISTRY.register("Chi_t_diffs", TopologicalSusceptibility.use_diffs)


class SusceptibilityExact(object):
    """Class for computing  the exact value of topological susceptibility
    (scaled by volume) for the quenched 2d Schwinger model.

    :arg beta: coupling constant beta
    :arg P: number of plaquettes
    """

    def __init__(self, beta, P, accuracy=1e-6):

        self.nmax = self._get_n_max(accuracy=accuracy, beta=beta)
        self.beta = beta
        self.P = P

        Isum = 0
        self._w = {}
        for n in range(-self.nmax, self.nmax + 1):
            rho = sp_special.ive(n, self.beta) / sp_special.ive(0, self.beta)
            self._w[n] = rho**self.P
            Isum += self._w[n]

        for n in range(-self.nmax, self.nmax + 1):
            self._w[n] /= Isum

    def evaluate(self):
        """return value of topological susceptibility"""
        S = 0.0
        for n in range(-self.nmax, self.nmax + 1):
            tmpA = self._ddI(self.beta, n) / self._I(self.beta, n)
            tmpB = self._dI(self.beta, n) / self._I(self.beta, n)
            S += self.P * self._w[n] * (tmpA - (self.P - 1) * tmpB**2)
        return S

    def _I(self, x, n):
        return sp_special.ive(n, x)

    def _dI(self, x, n):
        def integrand(phi):
            return phi * np.sin(n * phi) * np.exp(x * (np.cos(phi) - 1.0))

        intPhi = sint.quad(integrand, -np.pi, np.pi)
        return -1.0 / (4.0 * np.pi**2) * intPhi[0]

    def _ddI(self, x, n):
        def integrand(phi):
            return phi**2 * np.cos(n * phi) * np.exp(x * (np.cos(phi) - 1.0))

        intPhi = sint.quad(integrand, -np.pi, np.pi)
        return 1.0 / (8.0 * np.pi**3) * intPhi[0]

    def _get_n_max(self, accuracy, beta):

        eps = 1
        for i in range(1, 100):
            eps *= np.sqrt(1 + ((i + 0.5) / beta) ** 2) - (i + 0.5) / beta
            if eps < accuracy:
                return i
