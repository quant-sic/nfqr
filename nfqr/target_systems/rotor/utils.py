import numpy as np
from scipy import integrate as sint
from scipy import special as sp_special


class SusceptibilityExact:
    """Class for computing  the exact value of topological susceptibility
    (scaled by volume) for the quenched 2d Schwinger model.

    :arg beta: coupling constant beta
    :arg P: number of plaquettes
    """

    def __init__(self, beta, P):
        self.nmax = 10
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