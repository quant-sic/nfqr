from __future__ import print_function

import logging
import math

import numpy as np
import torch
from scipy.special import gammainc
from torch.utils import cpp_extension
from tqdm.auto import tqdm

from nfqr.globals import REPO_ROOT
from nfqr.utils.misc import create_logger

err_rho_cpp = cpp_extension.load(
    name="err_rho_cpp", sources=[REPO_ROOT / "nfqr/mcmc/ac/err_rho.cpp"], verbose=False
)

logger = create_logger(__name__)


class ReplicaNumError(Exception):
    pass


class NoFilesError(Exception):
    pass


class FileFormatError(Exception):
    pass


class TooManyArgsError(Exception):
    pass


class GammaPathologicalError(Exception):
    pass


class NoFluctuationsError(Exception):
    pass


class UnknownQuantityError(Exception):
    pass


class ValidationError(Exception):
    pass


class ParamError(Exception):
    pass


class EvalError(Exception):
    pass


class YAMLFormatError(Exception):
    pass


class WrongDataShape(Exception):
    pass


def all_equal(iterable):
    """
    This function returns True if the elements of iterable are all equal.
    """
    it = iter(iterable)

    try:
        first = next(it)
    except StopIteration:
        return True

    return all(first == elem for elem in it)


# -------------------------------------------------------------------------------
#       Compute gradient of derived quantity
# -------------------------------------------------------------------------------


def grad(f, abb, h):
    num_obs = abb.shape[0]
    fgrad = np.zeros(num_obs)
    ainc = abb.copy()

    for alpha in range(num_obs):
        if h[alpha] != 0:
            ainc[alpha] = abb[alpha] + h[alpha]
            fgrad[alpha] = f(ainc)
            ainc[alpha] = abb[alpha] - h[alpha]
            fgrad[alpha] -= f(ainc)
            ainc[alpha] = abb[alpha]
            fgrad[alpha] /= 2 * h[alpha]

    return fgrad


# -------------------------------------------------------------------------------
#       Error estimates of rho
# -------------------------------------------------------------------------------


def err_rho(N, t_max, w_opt, rho):
    """
    Takes:
        - N: total number of measurements.
        - t_max: parameter to truncate the series.
        - w_opt: optimal windowing value.
        - rho: normalised autocorrelation vector.

    Returns a vector with the error on the normalised autocorrelation.
    """

    if t_max*w_opt >=1e10:
        logger.info("Skipping AC error calculation, as Chain size is too large")
        return np.zeros(t_max + 1)
    else:
        logger.info("AC error calculation with ~{} memory accesses".format(t_max*w_opt))
        err_rho_out = err_rho_cpp.err_rho(
            N, t_max, w_opt, torch.from_numpy(rho).clone().to(torch.float32)
        ).numpy()

        # ext_rho = np.zeros(2 * t_max + w_opt + 1)
        # err_rho = np.zeros(t_max + 1)
        # ext_rho[: t_max + 1] = rho[:]

        # for w in tqdm(range(t_max + 1), desc="Error rho"):
        #     for k in range(max(1, w - w_opt), w + w_opt + 1):
        #         err_rho[w] += (
        #             ext_rho[k + w] + ext_rho[abs(k - w)] - 2.0 * ext_rho[w] * ext_rho[k]
        #         ) ** 2

        #     err_rho[w] = math.sqrt(err_rho[w] / N)

        return err_rho_out


# -------------------------------------------------------------------------------
#       Error estimates of integrated autocorrelation time (tau)
# -------------------------------------------------------------------------------


def err_tau(N, t_max, tau_int_fbb):
    """
    Takes:
        - N: total number of measurements.
        - t_max: parameter to truncate the series.
        - tau_int_fbb: autocorrelation time.

    Returns the error of the autocorrelation time.
    """
    nf = float(N)
    err_tau = np.zeros(t_max + 1)
    err_tau = 2.0 * tau_int_fbb * np.sqrt(np.arange(t_max + 1) / nf)

    return err_tau


# -------------------------------------------------------------------------------
#       Bias cancellation for the derived quantity
# -------------------------------------------------------------------------------


def cancel_bias(fbb, fbr, fb, n_rep, sigma_f):
    """
    Takes:
        - fbb: mean values or derived value.
        - fbr: means computed on replica.
        - fb: weighed mean of replica means.
        - n_rep: list of replica sizes.
        - sigma_f: error of derived or primary.

    Returns fbb, fbr, fb without bias.

    For further details on this function see the article.
    """

    r = len(n_rep)
    n = sum(n_rep)

    if r >= 2:
        bf = (fb - fbb) / (r - 1)
        fbb -= bf
        if abs(bf) > sigma_f / 4:
            bias = bf / sigma_f
            logger.warning(
                "A {} sigma bias of the mean has been cancelled!".format(bias)
            )
        fbr -= bf * n / n_rep
        fb -= bf * r

    return (fbb, fbr, fb)


# -------------------------------------------------------------------------------
#       Q value computation
# -------------------------------------------------------------------------------


def q_val(fbr, fb, n_rep, cfbb_opt):
    """
    Takes:
        - fbr: means computed on replica.
        - fb: weighed mean of replica means.
        - n_rep: list of replica sizes.
        - cfbb_opt: refined estimate of projected autocorrelation.

    Returns the Q-value of the replica distribution: goodness of fit to constant.
    """

    r = len(n_rep)

    if r >= 2:
        chisq = np.dot((fbr - fb) ** 2, n_rep) / cfbb_opt
        qval = 1.0 - gammainc((r - 1) * 0.5, chisq * 0.5)
    else:
        qval = 0.0

    return qval


# -------------------------------------------------------------------------------
#       Implementation of the Gamma-method
# -------------------------------------------------------------------------------


def gamma(fbb, fbr, fb, delpro, r, n, n_rep, stau=1.5, rep_equal=False):
    """
    Takes:
        - obs: is a dictionary containing the mean values over all the data (fbb),
               over each replicum (fbr), over all the replica (fb) and
               the deviation from the mean value (delpro).
        - stau: guess for the ratio of tau/tauint. If 0, no autocorrelation is assumed.

    This function returns a dictionary with the results of the analysis:
        - W_opt: optimal windowing
        - t_max: maximum fixed time for the analysis
        - value: unbiased expectation value
        - valr: unbiased expectation value over each replicum
        - valb: unbiased expectation value over all replica,
        - dvalue: statiscal error of value,
        - ddvalue: error of dvalue,
        - tau_int: integrated autocorrelation time at W_opt,
        - dtau_int: error of tau_int,
        - tau_int_fbb: partial autocorrelation times.
        - dtau_int_fbb: error of tau_int_fbb.
        - rho: normalised autocorrelation function.
        - drho: error of rho.
        - qval: Q-value.
    """

    if stau == 0:  # No autocorrelations assumed.
        w_opt = 0
        t_max = 0
        flag = False
    else:
        t_max = min(n_rep) // 2  # Do not take t larger than this.
        # // needed to take the floor of the division.
        flag = True
        g_int = 0.0

    if rep_equal:
        weight = None
    else:
        weight = n_rep

    gamma_fbb = np.zeros(t_max + 2, dtype=np.float64)

    # values for W=0:
    gamma_fbb[1] = (delpro * delpro).mean()

    variance = gamma_fbb[1] * n / (n - 1)
    naive_err = np.sqrt(variance / n)

    # sick case:
    if gamma_fbb[1] == 0:
        raise NoFluctuationsError("Data without fluctuations")

    t = 1

    logger.debug("t = %4d;\tGammaFbb = %.15e", t, gamma_fbb[t])

    with tqdm(total=t_max, desc="Calculating gamma_fbb") as pbar:
        while t <= t_max:
            for i in range(r):
                gamma_fbb[t + 1] += np.sum(
                    delpro[i, 0 : n_rep[i] - t] * delpro[i, t : n_rep[i]]
                )

            gamma_fbb[t + 1] /= n - r * t

            # Automatic windowing procedure

            if flag:
                g_int += gamma_fbb[t + 1] / gamma_fbb[1]  # g_int(W) = tau_int(W) - 0.5

                if g_int <= 0.0:  # No autocorrelation
                    tauw = np.spacing(1.0)  # Setting tau(W) to a tiny positive value
                else:
                    tauw = stau / (np.log((g_int + 1) / g_int))  # è uguale a eq 20'beto

                gw = np.exp(-t / tauw) - tauw / np.sqrt(n * t)

                if (
                    gw < 0.0
                ):  # g(W) has a minimum and this value of t is taken as the optimal value of W
                    w_opt = t
                    t_max = min(t_max, 2 * t)
                    flag = False  # Gamma up to t_max

            t += 1
            pbar.update(1)
    # while-loop end

    logger.debug("t = %4d;\tGammaFbb = %.15e", t, gamma_fbb[t])

    # Here flag is True if windowing failed.
    if flag:
        w_opt = t_max

    gamma_fbb = gamma_fbb[
        1 : t_max + 2
    ]  # chi è gamma_fbb[0]: ora è il vecchio gamma_fbb[1]
    cfbb_opt = gamma_fbb[0] + 2.0 * np.sum(gamma_fbb[1 : w_opt + 1])  # first estimate
    # eq 13 beto
    if cfbb_opt <= 0:
        raise GammaPathologicalError("Gamma pathological: estimated error^2 < 0")

    gamma_fbb += (
        cfbb_opt / n
    )  # bias in Gamma corrected (eq: non numerata dopo eq:19 beto)
    cfbb_opt = gamma_fbb[0] + 2 * np.sum(gamma_fbb[1 : w_opt + 1])  # refined estimate
    # eq 13
    sigma_f = np.sqrt(cfbb_opt / n)  # error of the expectation value of the observables
    # eq 14 beto
    rho = gamma_fbb / gamma_fbb[0]  # normalized autocorrelation function

    logger.debug("Calculating drho")
    drho = err_rho(n, t_max, w_opt, rho)

    tau_int_fbb = np.cumsum(rho) - 0.5
    logger.debug("Err tau")
    dtau_int_fbb = err_tau(n, t_max, tau_int_fbb)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("t = %4d;\trho = %.15e", 1, rho[0])
        logger.debug("t = %4d;\trho = %.15e", len(rho) + 1, rho[-1])

    # answers to be returned:
    logger.debug("Cancel bias")
    (value, valr, valb) = cancel_bias(fbb, fbr, fb, n_rep, sigma_f)

    logger.debug("qval")
    qval = q_val(valr, valb, n_rep, cfbb_opt)

    dvalue = sigma_f
    ddvalue = dvalue * np.sqrt((w_opt + 0.5) / n)  # Statistical error of the error
    tau_int = tau_int_fbb[w_opt]  # Equivalent to: cfbb_opt/2*gamma_fbb[0]
    dtau_int = tau_int * 2 * np.sqrt((w_opt - tau_int + 0.5) / n)

    return {
        "w_opt": w_opt,
        "t_max": t_max,
        "value": value,
        "dvalue": dvalue,
        "ddvalue": ddvalue,
        "tau_int": tau_int,
        "dtau_int": dtau_int,
        "tau_int_fbb": tau_int_fbb,
        "dtau_int_fbb": dtau_int_fbb,
        "variance": variance,
        "naive_err": naive_err,
        "rho": rho,
        "drho": drho,
        "qval": qval,
        "flag": flag,
    }


class Formatter(object):
    def __call__(self, obj):
        return str(obj.__dict__)


class PrimaryFormatter(Formatter):
    def __call__(self, obj):
        stringified = ""

        for alpha in range(obj.num_obs):
            stringified += """\n\nResults for {name}:
         value: {value:.15e}
         error: {dvalue:.15e}
error of error: {ddvalue:.15e}
   naive error: {naive_err:.15e}
      variance: {variance:.15e}
       tau_int: {tau_int:.15e}
 tau_int error: {dtau_int:.15e}
         W_opt: {wopt:d}
         t_max: {tmax:d}
         Q_val: {qval:.15e}
            """.format(
                name=obj.name[alpha],
                value=obj.value[alpha],
                dvalue=obj.dvalue[alpha],
                variance=obj.variance[alpha],
                naive_err=obj.naive_err[alpha],
                ddvalue=obj.ddvalue[alpha],
                tau_int=obj.tau_int[alpha],
                dtau_int=obj.dtau_int[alpha],
                wopt=obj.w_opt[alpha],
                tmax=obj.t_max[alpha],
                qval=obj.qval[alpha],
            )

        return stringified


class DerivedFormatter(Formatter):
    def __call__(self, obj):
        return """\nResults for {name}:
         value: {value:.15e}
         error: {dvalue:.15e}
error of error: {ddvalue:.15e}
   naive error: {naive_err:.15e}
      variance: {variance:.15e}
       tau_int: {tau_int:.15e}
 tau_int error: {dtau_int:.15e}
         W_opt: {wopt:d}
         t_max: {tmax:d}
         Q_val: {qval:.15e}
        """.format(
            name=obj.name,
            value=obj.value,
            variance=obj.variance,
            naive_err=obj.naive_err,
            dvalue=obj.dvalue,
            ddvalue=obj.ddvalue,
            tau_int=obj.tau_int,
            dtau_int=obj.dtau_int,
            wopt=obj.w_opt,
            tmax=obj.t_max,
            qval=obj.qval,
        )


class AnalysisData(object):
    def __init__(self, name=None, formatter=None):
        self.name = name
        self.formatter = formatter if formatter is not None else Formatter()

    def __str__(self):
        return self.formatter(self)


class Analysis(object):
    def __init__(self, data, rep_sizes, name=None, formatter=None):
        if data.ndim != 3:
            raise WrongDataShape("Data object should be a 3-dimensional array")

        self.name = name
        self.formatter = formatter
        self.data = data
        self.size = sum(rep_sizes)
        self.num_rep = data.shape[0]  # Number of replica.
        self.max_rep = data.shape[1]  # Size of the longest replicum.
        self.num_obs = data.shape[2]  # Number of primary observables.
        self.rep_sizes = rep_sizes
        self.rep_equal = all_equal(rep_sizes)
        self.weights = self.rep_sizes if not self.rep_equal else None

        self.results = AnalysisData(name=self.name, formatter=self.formatter)
        self.results.num_obs = self.num_obs

    def mean(self):
        if self.rep_equal:
            abr = np.mean(self.data, axis=1)
            abb = np.mean(abr, axis=0)
        else:
            abr = np.asarray(np.ma.mean(self.data, axis=1))
            abb = np.average(abr, axis=0, weights=self.weights)

        self.results.value = abb  # fbb
        self.results.rep_value = abr  # fbr
        self.results.rep_mean = np.average(abr, weights=self.weights, axis=0)  # fb
        self.results.deviation = self.data - abb  # delpro

        return self.results

    def errors(self, stau=1.5):
        logger.warning(
            "Method 'errors' of '{}' "
            "should implemented by subclasses".format(self.__class__.__name__)
        )
        return self.results


class PrimaryAnalysis(Analysis):
    def __init__(self, data, rep_sizes, name=None):
        super(PrimaryAnalysis, self).__init__(
            data, rep_sizes, name=name, formatter=PrimaryFormatter()
        )

    def errors(self, stau=1.5):
        r = self.num_rep
        n = self.size
        n_rep = self.rep_sizes
        n_alpha = self.num_obs
        fbb = self.results.value
        fbr = self.results.rep_value
        fb = self.results.rep_mean
        delpro = self.results.deviation

        self.results.t_max = [0] * n_alpha
        self.results.w_opt = [0] * n_alpha
        self.results.variance = [0] * n_alpha
        self.results.dvalue = [0] * n_alpha
        self.results.ddvalue = [0] * n_alpha
        self.results.naive_err = [0] * n_alpha
        self.results.tau_int = [0] * n_alpha
        self.results.dtau_int = [0] * n_alpha
        self.results.tau_int_fbb = [0] * n_alpha
        self.results.dtau_int_fbb = [0] * n_alpha
        self.results.rho = [0] * n_alpha
        self.results.drho = [0] * n_alpha
        self.results.qval = [0] * n_alpha

        for alpha in range(n_alpha):
            logger.info("Computing errors for %s", self.name[alpha])

            res = gamma(
                fbb[alpha],
                fbr[:, alpha],
                fb[alpha],
                delpro[:, :, alpha],
                r,
                n,
                n_rep,
                stau,
                self.rep_equal,
            )

            if res["flag"]:
                logger.warning(
                    "Windowing condition failed "
                    "for {} "
                    "up to W = {}".format(self.name[alpha], res["t_max"])
                )

            self.results.t_max[alpha] = res["t_max"]
            self.results.w_opt[alpha] = res["w_opt"]
            self.results.value[alpha] = res["value"]
            self.results.variance[alpha] = res["variance"]
            self.results.dvalue[alpha] = res["dvalue"]
            self.results.ddvalue[alpha] = res["ddvalue"]
            self.results.naive_err[alpha] = res["naive_err"]
            self.results.tau_int[alpha] = res["tau_int"]
            self.results.dtau_int[alpha] = res["dtau_int"]
            self.results.tau_int_fbb[alpha] = res["tau_int_fbb"]
            self.results.dtau_int_fbb[alpha] = res["dtau_int_fbb"]
            self.results.rho[alpha] = res["rho"]
            self.results.drho[alpha] = res["drho"]
            self.results.qval[alpha] = res["qval"]

        return self.results


class DerivedAnalysis(Analysis):
    def __init__(self, data, rep_sizes, name=None):
        super(DerivedAnalysis, self).__init__(
            data, rep_sizes, name=name, formatter=DerivedFormatter()
        )

    def apply(self, f, name=None):
        abb = self.results.value
        abr = self.results.rep_value

        dev = np.ma.std(self.data, axis=1)
        h = np.ma.average(dev, axis=0, weights=self.weights)
        h /= np.sqrt(self.size)

        fbb = f(abb)
        fbr = np.zeros(self.num_rep)

        for i in range(self.num_rep):
            fbr[i] = f(abr[i, :])

        fb = np.average(fbr, axis=0, weights=self.weights)

        fgrad = grad(f, self.results.value, h)

        spec_name = name if name else self.name
        self.applied = AnalysisData(name=spec_name, formatter=self.formatter)
        self.applied.value = fbb
        self.applied.rep_value = fbr
        self.applied.rep_mean = fb
        self.applied.deviation = np.dot(self.results.deviation, fgrad)  # delpro

        return self.applied

    def errors(self, stau=1.5):
        r = self.num_rep
        n = self.size
        n_rep = self.rep_sizes
        n_alpha = self.num_obs
        fbb = self.applied.value
        fbr = self.applied.rep_value
        fb = self.applied.rep_mean
        delpro = self.applied.deviation

        res = gamma(fbb, fbr, fb, delpro, r, n, n_rep, stau, self.rep_equal)

        if res["flag"]:
            if self.name is not None:
                logger.warning(
                    "Windowing condition failed "
                    "for derived observable '{}' "
                    "up to W = {}".format(self.name, res["t_max"])
                )
            else:
                logger.warning(
                    "Windowing condition failed "
                    "for derived observable "
                    "up to W = {}".format(res["t_max"])
                )

        self.applied.t_max = res["t_max"]
        self.applied.w_opt = res["w_opt"]
        self.applied.value = res["value"]
        self.applied.variance = res["variance"]
        self.applied.dvalue = res["dvalue"]
        self.applied.ddvalue = res["ddvalue"]
        self.applied.naive_err = res["naive_err"]
        self.applied.tau_int = res["tau_int"]
        self.applied.dtau_int = res["dtau_int"]
        self.applied.tau_int_fbb = res["tau_int_fbb"]
        self.applied.dtau_int_fbb = res["dtau_int_fbb"]
        self.applied.rho = res["rho"]
        self.applied.drho = res["drho"]
        self.applied.qval = res["qval"]

        return self.applied
