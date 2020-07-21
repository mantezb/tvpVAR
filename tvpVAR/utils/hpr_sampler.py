import numpy as np
from scipy.stats import norm
import numpy.linalg as lin
from tvpVAR.utils.tnr import tnr
from typing import Tuple


def hpr_sampler(lam20:np.ndarray, y: np.ndarray, g: np.ndarray, bigSig: np.ndarray, om: np.ndarray,
                tau2: np.ndarray, mu=None, pi0=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function samples omega's and the hyperparameters conditional on all
    other parameters and data.  If no mu's are provided, they are assumed to
    be zero.  If prior probabilities of exclusion (pi0) are provided,
    sampling will proceed without the hierarhical lasso shrinkage.
    :param lam20:
    :param y:
    :param g:
    :param bigSig:
    :param om:
    :param tau2:
    :param mu:
    :param pi0:
    :return om, om_st, tau2, lam2:
    """
    m = len(om)
    om_st = np.zeros((m, 1))
    lasso = True
    lam2 = None

    if mu is None:
        # for lasso specification
        # mu_j = 0 and pi0_j = .5
        mu = np.zeros((m, 1))

    if pi0 is None:
        # if mu is given with lasso prior
        # set pi0_j = Phi(-mu_j/tau_j)
        pi0 = norm.cdf(-mu/np.sqrt(tau2))
    else:
        # if pi0 are provided, don't use lasso prior
        # note: this is reserved for testing only
        lasso = False

    if lasso:
        # sample lam2 first
        lam2 = np.random.gamma(lam20[0] + m, 1 / lam20[2] + np.sum(tau2, axis=0) / 2)

    for j in range(m):
        nj = np.hstack((np.array(list(range(j))), np.array(list(range(j + 1, m)))))
        g_nj = g[:, nj]
        g_j = g[:, j]
        vj = y - g_nj @ om[nj]

        # compute posterior quantities
        tau2j_hat = 1 / tau2[j] + g_j.T @ bigSig @ g_j
        muj_hat = tau2j_hat @ (mu[j] / tau2[j] + g_j.T @ bigSig @ vj)

        # compute posterior probability of w_j = 0
        ln_xi0 = np.log(1 - pi0[j]) - np.log(pi0[j])
        ln_xi1 = np.log(norm.cdf(muj_hat) / np.sqrt(tau2j_hat)) + np.log(tau2j_hat) / 2
        ln_xi2 = np.log(norm.cdf(mu[j]) / np.sqrt(tau2[j])) + np.log(tau2[j]) / 2
        ln_xi3 = ((muj_hat**2) / tau2j_hat - (mu[j]**2) / tau2[j]) / 2
        pi_j = 1 / (1 + np.exp(ln_xi0 + ln_xi1 - ln_xi2 + ln_xi3))

        om_st[j] = tnr(muj_hat, np.sqrt(tau2j_hat), 0, np.inf)
        if pi_j > np.random.rand(1):
            om_st[j] = tnr(mu[j], np.sqrt(tau2[j]), np.NINF, 0)
        om[j] = om_st[j] * (om_st[j] > 0)

        if lasso:
            tau2[j] = 1 / np.random.wald(np.sqrt(lam2) / np.abs(om_st[j] - mu[j]), lam2)

        return om, om_st, tau2, lam2














