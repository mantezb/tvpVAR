import numpy as np
from scipy.stats import norm
from tvpVAR.utils.tnr import tnr
from typing import Tuple


def hpr_sampler(lam20: np.ndarray, y: np.ndarray, g: np.ndarray, bigSig: np.ndarray, om: np.ndarray,
                tau2: np.ndarray, mu=None, pi0=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function samples omega's and the hyperparameters conditional on all
    other parameters and data.  If no mu's are provided, they are assumed to
    be zero.  If prior probabilities of exclusion (pi0) are provided,
    sampling will proceed without the hierarhical lasso shrinkage.
    :param lam20: lambda_2
    :param y:
    :param g:
    :param bigSig:
    :param om: omega
    :param tau2: tau^2
    :param mu: mu
    :param pi0: prior probabilities of exclusion (optional, for testing only)
    :return om, om_st, tau2, lam2: omega, omega star, tau^2 and lambda_2
    """
    m = len(om)
    om_st = np.zeros((m, 1))
    lasso = True  # Always use lasso prior unless prior probabilities of exclusion (pi0) are provided
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
        lam2 = np.random.gamma(lam20[0] + m, 1 / (lam20[1] + np.sum(tau2) / 2))

    for j in range(m):
        nj = np.hstack(((np.arange(1, j)), np.arange(j + 1, m)))
        g_nj = g[:, nj]
        g_j = g[:, j].reshape((-1, 1))
        vj = y - g_nj @ om[nj]

        # compute posterior quantities
        tau2j_hat = 1 / (1 / tau2[j] + g_j.T @ bigSig @ g_j).ravel()
        muj_hat = tau2j_hat @ (mu[j] / tau2[j] + g_j.T @ bigSig @ vj)

        # compute posterior probability of w_j = 0
        ln_xi0 = np.log(1 - pi0[j]) - np.log(pi0[j])  # where is this coming in??? not in the paper althout it is = 0 when pi0 = 0.5
        ln_xi1 = np.log(norm.cdf(muj_hat / np.sqrt(tau2j_hat))) + np.log(tau2j_hat) / 2
        ln_xi2 = np.log(norm.cdf(-mu[j] / np.sqrt(tau2[j]))) + np.log(tau2[j]) / 2
        ln_xi3 = ((muj_hat**2) / tau2j_hat - (mu[j]**2) / tau2[j]) / 2
        pi_j = 1 / (1 + np.exp(ln_xi0 + ln_xi1 - ln_xi2 + ln_xi3))  # probability of time invariance (i.e. om_st < 0)

        # if pi_j.item() > 0.5: # TESTING
        if pi_j.item() > np.random.rand(1): # Random Bernoulli draw with probability of time invariance equal to pi_j
            om_st[j, 0] = tnr(mu[j], np.sqrt(tau2[j]), np.NINF, 0)
        else:
            om_st[j, 0] = tnr(muj_hat, np.sqrt(tau2j_hat), 0, np.inf)

        om[j, 0] = om_st[j, 0] * (om_st[j, 0] > 0)

        if lasso:
            tau2[j] = 1 / np.random.wald(np.sqrt(lam2) / np.abs(om_st[j] - mu[j]), lam2)

    return om, om_st, tau2, lam2














