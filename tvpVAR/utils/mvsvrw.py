import numpy as np
from scipy.stats import norm
from scipy.linalg import block_diag, cholesky
from scipy.sparse.linalg import spsolve
import scipy.sparse as sps
import tvpVAR.utils.settings as settings

import numpy.linalg as lin
from typing import Tuple


def mvsvrw(y_star: np.ndarray, h: np.ndarray, iSig: np.ndarray, iVh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function simulates log-volatilities for a multivariate stochastic
    volatility model with independent random-walk transitions.
    :param y_star:
    :param h:
    :param iSig:
    :param iVh:
    :return h, S:
    """

    n = iSig.shape[0]
    tn = h.shape[0]

    # Normal mixture
    pi = np.array([0.0073, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.2575])
    mi = (np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819]) - 1.2704) # means already adjusted!!
    sigi = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])
    sqrtsigi = np.sqrt(sigi)

    # Sample S from a 7-point discrete distribution
    temprand = np.random.rand(tn, 1)

    q = np.tile(pi, (tn, 1)) * norm.pdf(np.tile(y_star, (1, 7)),
                                        np.tile(h, (1, 7)) + np.tile(mi, (tn, 1)), np.tile(sqrtsigi, (tn, 1)))
    q = q / np.tile(np.reshape(np.sum(q, axis=1), (-1, 1)), (1, 7))
    S = 7 - np.reshape(np.sum(np.tile(temprand, (1, 7)) < np.cumsum(q, axis=1), axis=1), (-1, 1)) + 1

    # Sample h
    Hh = np.diag(-np.ones(tn-n), -n) + np.eye(tn)
    invSh = block_diag(iVh, np.kron(np.eye(int(tn / n - 1)), iSig))
    dconst = np.reshape(np.array([mi[i-1][0] for i in S]), (-1, 1))
    invOmega = np.diag(1/np.array([sigi[i-1][0] for i in S]), 0)
    Kh = Hh.T @ invSh @ Hh
    Ph = Kh + invOmega
    Ch = cholesky(Ph)
    hhat = spsolve(sps.csc_matrix(Ph), invOmega @ (y_star - dconst))
    h = np.reshape(hhat + spsolve(sps.csc_matrix(Ch), np.random.randn(tn, 1)), (-1, 1))


    return h, S




