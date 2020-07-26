import numpy as np
from scipy.stats import norm
from scipy.sparse.linalg import spsolve
import scipy.sparse as sps
from sksparse.cholmod import cholesky as chol
from tvpVAR.utils.utils import repmat
#import tvpVAR.utils.settings as settings # TESTING
from typing import Tuple


def mvsvrw(y_star: np.ndarray, h: sps.csc_matrix, iSig: sps.csc_matrix, iVh: sps.csc_matrix) -> Tuple[sps.csc_matrix, np.ndarray]:
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
    #temprand = settings.rand_temprand # TESTING

    q = repmat(pi, tn, 1) * norm.pdf(repmat(y_star, 1, 7),
                                        repmat(h, 1, 7).toarray() + repmat(mi, tn, 1), repmat(sqrtsigi, tn, 1))
    q = q / repmat(np.reshape(np.sum(q, axis=1), (-1, 1)), 1, 7)
    S = 7 - np.reshape(np.sum(np.less(repmat(temprand, 1, 7), np.cumsum(q, axis=1)), axis=1), (-1, 1)) + 1

    # Sample h
    Hh = sps.spdiags(-np.ones(tn-n), -n, tn, tn, format='csc') + sps.eye(tn, format='csc')
    invSh = sps.block_diag((iVh, sps.kron(sps.eye(int(tn / n - 1), format='csc'), iSig, format='csc')), format='csc')
    dconst = np.reshape(np.array([mi[i-1][0] for i in S]), (-1, 1))
    invOmega = sps.spdiags(1/np.array([sigi[i-1][0] for i in S]), 0, tn, tn, format='csc')
    Kh = Hh.T @ invSh @ Hh
    Ph = Kh + invOmega
    Ch = chol(Ph, ordering_method='natural').L().T
    hhat = spsolve(Ph, invOmega @ (y_star - dconst))
    h = sps.csc_matrix(np.reshape(hhat + spsolve(Ch, np.random.randn(tn, 1)), (-1, 1)))
    #h = sps.csc_matrix(np.reshape(hhat + spsolve(Ch, settings.rand_n[0:633]), (-1, 1))) # TESTING


    return h, S




