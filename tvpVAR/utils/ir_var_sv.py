import numpy as np
import timeit
from tqdm import tqdm
import numpy.linalg as lin
from typing import Tuple
import multiprocessing
from joblib import delayed, Parallel


def ir_sim(i:int, s:int, n: int, k: int, p: int, b0t: np.ndarray, ltidx: np.ndarray, cidx: np.ndarray, bidx:np.ndarray, bigj: np.ndarray, sigma: np.ndarray, beta: np.ndarray, t_start:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to estimate impulse responses per single simulation

    :param i: no of simulation
    :param s: impulse response horizon
    :param n: no of variables
    :param k: no of parameters across 3 equtions
    :param p: no of lags
    :param b0t: initial lower triangular matrix to store B0t
    :param ltidx: identifier for 3 free parameters of B0t
    :param cidx:  identifier for lower triangular matrix B0t
    :param bidx:  identifier for all beta coefficients that are not free parameters of B0t
    :param bigj: matrix to identify shocks application
    :param sigma: simulated variance diagonal matrices
    :param beta: simulated beta matrices
    :param t_start: an array of 3 time points of starting values
    :return: impulse responses of first 2 variables to the shock on the third variable, starting at 3 speficied time points
    """
    biga = np.zeros((n * p, n * p))

    for j in range(p - 1):
        biga[(j + 1) * n:n * (j + 2), n * j:(j + 1) * n] = np.eye(n)

    for j in range(beta.shape[1]):
        # Decompose the reduced form cov matrix using lower triangular identification scheme as in Primiceri (2005)
        b0t = b0t.T
        b0t[ltidx.T] = beta[cidx.ravel(), j, i]
        b0t = b0t.T
        ikap = lin.inv(b0t) @ np.sqrt(np.diag((sigma[:, j, i]).ravel()))  # Cholesky of variance covariance matrix for the error, B0t^-1 @ sqrt(sigma_t) where sigma is variance matrix
        # compute impulse response for all combinations
        normkap = np.diag(1/(np.diag(ikap))) @ ikap    # normalize shocks to arrive at % unit initial shock
        bt = np.reshape(beta[bidx.ravel(), j, i], (k, n), order='F').T  # 3 equations, for each variable in VAR
        ct = bt[:, 1:]  # only endogenous variables and their lags (excluding constants)
        biga[0:n, :] = ct
        imp_resp = np.zeros((n, n * (s + 1)))
        imp_resp[0:n, 0:n] = normkap  # First shock is the Cholesky of the VAR covariance
        bigai = biga

        for dt in range(s):
            imp_resp[:, (dt + 1) * n:(dt + 2) * n] = bigj @ bigai @ bigj.T @ normkap
            bigai = bigai @ biga

        if j == t_start[0]:
            impf_m = np.zeros((n, s + 1))
            jj = -1
            for ij in range(s + 1):
                jj = jj + n
                impf_m[:, ij] = imp_resp[:, jj]  # restrict to the n'th equation, the interest rate
            impf_m1= impf_m  # store draws of responses

        if j == t_start[1]:
            impf_m = np.zeros((n, s + 1))
            jj = -1
            for ij in range(s + 1):
                jj = jj + n
                impf_m[:, ij] = imp_resp[:, jj]  # restrict to the n'th equation, the interest rate
            impf_m2 = impf_m  # store draws of responses

        if j == t_start[2]:
            impf_m = np.zeros((n, s + 1))
            jj = -1
            for ij in range(s + 1):
                jj = jj + n
                impf_m[:, ij] = imp_resp[:, jj]  # restrict to the n'th equation, the interest rate
            impf_m3 = impf_m  # store draws of responses

    return impf_m1, impf_m2, impf_m3


def ir_var_sv(beta: np.ndarray, sigma: np.ndarray, cidx: np.ndarray, t_start: np.ndarray, s: int, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function computes impulses responses for the VAR models with
    stochastic volatility. Structural parameters are recovered from the
    estimated parameters using lower triangular identification scheme as per Primiceri (2005)
    recovered from the values simulated by MCMC algorithm by Eisenstat (2016).

    :param beta: coefficients of all deterministic variables (exogenous, endogenous etc), defined as alpha + w @ Phi.T @ gamma
    :param sigma: variance of the additive shock in the measurement equation (volatility^2)
    :param cidx: structure to identify contemporaneous parameters B0t
    :param t_start: starting time point for impulse response simulation (no of row -1)
    :param s: no of periods for impulse response projection
    :param p: no of lags
    :param d: scale adjustment to revert normalisation performed at the start of the simulation
    :return a, b, c, ir, err: free parameters of B0t, impulse response and error identifier
    """
    # Allocate space
    nsims = sigma.shape[2]
    n = sigma.shape[0]
    k = int((beta.shape[0] - 0.5 * n * (n-1)) / n)  # parameters for each VAR equation excluding free parameters of B0t
    bidx = np.array(1-cidx, dtype=bool)

    # For constructing the contemporaneous coefficients
    b0t = np.eye(n)  # B0_t
    ltidx = np.array(np.tril(np.ones((n, n))) - np.eye(n), dtype=bool)  # index to identify free parameters of B0t

    ir1 = np.zeros((n, 1 + s, nsims))
    ir2 = np.zeros((n, 1 + s, nsims))
    ir3 = np.zeros((n, 1 + s, nsims))

    bigj = np.zeros((n, n*2))
    bigj[0:n,0:n] = np.eye(n)

    cores = multiprocessing.cpu_count()
    print('Simulating impulse responses...')

    start = timeit.default_timer()

    res = Parallel(n_jobs=cores)(delayed(ir_sim)(i, s, n, k, p, b0t, ltidx, cidx, bidx, bigj, sigma, beta, t_start) for i in tqdm(range(nsims)))

    for i in tqdm(range(nsims)):
        ir1[:, :, i] = res[i][0]
        ir2[:, :, i] = res[i][1]
        ir3[:, :, i] = res[i][2]

    stop = timeit.default_timer()
    print('Impulse Response simulation completed after', stop - start)

    return ir1, ir2, ir3


