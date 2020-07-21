import numpy as np
from scipy.stats import norm


def tnr_icdf(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:

    p1 = norm.cdf(a)
    p2 = norm.cdf(b)
    z = norm.ppf(p1 + np.random.rand(n, 1) * (p2 - p1))

    return z


def tnr_ar1(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:

    alpha = 0.5 * (a + np.sqrt(a**2 + 4))
    z = np.tile(np.nan, (n, 1))
    idx = np.array(list(range(n)))

    while len(idx) > 0:
        z_new = a[idx] + np.random.exponential(1 / alpha[idx])
        r = np.exp(-((z_new - alpha[idx])**2) / 2)
        u = np.random.rand(len(idx), 1)
        z[idx[(u <= r) & (z_new <= b[idx])]] = z_new[(u <= r) & (z_new <= b[idx])]
        idx = idx[(u > r) | (z_new > b[idx])]

    return z


def tnr_ar2(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:

    z = np.tile(np.nan, (n, 1))
    idx = np.array(list(range(n)))

    while len(idx) > 0:
        z_new = np.random.uniform(a[idx], b[idx])
        c = a[idx] * (a[idx] > 0) + b[idx] * (b[idx] < 0)
        r = np.exp((c**2-z_new**2) / 2)
        u = np.random.rand(len(idx), 1)
        z[idx[u <= r]] = z_new[u <= r]
        idx = idx[u > r]

    return z


def tnr(mu: np.ndarray, sig: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """
    Univariate truncated normal sampling
    based on Robert (1995).  Uses the inverse
    CDF method if the closest bound is within
    certain number of standard deviations away
    from the mean; otherwise, the Robert (1995)
    rejection sampling techniques.
    All input vectors must be same-length column vectors.

    :param mu: n x 1 vector representing location parameters
    :param sig: n x 1 vector representing scale parameters
    :param lb: n x 1 vector representing lower bounds
    :param ub: n x 1 vector representing upper bounds
    :return y: n draws of independent truncated normal variables,
                each corresponding to the respective parameter inputs.
    """
    utol = 7
    ltol = -7

    n = len(mu)
    a = (lb - mu) / sig
    b = (ub - mu) / sig

    idx_icdf = np.argwhere((a < utol) & (b > ltol))
    idx_ar = np.argwhere((a <= utol) | (b >= ltol))
    idx_arl = np.argwhere(a[idx_ar] > 0)
    idx_arr = np.argwhere(a[idx_ar] < 0)

    dbd_l = 2 * (np.sqrt(np.exp(1))) / ((a + np.sqrt(a**2 + 4)) * np.exp((a**2 - a * np.sqrt(a**2 + 4)) / 4))
    dbd_r = 2 * (np.sqrt(np.exp(1))) / ((-b + np.sqrt(b**2 + 4)) * np.exp((b**2 + b * np.sqrt(b**2 + 4)) / 4))

    idx_ar1l = np.argwhere(b[idx_ar[idx_arl]] - a[idx_ar[idx_arl]] > dbd_l[idx_ar[idx_arl]])
    idx_ar2l = np.argwhere(b[idx_ar[idx_arl]] - a[idx_ar[idx_arl]] <= dbd_l[idx_ar[idx_arl]])
    idx_ar1r = np.argwhere(b[idx_ar[idx_arr]] - a[idx_ar[idx_arr]] > dbd_r[idx_ar[idx_arr]])
    idx_ar2r = np.argwhere(b[idx_ar[idx_arr]] - a[idx_ar[idx_arr]] <= dbd_r[idx_ar[idx_arr]])
    idx_ar2 = np.vstack((idx_arl[idx_ar2l], idx_arr[idx_ar2r]))

    x = np.tile(np.nan, (n, 1))

    if len(idx_icdf) > 0:
        # inverse cdf method
        x[idx_icdf] = tnr_icdf(a[idx_icdf, b[idx_icdf], len(idx_icdf)])

    if len(idx_ar1l) > 0:
        # left truncated
        x[idx_ar[idx_arl[idx_ar1l]]] = tnr_ar1(a[idx_ar[idx_arl[idx_ar1l]]], b[idx_ar[idx_arl[idx_ar1l]]],
                                               len(idx_ar1l))

    if len(idx_ar1r) > 0:
        # right truncated
        x[idx_ar[idx_arr[idx_ar1r]]] = -tnr_ar1(-b[idx_ar[idx_arr[idx_ar1r]]], -a[idx_ar[idx_arr[idx_ar1r]]],
                                               len(idx_ar1r))

    if len(idx_ar2) > 0:
        # two-sided
        x[idx_ar[idx_ar[idx_ar2]]] = tnr_ar2(a[idx_ar[idx_ar2]], b[idx_ar[idx_ar2]], len(idx_ar2))

    y = mu + sig * x

    return y
