import numpy as np
from scipy.linalg import ldl
import numpy.linalg as lin
import warnings

def udl(A):
    n = a.shape[0]
    r = np.eye(n)
    R = np.fliplr(r)
    L, D, _ = ldl(R @ A @ R)
    U = R @ L @ R
    D = R @ D @ R
    return U, D


def solve_struct(q, psi, case1=True):
    """
    This function recovers the structural parameters from the estimated
    parameters using the procedure described in the appendix.  It used for
    calculating impulse responses.
    :param q:
    :param psi:
    :param casel:
    :return:
    """
    err = False
    q11 = q[:1,:1]
    q12 = q[:1,2]
    q3 = q[3, 3]

    s2_3 = (psi.T @ q11 @ psi -2 * psi.T @ q12 + q3) / ((psi.T @ q12 - q3)**2)
    t = psi.T @ (q12 - q11 @ psi)
    cgamma1 = (1 + np.sqrt(1 - 4 * s2_3 * t)) / (2 * t) * (q12 - q11 @ psi).T
    cgamma2 = (1 - np.sqrt(1 - 4 * s2_3 * t)) / (2 * t) * (q12 - q11 @ psi).T

    if case1:
        cdelta1, cxi1, _ = ldl(lin.inv(q11 - cgamma1.T @ cgamma1 /s2_3))
        cdelta2, cxi2, _ = ldl(lin.inv(q11 - cgamma2.T @ cgamma2 /s2_3))

    else:
        cdelta1, cxi1, _ = ldl(lin.inv(q11 - cgamma1.T @ cgamma1 / s2_3), lower=False)
        cdelta2, cxi2, _ = ldl(lin.inv(q11 - cgamma2.T @ cgamma2 / s2_3), lower=False)

   # else:
    #    cdelta1, cxi1 = udl(lin.solve(q11 - cgamma1.T @ cgamma1 / s2_3))
     #   cdelta2, cxi2 = udl(lin.solve(q11 - cgamma2.T @ cgamma2 / s2_3))

    w1 = np.vstack((lin.inv(cdelta1), np.zeros((2,1)), np.array([0,0,1]))) @ \
         np.vstack((np.hstack((np.eye(2), psi)), np.hstack((cgamma1, 1))))
    s1 = np.vstack((np.hstack((cxi1, np.zeros((2,1)))), np.array([0, 0, s2_3])))
    k1 = lin.lstsq(np.sqrt(s1), w1, rcond=None)[0]
    z1 = k1.T @ k1 - q

    w2 = np.vstack((lin.inv(cdelta2), np.zeros((2, 1)), np.array([0, 0, 1]))) @ \
         np.vstack((np.hstack((np.eye(2), psi)), np.hstack((cgamma2, 1))))
    s2 = np.vstack((np.hstack((cxi2, np.zeros((2, 1)))), np.array([0, 0, s2_3])))
    k2 = lin.lstsq(np.sqrt(s2), w2, rcond=None)[0]
    z2 = k2.T @ k2 - q

    gamma = None
    delta = None
    xi = None

    if np.logical_xor(np.any(np.abs(z1.ravel()) > 1e-9), np.any(np.abs(z2.ravel()) > 1e-9)):
        if np.all(np.abs(z1.ravel()) <= 1e-9):
            gamma = cgamma1
            delta = cdelta1
            xi = cxi1
            k = k1
        else:
            gamma = cgamma2
            delta = cdelta2
            xi = cxi2
            k = k2
    else:
        err = True
        warnings.warn(' No solution or multiple solutions found.')

    # Last check
    if np.any(~np.isreal(k.ravel())):
        err = True
        warnings.warn(' Some structureal parameters are not real.')

    return k, gamma, delta, xi, s2_3, err
