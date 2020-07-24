import timeit
from os import path

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import pandas as pd
from scipy.linalg import block_diag, ldl, cholesky
import scipy.sparse as sps
from scipy.sparse.linalg import lsqr, spsolve
from scipy.stats import invwishart
from tqdm import tqdm
from sksparse.cholmod import cholesky as chol
from tvpVAR.utils.hpr_sampler import hpr_sampler
from tvpVAR.utils.ineff_factor import ineff_factor
from tvpVAR.utils.ir_plots import ir_plots
from tvpVAR.utils.ir_vecm_sv import ir_vecm_sv
from tvpVAR.utils.mvsvrw import mvsvrw
# import tvpVAR.utils.settings as settings  # TESTING

# Specification of directories
base_path = path.dirname(__file__)  # Location of the main.py
data_path = path.abspath(path.join(base_path, 'data'))  # The path where the data is stored

""" User Settings """
# Data specification
filename = 'BP2002data_v19.csv'

# Standardisation controls
scale_data = 1  # standardise series to have std. dev. of 1
center_data = 1  # standardise series to have mean 0

# Algorithm specific controls
rand_mu = 0  # sample mu randomly (experimental)
lasso_alpha = 0  # use a lasso prior on the variance of alpha
lasso_Phi = 1  # use a lasso prior on the state covariances
do_expansion = 0  # use parameter expansion on om, Phi, gamma
do_ishift = [1, 1]  # do a distn-invariant translation / scaling of draws
nsims = 1000  # desired number of MCMC simulations
burnin = 0.1 * nsims  # burn-in simulations to discard
p = 3  # number of AR lags

# Setting to save every "simstep"^th draw; useful for running long chains
# on windows machines with limited memory
simstep = 5
svsims = int(np.floor(nsims / simstep))

""" Data processing """
data = pd.read_csv(path.join(data_path, filename), header=None)
y_data = data.to_numpy()[:, :3]
t, n = y_data.shape
do_ishift = np.array(do_ishift)

# Standardisation
dscale = 1 + scale_data * (np.std(y_data, axis=0) - 1)
dcenter = center_data * np.mean(y_data, axis=0)
y_data = (y_data - dcenter) / dscale

y0 = y_data[:p, :].T  # store the first p observations as init cond
y_short = y_data[p:, :].T  # store observations excluding init cond (y0)

t = t - p
y = y_short.T.ravel().reshape((y_short.ravel().shape[0], 1))

x0 = np.empty((n * p, t))
for j in range(p):
    x0[(j * n):((j + 1) * n), :] = np.hstack((y0[:, p - j - 1:], y_short[:, :(t - j - 1)]))

# Co-integration terms (tax - spend, spend - gdp) - potentially to be re-used to include
# exogenous variables such as oil prices
surp = data.to_numpy()[p:, -2:]
dscale_surp = 1 + 0 * scale_data * (np.std(surp, axis=0) - 1)  # experimental, hardcoded 0
dcenter_surp = 0 * center_data * np.mean(surp, axis=0)  # experimental, hardcoded 0
surp = (surp - dcenter_surp) / dscale_surp

x_data = np.vstack((np.ones((1, t)), surp.T))
x0 = np.vstack((x_data, x0))

""" Data constructs for posterior computation """
# Define useful constants
k = n * p + len(x_data)  # Number of coefficients in each equation (endogenous + exogenous + constant)

# Size of parameters vector, where n * k corresponds to coefficients
# across all VAR equations and the first item corresponds to covariance elements of
# error covariance matrix
nk = int(0.5 * n * (n - 1) + n * k)  # size of parameters vector

# Construct x - confirm what is x! - and construct indices cidx and bidx to recover covariance coefficients
x = x0
cidx = np.zeros((k, 1))
for i in range(2):
    x = np.vstack((x, -y_short[:(i + 1), :], x0))
    cidx = np.array(np.vstack((cidx, np.ones((i + 1, 1)), np.zeros((k, 1)))), dtype=bool)
bidx = np.array(1 - cidx, dtype=bool)

# Construct prototype f, g, z, q used for priors and posterior computation
z0 = [None] * n
for i in range(n):
    z0[i] = np.ones((k + i, 1))
z = np.tile(block_diag(*z0), (1, t))
w = np.kron(np.eye(t), block_diag(*z0).T)
widx = np.array(w.copy(), dtype=bool)
g = z.copy()
gidx = g.copy() != 0  # note: this is actually G' in the paper
z[z != 0] = x.flatten()
z = z.T
f = []
for j in range(1, nk):
    f = block_diag(f, np.ones((j, 1)))
f = f[1:, :]
f = np.tile(np.hstack((np.zeros((f.shape[0], 1)), f)), (1, t))  # This will hold the gammas for sampling Phi

# Useful indices
fidx = f.copy() != 0
uidx = np.triu(np.ones((nk, nk)))
uidx[:, -1] = 0
biguidx = np.array(np.tile(uidx, (1, t)), dtype=bool)
Phi_idx = np.array(np.triu(np.ones((nk, nk)) - np.eye(nk)), dtype=bool)

""" Priors """
# For the Inverse Wishart (IW) prior set (used here):
s0h = n + 11  # v0 in the paper, df par for transition covariance R~IW(s0h,S0h); check what 11 is for
S0h = 0.01 ** 2 * (s0h - n - 1) * np.eye(n)  # R0 in the paper, scale parameter for transition covariance R~IW(s0h,S0h)

# For the Inverse Gamma (IG) priors, the following corresponds to the marginal prior on
# Sig_h(i,i) under the Inverse Wishart (used in the diagonal transition cov
# version):
s0hIG = (s0h - n + 1) / 2  # shape parameter for transition variance r_i, r_i ~ IG(s0hIG,S0hIG)
S0hIG = np.diag(S0h) / 2  # scale parameter for transition variance r_i, r_i ~ IG(s0hIG,S0hIG)

# Standard priors for other distributions
a0 = np.zeros((nk, 1))
A0 = sps.eye(nk)
phi0 = np.zeros((int(nk * (nk - 1) / 2), 1))  # not found in the paper - should correspond to phi's
Phi0 = np.eye(len(phi0)) * 10  # not found in the paper - should correspond to phi's
L01 = 0.1
L02 = 0.1
lam20 = np.array([L01, L02])  # The same for all Lasso's
mu0 = np.zeros((nk, 1))  # no ref in paper as to which mu this corresponds to (omega)?
M0 = np.ones((nk, 1))  # no ref in paper found

# Initialisations for MCMC sampler
H_small = np.diag(-np.ones((t - 1)), -1) + np.eye(t)
H = sps.spdiags(-np.ones((t - 1) * nk), -nk,t*nk,t*nk) + sps.eye(t*nk)
HH = H.T @ H
K = sps.block_diag((A0, HH), 'csc')  # no ref in paper found
g0 = np.ones((nk, 1))  # no ref in paper found

# Starting values
_, Sig, _ = ldl(np.cov(y_data, rowvar=False))  # Block LDL' factorization for Hermitian indefinite matrices, return D
h = np.tile(np.log(np.diag(Sig).reshape(np.diag(Sig).shape[0], 1)), (t, 1))
Sigh = S0h / (s0h + n + 1)  # Start with the prior mode for transition covariance, R~IW(s0h,S0h), mode = S0h/(s0h+p+1)
bigSig = np.diag(np.exp(-h.ravel()))  # why -h?
tau2 = np.random.exponential(1, (nk, 1))  # why 1 is used?
mu = np.zeros((nk, 1))  # no ref in paper as to which mu this corresponds to (omega)?
om_st = np.sqrt(tau2) * np.random.randn(nk, 1)  # Latent variable Omega star, following N(0,tau2)
om = om_st.copy()
om[[om_st <= 0]] = 0  # Truncating om_st values to be > 0

alpha = np.random.randn(nk, 1)  # constant coefficients
gamma = np.random.randn(nk, t)  # time varying coefficients
Phi = sps.eye(nk,format='csc')  # standard normal CDF values used for pi = Phi(-mu/tau)
scl2_1 = np.ones((nk, 1))  # no ref in paper found
scl2_2 = 1  # no ref in paper found
loc = np.zeros((nk, 1))  # no ref in paper found
lam2_mu = None
lam2_a = None
lam2_f = None

# Allocate space for draws
s_alpha = np.zeros((nk, svsims))
s_gamma = np.zeros((nk, t, svsims))
s_beta = np.zeros((nk, t, svsims))
s_Phi = np.zeros((nk, nk, svsims))
s_Sig = np.zeros((n, t, svsims))
s_Sigh = np.zeros((n, n, svsims))
s_lam2 = np.zeros((1, svsims))
s_tau2 = np.zeros((nk, svsims))
s_mu = np.zeros((nk, svsims))
s_om_st = np.zeros((nk, svsims))
s_om = np.zeros((nk, svsims))
s_adj = np.zeros((nk, 3, svsims))  # hardcoded 3 - double check!

" MCMC Sampler"

# Final initialisations
print('Sampling', int(burnin + nsims), 'draws from the posterior...')
start = timeit.default_timer()

for isim in tqdm(range(int(burnin + nsims))):
    w[widx] = (x * np.tile(om, (1, t))).T.ravel()
    wPhi = sps.csc_matrix(w) @ sps.kron(sps.eye(t,format='csc'), Phi.T, format='csc')

    # Sample alpha | gamma
    A_hat = sps.csc_matrix(A0 + z.T @ bigSig @ z)
    a_hat = spsolve(A_hat, A0 @ a0 + z.T @ bigSig @ (y - wPhi @ np.reshape(gamma.T.ravel(), (-1, 1))))
    alpha = np.reshape(a_hat + spsolve(chol(A_hat.T).L(), np.random.randn(nk, 1)), (-1, 1))

    # Sample gamma | alpha
    Gam_hat = sps.csc_matrix(HH + wPhi.T @ bigSig @ wPhi)
    gam_hat = np.reshape(spsolve(Gam_hat, wPhi.T @ bigSig @ (y - z @ alpha)), (nk, t), order='F')
    gamma = gam_hat + np.reshape(spsolve(chol(Gam_hat.T).L(), np.random.randn(t * nk, 1)), (nk, t),
                                 order='F')

    # Do lasso on A0
    if lasso_alpha:
        lam2_a = np.random.gamma(lam20[0] + nk, 1 / (lam20[1] + np.sum(1 / np.diag(A0)) / 2))
        A0 = np.diag(np.random.wald(np.sqrt(lam2_a) / np.abs(alpha - a0), lam2_a).ravel())

    # Sample Sig_t
    err = y - np.hstack((z, wPhi)) @ np.vstack((alpha, gamma.T.ravel().reshape(len(gamma.ravel()), 1))) # error - sparse
    h, _ = mvsvrw(np.log(err ** 2 + 0.001), h, lin.inv(Sigh), np.eye(n))
    bigSig = np.diag(np.exp(-h.T.ravel()))
    shorth = np.reshape(h, (n, t), order='F')
    errh = shorth[:, 1:] - shorth[:, :-1]
    sseh = errh @ errh.T
    Sigh = invwishart.rvs(s0h + t - 1, S0h + sseh)  # Correlated volatilities IW(df, scale)
    # Sample om, om_st, tau2, lam2
    g[gidx] = (x * (Phi.T @ gamma)).ravel()

    om, om_st, tau2, lam2 = hpr_sampler(lam20, y - z @ alpha, g.T, bigSig, om, tau2, mu)

    if rand_mu:
        # Sample mu
        M_hat = 1 / (M0 + 1 / tau2)
        mu_hat = M_hat * (M0 * mu0 + om_st / tau2)
        mu = mu_hat + np.sqrt(M_hat) * np.random.randn(nk, 1)
        lam2_mu = np.random.gamma(lam20[0] + nk, 1 / (lam20[1] + np.sum(1 / M0) / 2))
        M0 = np.random.wald(np.sqrt(lam2_mu) / np.abs(mu - mu0), lam2_mu)

    # Sample Phi
    # The conditional model is: y = z @ alpha + w @ gamma + w @ f.t @ phi + e
    bigGam = np.reshape(np.tile(gamma, (nk, 1)), (nk, nk * t), order='F')
    f = f.T
    f[fidx.T] = bigGam.T[biguidx.T].T
    err = y - np.hstack((z, w)) @ np.vstack((alpha, gamma.T.ravel().reshape((len(gamma.ravel()), 1))))
    wf = w @ f
    f = f.T

    Phi_hat = sps.csc_matrix(Phi0 + wf.T @ bigSig @ wf)
    phi_hat = spsolve(Phi_hat, Phi0 @ phi0 + wf.T @ bigSig @ err)
    Phi = Phi.T
    Phi[Phi_idx.T] = phi_hat + spsolve(chol(Phi_hat.T).L(), np.random.randn(len(phi_hat), 1))
    Phi = Phi.T

    if lasso_Phi:
        # Do lasso on Phi0
        lam2_f = np.random.gamma(lam20[0] + len(phi0), 1 / (lam20[1] + np.sum(1 / np.diag(Phi0)) / 2))
        Phi = Phi.T
        Phi0 = np.diag(np.random.wald(np.sqrt(lam2_f) / np.abs(Phi[Phi_idx.T] - phi0.ravel()), lam2_f))
        Phi = Phi.T

    if do_expansion:
        sseg = np.reshape(np.sum((H_small @ gamma.T) ** 2, axis=0), (-1, 1))
        g0 = np.random.gamma((1 + t) / 2, 2 / (n * t + sseg))

        long_g0 = np.tile(np.sqrt(g0), (t, 1))
        Hg0 = np.diag(long_g0.ravel(), 0) + np.diag(-long_g0[:-nk, :].ravel(), -nk)
        HH = Hg0.T @ Hg0  # very slow, consider sparsifying

    if do_ishift[0]:
        # Apply one distn - invariant scale transformation to lam2 and tau2 (e.g. Liu and Sabatti, 2000)
        scl2_2 = np.random.gamma(lam20[0] + nk / 2,
                                 1 / (lam20[1] * lam2 + np.sum((om_st - mu) ** 2 / tau2) / 2))
        tau2 = tau2 / scl2_2

        if rand_mu:
            # one more time for lam2_mu, M0
            scl2_2m = np.random.gamma(lam20[0] + nk / 2,
                                      1 / (lam20[1] * lam2_mu + np.sum((mu - mu0) ** 2 * M0) / 2))
            M0 = M0 * scl2_2m

        if lasso_alpha:
            # one more for lam2_a, A0
            scl2_2a = np.random.gamma(lam20[0] + nk / 2,
                                      1 / (lam20[1] * lam2_a + np.sum((alpha - a0) ** 2 * np.diag(A0)) / 2))
            A0 = A0 * scl2_2a

        if lasso_Phi:
            # one more for lam2_f, Phi0
            Phi = Phi.T
            scl2_2f = np.random.gamma(lam20[0] + len(phi0) / 2,
                        1 / (lam20[1] * lam2_f + np.sum((Phi[Phi_idx.T] - phi0.ravel()) ** 2 * np.diag(Phi0)) / 2))
            Phi = Phi.T
            Phi0 = Phi0 * scl2_2f

    if do_ishift[1]:
        # apply j distn-invariant location transformations to alpha and gamma (e.g. Liu and Sabatti, 2000)
        Om = np.diag(om.ravel()) @ Phi.T
        Loc_hat = sps.csc_matrix(np.eye(nk) + Om.T @ A0 @ Om)
        loc_hat = spsolve(Loc_hat, np.reshape(gamma[:, 0], (-1, 1)) - Om.T @ A0 @ (alpha - a0))
        loc = np.reshape(loc_hat + spsolve(chol(Loc_hat.T).L(), np.random.randn(nk, 1)), (-1, 1))
        alpha = alpha + om * loc
        gamma = gamma - np.tile(loc, (1, t))

    # Save draws
    if (isim + 1 > burnin) and (np.mod(isim + 1 - burnin, simstep) == 0):

        isave = int((isim + 1 - burnin) / simstep - 1)
        s_alpha[:, isave] = alpha.ravel()
        s_gamma[:, :, isave] = np.diag(np.sqrt(g0).ravel()) @ gamma
        s_beta[:, :, isave] = np.tile(alpha, (1, t)) + np.diag(om.ravel()) @ Phi.T @ gamma
        s_Phi[:, :, isave] = np.diag(np.sqrt(g0).ravel()) @ Phi.T @ np.diag(np.sqrt(1 / g0).ravel())
        s_Sig[:, :, isave] = np.exp(shorth)  # Sig
        s_Sigh[:, :, isave] = Sigh
        s_lam2[:, isave] = lam2
        s_tau2[:, isave] = tau2.ravel()
        s_mu[:, isave] = mu.ravel()
        s_om_st[:, isave] = (om_st * np.sqrt(1 / g0)).ravel()
        s_om[:, isave] = (om * np.sqrt(1 / g0)).ravel()
        s_adj[:, :, isave] = np.hstack((np.sqrt(scl2_1), np.vstack((np.sqrt(scl2_2), np.zeros((nk - 1, 1)))), loc))

stop = timeit.default_timer()
print('Sampling completed after', stop - start)

""" Diagnostics """

mbeta0 = np.mean(s_beta, axis=0).T
mbeta = np.zeros(mbeta0.shape)
mbeta[:, :np.count_nonzero(cidx)] = mbeta0[:, cidx]
mbeta[:, np.count_nonzero(cidx):] = mbeta0[:, bidx]

# MCM mixing analysis and beta plots
ef_om_st = ineff_factor(s_om_st)
ef_beta = ineff_factor(np.reshape(s_beta, (nk * t, svsims), order='F'))
eff_factors_df = pd.DataFrame({'Omega_star': ef_om_st, 'Beta': ef_beta})
# Plotting
eff_factors_df.boxplot()  # A boxplot of Omega_star and Bet inefficiency factors
plt.plot(np.max(mbeta, axis=0) - np.min(mbeta, axis=0), marker='o', c='teal', linestyle='dashed')
plt.ylabel('Max vs min beta')
plt.show()

# Plot the time invariant/constant param probabilities
p_tiv0 = np.sum(s_om == 0, axis=1).T.reshape((1, len(s_om == 0))) / svsims
p_tiv = np.zeros(p_tiv0.shape)
p_tiv[:np.count_nonzero(cidx)] = p_tiv0[cidx]
p_tiv[np.count_nonzero(cidx):] = p_tiv0[bidx]
# Plotting
plt.plot(p_tiv, marker='o', c='teal', linestyle='dashed')
plt.ylabel('TIV probability')
plt.show()

# Impulse responses
scale_adj = dscale.T * (1 / dscale)
per = np.arange(1959.5, 2011.75, 0.25)
t_start = np.argwhere(per == 2000)
s = 20
ab2, c, svars, ir, err = ir_vecm_sv(s_beta, s_Sig, cidx, t_start, s, 2.08 * dscale[2] / dscale[0], p,
                                    np.diag((1 / dscale_surp).ravel()) \
                                    * np.array([[1, -1, 0], [0, 1, -1]]) * np.diag(dscale.ravel()))
nerr = np.count_nonzero(err)
bands = np.floor(np.array([0.16, 0.5, 0.85]) * (svsims - nerr))
ir_sort = np.sort(ir[:, :, :, ~err], 3)

# make ir plots
""" Generate Impulse Response Plots """

plt.plot(np.squeeze(np.arange(0, s), ir_sort[0, 1, :, bands[[0, 3]]]) * scale_adj[0, 1], c='cadetblue', linewidth=2,
         linestyle='dashed')
plt.plot(np.squeeze(np.arange(0, s), ir_sort[0, 1, :, bands[[2]]]) * scale_adj[0, 1], c='teal', linewidth=3)
plt.title('Impulse Response of variable x to y')
plt.ylabel('Impulse Response')
plt.xlabel('Time (in quarters)')
plt.show()

plt.plot(np.squeeze(np.arange(0, s), ir_sort[1, 1, :, bands[[0, 3]]]) * scale_adj[0, 1], c='cadetblue', linewidth=2,
         linestyle='dashed')
plt.plot(np.squeeze(np.arange(0, s), ir_sort[1, 1, :, bands[[2]]]) * scale_adj[0, 1], c='teal', linewidth=3)
plt.title('Impulse Response of variable y to y')
plt.ylabel('Impulse Response')
plt.xlabel('Time (in quarters)')
plt.show()

plt.plot(np.squeeze(np.arange(0, s), ir_sort[2, 1, :, bands[[0, 3]]]) * scale_adj[0, 1], c='cadetblue', linewidth=2,
         linestyle='dashed')
plt.plot(np.squeeze(np.arange(0, s), ir_sort[2, 1, :, bands[[2]]]) * scale_adj[0, 1], c='teal', linewidth=3)
plt.title('Impulse Response of variable z to y')
plt.ylabel('Impulse Response')
plt.xlabel('Time (in quarters)')
plt.show()

plt.plot(np.squeeze(np.arange(0, s), ir_sort[0, 1, :, bands[[0, 3]]]) * scale_adj[0, 1], c='cadetblue', linewidth=2,
         linestyle='dashed')
plt.plot(np.squeeze(np.arange(0, s), ir_sort[1, 1, :, bands[[2]]]) * scale_adj[0, 1], c='teal', linewidth=3)
plt.title('Impulse Response of variable y to y, different confidence bands')
plt.ylabel('Impulse Response')
plt.xlabel('Time (in quarters)')
plt.show()

# Plot using a function
ir_plots(ir_sort, bands, scale_adj, s)
