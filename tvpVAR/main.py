import timeit
from os import path

import numpy as np
import numpy.linalg as lin
import pandas as pd
from scipy.linalg import ldl
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.stats import invwishart
from tqdm import tqdm
from sksparse.cholmod import cholesky as chol
from tvpVAR.utils.hpr_sampler import hpr_sampler
from tvpVAR.utils.mvsvrw import mvsvrw
from tvpVAR.utils.utils import repmat
import tvpVAR.utils.settings as settings  # TESTING
import scipy

settings.init()

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
nsims = 25000  # desired number of MCMC simulations
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
y = sps.csc_matrix(y_short.T.ravel().reshape((y_short.ravel().shape[0], 1)))

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
z = repmat(sps.block_diag(z0, format='csc'), 1, t)
w = sps.kron(sps.eye(t, format='csc'), sps.block_diag(z0, format='csc').T, format='csc')
widx = w.copy().astype('bool')
g = z.copy()
gidx = g.copy() != 0  # note: this is actually G' in the paper
z[z != 0] = x.flatten()
z = z.T.tocsc()
f = []
for j in range(1, nk):
    f = sps.block_diag((f, np.ones((j, 1))), format='csc')
f = f[1:, :]
f = repmat(sps.hstack((sps.csc_matrix(np.zeros((f.shape[0], 1))), f)), 1, t)  # This will hold the gammas for sampling Phi

# Useful indices
fidx = f.copy() != 0
uidx = sps.triu(np.ones((nk, nk)), format='csc')
uidx[:, -1] = 0
biguidx = repmat(uidx, 1, t).astype('bool')
Phi_idx = sps.triu(np.ones((nk, nk))- sps.eye(nk, format='csc'), format='csc').astype('bool')

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
a0 = sps.csc_matrix(np.zeros((nk, 1)))
A0 = sps.eye(nk, format='csc')
phi0 = sps.csc_matrix(np.zeros((int(nk * (nk - 1) / 2), 1)))  # not found in the paper - should correspond to phi's
Phi0 = sps.eye(phi0.shape[0]).dot(10)  # not found in the paper - should correspond to phi's
L01 = 0.1
L02 = 0.1
lam20 = np.array([L01, L02])  # The same for all Lasso's
mu0 = np.zeros((nk, 1))  # no ref in paper as to which mu this corresponds to (omega)?
M0 = sps.csc_matrix(np.ones((nk, 1)))  # no ref in paper found

# Initialisations for MCMC sampler
H_small = sps.spdiags(-np.ones((t - 1)), -1, t, t, format='csc') + sps.eye(t, format='csc')
H = sps.spdiags(-np.ones((t - 1) * nk), -nk, t*nk, t*nk, format='csc') + sps.eye(t*nk, format='csc')
HH = H.T @ H
K = sps.block_diag((A0, HH), 'csc')  # no ref in paper found
g0 = sps.csc_matrix(np.ones((nk, 1)))  # no ref in paper found

# Starting values
_, Sig, _ = ldl(np.cov(y_data, rowvar=False))  # Block LDL' factorization for Hermitian indefinite matrices, return D
h = sps.csc_matrix(repmat(np.log(np.diag(Sig).reshape(np.diag(Sig).shape[0], 1)), t, 1))
Sigh = S0h / (s0h + n + 1)  # Start with the prior mode for transition covariance, R~IW(s0h,S0h), mode = S0h/(s0h+p+1)
bigSig = sps.spdiags(np.exp(-h.reshape((-1, 1)).data), 0, t * n, t * n, format='csc')  # why -h?
tau2 = np.random.exponential(1, (nk, 1))  # why 1 is used?
mu = np.zeros((nk, 1))  # no ref in paper as to which mu this corresponds to (omega)?
om_st = np.sqrt(tau2) * np.random.randn(nk, 1)  # Latent variable Omega star, following N(0,tau2)
om = om_st.copy()
om[[om_st <= 0]] = 0  # Truncating om_st values to be > 0

alpha = sps.csc_matrix(np.random.randn(nk, 1))  # constant coefficients
gamma = sps.csc_matrix(np.random.randn(nk, t))  # time varying coefficients
Phi = sps.eye(nk, format='csc')  # standard normal CDF values used for pi = Phi(-mu/tau)
scl2_1 = np.ones((nk, 1))  # no ref in paper found
scl2_2 = 1  # no ref in paper found
loc = sps.csc_matrix(np.zeros((nk, 1)))  # no ref in paper found
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

rand_om_st = settings.rand_om_st.copy()
rand_n_39 = settings.rand_n[0:39].copy()
rand_n_211 = settings.rand_n[0:211].copy()
rand_n = settings.rand_n.copy()
rand_tau2 = settings.rand_tau2.copy()
rand_sigh = settings.rand_sigh.copy()
rand_om = rand_om_st.copy()
rand_om[[rand_om_st<=0]] = 0


" MCMC Sampler"

# Final initialisations
print('Sampling', int(burnin + nsims), 'draws from the posterior...')
start = timeit.default_timer()

for isim in tqdm(range(int(burnin + nsims))):
    w[widx] = (x * np.tile(om, (1, t))).T.ravel()
    w[widx] = (x * np.tile(rand_om, (1, t))).T.ravel() # TESTING
    wPhi = sps.csc_matrix(w) @ sps.kron(sps.eye(t, format='csc'), Phi.T, format='csc')

    # Sample alpha | gamma
    A_hat = sps.csc_matrix(A0 + z.T @ bigSig @ z)
    gamma = rand_n.reshape((nk,t), order='F')
    a_hat = sps.csc_matrix(spsolve(A_hat, A0 @ a0 + z.T @ bigSig @ (y - wPhi @ gamma.T.reshape((-1, 1)))))
    alpha = (a_hat + sps.csc_matrix(spsolve(chol(A_hat, ordering_method='natural').L().T, np.random.randn(nk, 1)))).T
    alpha = (a_hat + sps.csc_matrix(spsolve(chol(A_hat, ordering_method='natural').L().T, rand_n_39))).T # TESTING

    # Sample gamma | alpha
    Gam_hat = HH + wPhi.T @ bigSig @ wPhi
    gam_hat = sps.csc_matrix(np.reshape(spsolve(Gam_hat, wPhi.T @ bigSig @ (y - z @ alpha)), (nk, t), order='F'))
    gamma = gam_hat + sps.csc_matrix(np.reshape(spsolve(chol(Gam_hat, ordering_method='natural').L().T, np.random.randn(t * nk, 1)), (nk, t),
                                 order='F'))
    gamma = gam_hat + sps.csc_matrix(np.reshape(spsolve(chol(Gam_hat, ordering_method='natural').L().T, rand_n), (nk, t),
                                                order='F')) #TESTING

    # Do lasso on A0
    if lasso_alpha:
        lam2_a = np.random.gamma(lam20[0] + nk, 1 / (lam20[1] + np.sum(1 / np.diag(A0.toarray())) / 2))
        lam2_a = 2.17 # TESTING ONLY
        A0 = sps.spdiags(np.random.wald(np.sqrt(lam2_a) / np.abs(alpha - a0).toarray(), lam2_a).ravel(), 0, nk, nk, format='csc')

    # Sample Sig_t
    err = y - sps.hstack((z, wPhi), format = 'csc') @ sps.vstack((alpha, gamma.T.reshape((-1, 1))))  # error - sparse
    h, _ = mvsvrw(np.log(err.power(2).data + 0.001).reshape((-1, 1)), h, sps.csc_matrix(lin.inv(Sigh)), sps.eye(n, format='csc'))
    bigSig = sps.spdiags(np.exp(-h.reshape((-1, 1)).data), 0, t * n, t * n, format='csc')
    shorth = np.reshape(h, (n, t), order='F').tocsc()
    errh = shorth[:, 1:] - shorth[:, :-1]
    sseh = errh @ errh.T
    Sigh = invwishart.rvs(s0h + t - 1, S0h + sseh)  # Correlated volatilities IW(df, scale)
    Sigh = rand_sigh # TESTING
    # Sample om, om_st, tau2, lam2
    g[gidx] = (sps.csc_matrix(x).multiply((Phi.T @ gamma))).reshape((1, -1))

    om, om_st, tau2, lam2 = hpr_sampler(lam20, (y - z @ alpha), g.T, bigSig, om, tau2, mu) # DISABLE FOR TESTING
    #om, om_st, tau2, lam2 = hpr_sampler(lam20, (y - z @ alpha), g.T, bigSig, rand_om, rand_tau2, mu) # TESTING
    if rand_mu:
        # Sample mu
        M_hat = 1 / (M0.toarray() + 1 / tau2)
        mu_hat = M_hat * (M0.multiply(mu0).toarray() + om_st / tau2)
        mu = mu_hat + np.sqrt(M_hat) * np.random.randn(nk, 1)
        lam2_mu = np.random.gamma(lam20[0] + nk, 1 / (lam20[1] + np.sum(1 / M0.toarray()) / 2))
        M0 = sps.csc_matrix(np.random.wald(np.sqrt(lam2_mu) / np.abs(mu - mu0), lam2_mu))

    # Sample Phi
    # The conditional model is: y = z @ alpha + w @ gamma + w @ f.t @ phi + e
    bigGam = np.reshape(repmat(gamma, nk, 1), (nk, nk * t), order='F').tocsc()
    f = f.T
    f[fidx.T] = bigGam.T[biguidx.T]
    err = y - sps.hstack((z, w)) @ sps.vstack((alpha, gamma.T.reshape((-1, 1))))
    wf = w @ f
    f = f.T
    Phi_hat = sps.csc_matrix(Phi0 + wf.T @ bigSig @ wf)
    phi_hat = spsolve(Phi_hat, Phi0 @ phi0 + wf.T @ bigSig @ err)
    Phi = Phi.T
    Phi[Phi_idx.T] = phi_hat + spsolve(chol(Phi_hat,ordering_method='natural').L().T, np.random.randn(len(phi_hat), 1))
    Phi[Phi_idx.T] = phi_hat + spsolve(chol(Phi_hat, ordering_method='natural').L().T, rand_n[0:741]) # TESTING
    Phi = Phi.T

    if lasso_Phi:
        # Do lasso on Phi0
        lam2_f = np.random.gamma(lam20[0] + phi0.shape[0], 1 / (lam20[1] + np.sum(1 / np.diag(Phi0.toarray())) / 2))
        Phi0 = sps.spdiags(np.random.wald(np.sqrt(lam2_f) / np.abs(np.array(Phi.T[Phi_idx.T]).ravel() - phi0.toarray().ravel()),
                                          lam2_f), 0, phi0.shape[0], phi0.shape[0], format='csc')


    if do_expansion:
        sseg = np.reshape(np.array(np.sum((H_small @ gamma.T).power(2), axis=0)).ravel(), (-1, 1))
        g0 = np.random.gamma((1 + t) / 2, 2 / (n * t + sseg))

        long_g0 = repmat(np.sqrt(g0), t, 1)
        Hg0 = sps.spdiags(long_g0.ravel(), 0, t * nk, t * nk, format='csc') + sps.spdiags(-long_g0[:-nk, :].ravel(), -nk, t * nk, t * nk, format='csc')
        HH = Hg0.T @ Hg0

    if do_ishift[0]:
        # Apply one distn - invariant scale transformation to lam2 and tau2 (e.g. Liu and Sabatti, 2000)
        scale =  1 / (lam20[1] * lam2 + np.sum((om_st - mu) ** 2 / tau2) / 2)
        if np.isnan(scale):
            scl2_2 = 1
        else:
            scl2_2 = np.random.gamma(lam20[0] + nk / 2,
                                 1 / (lam20[1] * lam2 + np.sum((om_st - mu) ** 2 / tau2) / 2))
        tau2 = tau2 / scl2_2

        if rand_mu:
            # one more time for lam2_mu, M0
            scale = 1 / (lam20[1] * lam2_mu + np.sum((mu - mu0) ** 2 * M0.toarray()) / 2)
            if np.isnan(scale):
                scl2_2m = 1
            else:
                scl2_2m = np.random.gamma(lam20[0] + nk / 2, 1 / (lam20[1] * lam2_mu + np.sum((mu - mu0) ** 2 * M0.toarray()) / 2))

            M0 = M0.dot(scl2_2m)

        if lasso_alpha:
            # one more for lam2_a, A0
            scale = 1 / (lam20[1] * lam2_a + np.sum((alpha - a0).power(2).toarray().ravel() * np.diag(A0.toarray())) / 2)
            if np.isnan(scale):
                scl2_2a = 1
            else:
                scl2_2a = np.random.gamma(lam20[0] + nk / 2,
                                      1 / (lam20[1] * lam2_a + np.sum((alpha - a0).power(2).toarray().ravel() * np.diag(A0.toarray())) / 2))

            A0 = A0.dot(scl2_2a)

        if lasso_Phi:
            # one more for lam2_f, Phi0
            scale = 1 / (lam20[1] * lam2_f + np.sum((np.array(Phi.T[Phi_idx.T]).ravel() - phi0.toarray().ravel()) ** 2 * np.diag(Phi0.toarray())) / 2)
            if np.isnan(scale):
                scl2_2f = 1
            else:
                scl2_2f = np.random.gamma(lam20[0] + phi0.shape[0] / 2,
                        1 / (lam20[1] * lam2_f + np.sum((np.array(Phi.T[Phi_idx.T]).ravel() - phi0.toarray().ravel()) ** 2 * np.diag(Phi0.toarray())) / 2))

            Phi0 = Phi0.dot(scl2_2f)

    if do_ishift[1]:
        # apply j distn-invariant location transformations to alpha and gamma (e.g. Liu and Sabatti, 2000)
        Om = sps.spdiags(om.ravel(), 0, om.shape[0], om.shape[0], format='csc') @ Phi.T
        Om = sps.spdiags(rand_om.ravel(), 0, om.shape[0], om.shape[0], format='csc') @ Phi.T # TESTING
        Loc_hat = sps.eye(nk, format='csc') + Om.T @ A0 @ Om
        loc_hat = spsolve(Loc_hat, np.reshape(gamma[:, 0], (-1, 1)) - Om.T @ A0 @ (alpha - a0))
        loc = np.reshape(loc_hat + spsolve(chol(Loc_hat, ordering_method='natural').L().T, np.random.randn(nk, 1)), (-1, 1))
        loc = np.reshape(loc_hat + spsolve(chol(Loc_hat, ordering_method='natural').L().T, rand_n_39),
                         (-1, 1)) # TESTING
        #alpha = alpha + sps.csc_matrix(om * loc) # DISABLE FOR TESTING
        alpha = alpha + sps.csc_matrix(rand_om * loc) # TESTING
        gamma = gamma - sps.csc_matrix(repmat(loc, 1, t))

    # Save draws
    if True: # TESTING
    #if (isim + 1 > burnin) and (np.mod(isim + 1 - burnin, simstep) == 0): # DISABLE FOR TESTING

        isave = int((isim + 1 - burnin) / simstep - 1)
        isave = 0 # TESTING
        s_alpha[:, isave] = alpha.toarray().ravel()
        s_gamma[:, :, isave] = (sps.spdiags(np.sqrt(g0.toarray()).ravel(), 0, g0.shape[0], g0.shape[0], format='csc') @ gamma).toarray()
        s_beta[:, :, isave] = (repmat(alpha, 1, t) + sps.spdiags(om.ravel(), 0, om.shape[0], om.shape[0], format='csc') @ Phi.T @ gamma).toarray()
        s_Phi[:, :, isave] = (sps.spdiags(np.sqrt(g0.toarray()).ravel(), 0, g0.shape[0], g0.shape[0], format='csc') @ Phi.T @
                              sps.spdiags(np.sqrt(1 / g0.toarray()).ravel(), 0, g0.shape[0], g0.shape[0], format='csc')).toarray()
        s_Sig[:, :, isave] = np.exp(shorth.data).reshape((n, t), order='F') # Sig
        s_Sigh[:, :, isave] = Sigh
        s_lam2[:, isave] = lam2
        s_tau2[:, isave] = tau2.ravel()
        s_mu[:, isave] = mu.ravel()
        s_om_st[:, isave] = (om_st * np.sqrt(1 / g0.toarray())).ravel()
        s_om[:, isave] = (om * np.sqrt(1 / g0.toarray())).ravel()
        s_adj[:, :, isave] = np.hstack((np.sqrt(scl2_1), np.vstack((np.sqrt(scl2_2), np.zeros((nk - 1, 1)))), loc))

stop = timeit.default_timer()
print('Sampling completed after', stop - start)

""" Saving results """

np.savez('resultsMCMC_v2.npz', s_alpha=s_alpha, s_gamma=s_gamma, s_beta=s_beta, s_Phi=s_Phi, s_Sig=s_Sig,
         s_Sigh=s_Sigh, s_lam2=s_lam2, s_tau2=s_tau2, s_mu=s_mu, s_om_st=s_om_st, s_om=s_om, s_adj=s_adj, svsims=svsims,
         cidx=cidx, bidx=bidx, dscale=dscale, dscale_surp=dscale_surp, x=x, nk=nk, t=t, p=p)

# Use  diagnostics.py and analytics.py