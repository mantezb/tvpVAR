import numpy as np
import pandas as pd
import timeit
from tqdm import tqdm
from os import path
import scipy
from scipy.stats import invwishart
from scipy.linalg import block_diag, ldl
import numpy.linalg as lin
from tvpVAR.utils.mvsvrw import mvsvrw

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
burnin = 0.1 * nsims # burn-in simulations to discard
p = 3  # number of AR lags

# Setting to save every "simstep"^th draw; useful for running long chains
# on windows machines with limited memory
simstep = 5
svsims = int(np.floor(nsims/simstep))

""" Data processing """
data = pd.read_csv(path.join(data_path, filename), header=None)
y_data = data.to_numpy()[:, :3]
t, n = y_data.shape

# Standardisation
dscale = 1 + scale_data * (np.std(y_data, axis=0) - 1)
dcenter = center_data * np.mean(y_data, axis=0)
y_data = (y_data - dcenter) / dscale

y0 = y_data[:p, :].T  # store the first p observations as init cond
y_short = y_data[p:, :].T  # store observations excluding init cond (y0)

t = t - p
y = y_short.flatten().reshape(y_short.flatten().shape[0],1)


x0 = np.empty((n * p, t))
for j in range(p):
    x0[(j * n):((j+1) * n), :] = np.hstack((y0[:, p-j-1:], y_short[:, :(t - j - 1)]))

# Co-integration terms (tax - spend, spend - gdp) - potentially to be re-used to include
# exogenous variables such as oil prices
surp = data.to_numpy()[p:, -2:]
dscale_surp = 1 + 0 * scale_data * (np.std(surp, axis=0)-1)  # experimental, hardcoded 0
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
    x = np.vstack((x, -y_short[:(i+1), :], x0))
    cidx = np.array(np.vstack((cidx, np.ones((i+1, 1)), np.zeros((k, 1)))), dtype=bool)
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
phi_idx = np.array(np.triu(np.ones((nk, nk)) - np.eye(nk)), dtype=bool)

""" Priors """
# For the Inverse Wishart (IW) prior set (used here):
s0h = n + 11  # v0 in the paper, df par for transition covariance R~IW(s0h,S0h); check what 11 is for
S0h = 0.01**2 * (s0h - n - 1) * np.eye(n)  # R0 in the paper, scale parameter for transition covariance R~IW(s0h,S0h)

# For the Inverse Gamma (IG) priors, the following corresponds to the marginal prior on
# Sig_h(i,i) under the Inverse Wishart (used in the diagonal transition cov
# version):
s0hIG = (s0h - n + 1) / 2  # shape parameter for transition variance r_i, r_i ~ IG(s0hIG,S0hIG)
S0hIG = np.diag(S0h) / 2  # scale parameter for transition variance r_i, r_i ~ IG(s0hIG,S0hIG)

# Standard priors for other distributions
a0 = np.zeros((nk, 1))
A0 = np.eye(nk)
phi0 = np.zeros((int(nk * (nk - 1) / 2), 1))  # not found in the paper - should correspond to phi's
Phi0 = np.eye(len(phi0)) * 10  # not found in the paper - should correspond to phi's
L01 = 0.1
L02 = 0.1
lam20 = [L01, L02]  # The same for all Lasso's
mu0 = np.zeros((nk, 1))  # no ref in paper as to which mu this corresponds to (omega)?
M0 = np.ones((nk, 1)) # no ref in paper found

# Initialisations for MCMC sampler
H_small = np.diag(-np.ones((t - 1)), -1) + np.eye(t)
H = np.diag(-np.ones((t-1) * nk), -nk) + np.eye(t * nk)
HH = H.T @ H
K = block_diag(A0, HH)  # no ref in paper found
G0 = np.ones((nk, 1))  # no ref in paper found


# Starting values
_, Sig, _ = ldl(np.cov(y_data, rowvar=False))  # Block LDL' factorization for Hermitian indefinite matrices, return D
h = np.tile(np.log(np.diag(Sig).reshape(np.diag(Sig).shape[0],1)), (t, 1))
Sigh = S0h / (s0h + n + 1)  # Start with the prior mode for transition covariance, R~IW(s0h,S0h), mode = S0h/(s0h+p+1)
bigSig = np.diag(np.exp(-h.ravel()))  # why -h?
tau2 = np.random.exponential(1, (nk, 1))  # why 1 is used?
mu = np.zeros((nk, 1))  # no ref in paper as to which mu this corresponds to (omega)?
om_st = np.sqrt(tau2) * np.random.randn(nk, 1)  # Latent variable Omega star, following N(0,tau2)
om = om_st.copy()
om[[om_st <= 0]] = 0  # Truncating om_st values to be > 0
alpha = np.random.randn(nk, 1)
gamma = np. random.randn(nk, t)
Phi = np.eye(nk)  # no ref in paper found
scl2_1 = np.ones((nk, 1))  # no ref in paper found
scl2_2 = 1  # no ref in paper found
loc = np.zeros((nk, 1))  # no ref in paper found

# Allocate space for draws
s_alpha = np.zeros((nk, svsims))
s_gamma = [np.zeros((nk, t)) for i in range(svsims)]
s_beta = [np.zeros((nk, t)) for i in range(svsims)]
s_Phi = [np.zeros((nk, nk)) for i in range(svsims)]
s_Sig = [np.zeros((n, t)) for i in range(svsims)]
s_Sigh = [np.zeros((n, n)) for i in range(svsims)]
s_lam2 = np.zeros((1, svsims))
s_tau2 = np.zeros((nk, svsims))
s_mu = np.zeros((nk, svsims))
s_om_st = np.zeros((nk, svsims))
s_om = np.zeros((nk, svsims))
s_adj = [np.zeros((nk, 3)) for i in range(svsims)] # hardcoded 3 - double check!


" MCMC Sampler"

# Final initialisations
print('Sampling',burnin+nsims,'draws from the posterior...')

for isim in tqdm(range(int(burnin+nsims))):
    w[widx] = (x * np.tile(om, (1, t))).ravel()
    wPhi = w @ np.kron(np.eye(t), Phi.T)

    # Sample alpha | gamma
    A_hat = A0 + z.T @ bigSig @ z
    a_hat = lin.lstsq(A_hat, A0 @ a0 + z.T @ bigSig @ (y - wPhi @ gamma.ravel()))[0]
    alpha = a_hat + lin.lstsq(lin.cholesky(A_hat), np.random.randn(nk, 1))[0]

    # Sample gamma | alpha
    Gam_hat = HH + wPhi.T @ bigSig @ wPhi
    gam_hat = np.reshape(lin.lstsq(Gam_hat, wPhi.T @ bigSig @ (y - z @ alpha))[0], (nk, t))
    gamma = gam_hat + np.reshape(lin.lstsq(lin.cholesky(Gam_hat), np.random.randn(t * nk, 1)), (nk, t))

    # Do lasso on A0
    if lasso_alpha:
        lam2_a = np.random.gamma(lam20[0] + nk, 1 / (lam20[2] + np.sum(1 / np.diag(A0))/2))
        A0 = np.diag(np.random.wald(np.sqrt(lam2_a) / np.abs(alpha - a0), lam2_a))

    # Sample Sig_t
    err = y - np.hstack(z, wPhi) @ np.vstack(alpha, gamma.ravel())
    h = mvsvrw(np.log(err**2 + 0.001), h, lin.inv(Sigh), np.eye(n))
    bigSig = np.diag(np.exp(-h.ravel()))

    shorth = np.reshape(h, (n, t))
    errh = shorth[:, 1:] - shorth[:, :-1]
    sseh = errh @ errh.T
    Sigh = invwishart.rvs(s0h + t - 1, S0h + sseh)  # Correlated volatilities IW(df, scale)



