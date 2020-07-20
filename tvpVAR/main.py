import numpy as np
import pandas as pd
import timeit

from numpy.core._multiarray_umath import ndarray
from tqdm import tqdm
from os import path
import scipy
from scipy.linalg import block_diag, ldl

# Specification of directories
base_path = path.dirname(__file__)  # Location of the main.py
data_path = path.abspath(path.join(base_path,'data'))  # The path where the data is stored

""" User Settings """
# Data specification
filename = 'BP2002data_v19.csv'

# Standardisation controls
scale_data = 1  # standardise series to have std. dev. of 1
center_data = 1 # standardise series to have mean 0

# Algorithm specific controls
rand_mu = 0  # sample mu randomly (experimental)
agjoint = 0  # sample alpha, gamma jointly (experimental)
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
y = y_short.flatten()


x0 = np.empty((n * p, t))
for j in range(p):
    x0[(j * n):((j+1) * n), :] = np.hstack((y0[:, p-j-1:], y_short[:, :(t - j - 1)]))

# Co-integration terms (tax - spend, spend - gdp) - potentially to be re-used to include
# exogenous variables such as oil prices
surp = data.to_numpy()[p:, -2:]
dscale_surp = 1 + 0 * scale_data * (np.std(surp, axis=0)-1)  # experimental, hardcoded 0
dcenter_surp = 0 * center_data * np.mean(surp, axis=0) # experimental, hardcoded 0
surp = (surp - dcenter_surp) / dscale_surp

x_data = np.vstack((np.ones((1,t)),surp.T))
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
    x = np.vstack((x,-y_short[:(i+1), :], x0))
    cidx = np.vstack((cidx, np.ones((i+1, 1)), np.zeros((k, 1))))
bidx = 1 - cidx

# Construct prototype f, g, z, q used for priors and posterior computation
z0 = [None] * n
for i in range(n):
    z0[i] = np.ones((k + i, 1))
z = np.tile(block_diag(*z0), (1,t))
w = np.kron(np.eye(t), block_diag(*z0).T)
widx = w.copy()
g = z.copy()
gidx = g.copy() != 0  # note: this is actually G' in the paper
z[z != 0] = x.flatten()
z = z.T
f = []
for j in range(1, nk):
    f = block_diag(f, np.ones((j, 1)))
f = f[1:,:]
f = np.tile(np.hstack((np.zeros((f.shape[0], 1)), f)), (1, t))  # This will hold the gammas for sampling Phi

# Useful indices
fidx = f.copy() != 0
uidx = np.triu(np.ones((nk, nk)))
uidx[:, -1] = 0
biguidx = np.tile(uidx, (1, t))
phi_idx = np.triu(np.ones((nk, nk)) - np.eye(nk))

""" Priors """
# For the Inverse Wishart (IW) prior set (used here):
s0h = n + 11  # v0 in the paper, check what 11 is for
S0h = 0.01**2 * (s0h - n - 1) * np.eye(n)  # RR0 in the paper

# For the Inverse Gamma (IG) priors, the following corresponds to the marginal prior on
# Sig_h(i,i) under the Inverse Wishart (used in the diagonal transition cov
# version):
s0hIG = (s0h - n + 1) / 2
S0hIG = np.diag(S0h) / 2

# Standard priors for other distributions
a0 = np.zeros((nk, 1))
A0 = np.eye(nk)
phi0 = np.zeros((int(nk * (nk - 1) / 2), 1))
Phi0 = np.eye(len(phi0)) * 10
Hsmall = np.diag(-np.ones((t - 1)), -1) + np.eye(t)
H = np.diag(-np.ones((t-1) * nk), -nk) + np.eye(t * nk)
HH = H.T @ H
K = block_diag(A0, HH)
G0 = np.ones((nk, 1))
mu0 = np.zeros((nk, 1))
M0 = np.ones((nk, 1))
L01 = 0.1
L02 = 0.1
lam20 = [L01, L02]  # The same for all Lasso's

# Starting values
_, Sig, _ = ldl(np.cov(y_data, rowvar=False))  #Block LDL' factorization for Hermitian indefinite matrices, return D
h = np.tile(np.log(np.diag(Sig).reshape(np.diag(Sig).shape[0],1)), (t, 1))
Sigh = S0h / (s0h + n + 1)  # Start with the prior mode
bigSig = np.diag(np.exp(-h.ravel()))
tau2 = np.random.exponential(1, (nk, 1))
mu = np.zeros((nk, 1))
om_st = np.sqrt(tau2) * np.random.randn(nk, 1)
om = om_st.copy()
om[[om_st <= 0]] = 0  # Truncating om_st values to be > 0
alpha = np.random.randn(nk, 1)
gamma = np. random.randn(nk, t)
Phi = np.eye(nk)
scl2_1 = np.ones((nk, 1))
scl2_2 = 1
loc = np.zeros((nk, 1))

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

# Final initialisations
print('Sampling',burnin+nsims,'draws from the posterior...')
