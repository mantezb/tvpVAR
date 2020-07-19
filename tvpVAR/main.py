import numpy as np
import pandas as pd
import timeit
from tqdm import tqdm
from os import path

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
svsims = np.floor(nsims/simstep)

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
