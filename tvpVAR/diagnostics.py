import numpy as np
import matplotlib.pyplot as plt
from tvpVAR.utils.ineff_factor import ineff_factor

""" User Settings """
output = 'resultsMCMC_diag_lasso_alpha.npz'
burnin_sims = 0  # extra simulations to be burned for the below calculations based on diagnostics


""" Data Load """

# Load the relevant np.ndarrays from MCMC sampler results file saved in .npz format
data = np.load(output)
s_beta = data['s_beta']
cidx = data['cidx']
bidx = data['bidx']
nk = data['nk'].item()
t = data['t'].item()
svsims = data['svsims'].item()
s_om_st = data['s_om_st']

mbeta0 = np.mean(s_beta, axis=2).T
mbeta = np.zeros(mbeta0.shape)
mbeta[:, :np.count_nonzero(cidx)] = mbeta0[:, cidx.ravel()]
mbeta[:, np.count_nonzero(cidx):] = mbeta0[:, bidx.ravel()]

# Convergence of betas
beta = np.mean(s_beta, axis=1)
for i in np.arange(0, s_beta.shape[0]):
    plt.plot(beta[i, :])
    plt.xlabel('No of simulation')
    plt.ylabel('Beta')
    plt.title('Simulations of Beta')
plt.show()

for i in np.arange(0, s_om_st.shape[0]):
    plt.plot(s_om_st[i, :])
    plt.xlabel('No of simulation')
    plt.ylabel('Omega Star')
    plt.title('Simulations of Omega Star')
plt.show()

# Set which data to be used

s_om_st = s_om_st[:, burnin_sims:]
s_beta = s_beta[:, :, burnin_sims:]

# MCM mixing analysis and beta plots
ef_om_st, _ = ineff_factor(s_om_st)
ef_beta, _ = ineff_factor(np.reshape(s_beta, (nk * t, svsims-burnin_sims), order='F'))

# Plotting
flierprops = dict(marker='+', markerfacecolor='teal', markersize=7, linestyle='none', markeredgecolor='teal')
medianprops = dict(color='lightseagreen')
plt.boxplot([ef_om_st[:, 0], ef_beta[:, 0]], labels=['Omega Star', 'Beta'], flierprops=flierprops,
            medianprops=medianprops, showmeans=True, meanline=True)
plt.xlabel('Estimated Parameters')
plt.ylabel('Inefficiency Factors')
plt.title(' Boxplot of Inefficiency Factors')
plt.show()
# A boxplot of Omega_star and Bet inefficiency factors
plt.plot(np.max(mbeta, axis=0) - np.min(mbeta, axis=0), marker='x', c='teal', linestyle='dashed', linewidth=1)
plt.ylabel('Max Beta - Min Beta')
plt.xlabel('Parameter Index')
plt.title('Maximum vs Minimum Average Beta Parameters')
plt.show()

