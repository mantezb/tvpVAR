import numpy as np
import matplotlib.pyplot as plt
from tvpVAR.utils.ineff_factor import ineff_factor

""" User Settings """
output = 'resultsMCMC_AWM_diag_4vars_conv_2lags_25k_1970_v2.npz'
burnin_sims = 0  # extra simulations to be burned for the below calculations based on diagnostics
vars = 5
save_plots = True  # save plots as pdf
show_plots = True  # show plots
ineff_file = "ineff_full_5vars_conv_2lags_25k_1970_v4.npz"

""" Data Load """

# Load the relevant np.ndarrays from MCMC sampler results file saved in .npz format
data = np.load(output)
data_ineff =np.load(ineff_file)
s_beta = data['s_beta']
cidx = data['cidx']
bidx = data['bidx']
nk = data['nk'].item()
t = data['t'].item()
svsims = data['svsims'].item()
s_om_st = data['s_om_st']
p = data['p'].item()
s_om = data['s_om']

# Create a list with the names of the parameters
parameters = []
for i in range(int(np.sum(cidx.ravel()))):
    parameters.append(f'b{i+1}')

for n in range(vars):
    parameters.append(f'c{n + 1}')
    for m in range(p):
        for j in range(vars):
            parameters.append(f'y{n+1}{j+1}{m+1}') #y equation, variable, lag


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
if save_plots:
    plt.savefig('Beta_sims.pdf', format='pdf')
if show_plots:
    plt.show()

for i in np.arange(0, s_om_st.shape[0]):
    plt.plot(s_om_st[i, :])
    plt.xlabel('No of simulation')
    plt.ylabel('Omega Star')
    plt.title('Simulations of Omega Star')
if save_plots:
    plt.savefig('Om_st_sims.pdf', format='pdf')
if show_plots:
    plt.show()

# Plot posterior mean of parameters
plt.plot(parameters, np.mean(s_om_st, axis=1))
plt.xlabel('Parameter')
plt.ylabel('Omega Star')
plt.title('Posterior Mean of Omega Star')
if save_plots:
    plt.savefig('Om_st_posterior_mean.pdf', format='pdf')
if show_plots:
    plt.show()

for i in np.arange(0, s_beta.shape[0]):
    plt.plot(np.mean(s_beta, axis=2)[i,:])
    plt.xlabel('Time')
    plt.ylabel('Beta')
    plt.title('Posterior Mean of Beta')
if save_plots:
    plt.savefig('Beta_posterior_mean.pdf', format='pdf')
if show_plots:
    plt.show()

# MCM mixing analysis and beta plots
ef_om_st = data_ineff['ef_om_st']
ef_beta = data_ineff['ef_beta']

# Plotting
flierprops = dict(marker='+', markerfacecolor='teal', markersize=7, linestyle='none', markeredgecolor='teal')
medianprops = dict(color='lightseagreen')
plt.boxplot([ef_om_st[:, 0], ef_beta[:, 0]], labels=['Omega Star', 'Beta'], flierprops=flierprops,
            medianprops=medianprops, showmeans=True, meanline=True)
plt.xlabel('Estimated Parameters')
plt.ylabel('Inefficiency Factors')
plt.title(' Boxplot of Inefficiency Factors')
if save_plots:
    plt.savefig('Ineff_factors.pdf', format='pdf')
if show_plots:
    plt.show()
# A boxplot of Omega_star and Bet inefficiency factors
plt.plot(np.max(mbeta, axis=0) - np.min(mbeta, axis=0), marker='x', c='teal', linestyle='dashed', linewidth=1)
plt.ylabel('Max Beta - Min Beta')
plt.xlabel('Parameter Index')
plt.title('Maximum vs Minimum Average Beta Parameters')
if save_plots:
    plt.savefig('MTV_beta.pdf', format='pdf')
if show_plots:
    plt.show()

""" Time invariant/constant parameter probabilities"""
# Plot the time invariant/constant param probabilities
p_tiv0 = np.sum(s_om == 0, axis=1)/svsims
p_tiv = np.zeros(p_tiv0.shape)
p_tiv[:np.count_nonzero(cidx)] = p_tiv0[cidx.ravel()]
p_tiv[np.count_nonzero(cidx):] = p_tiv0[bidx.ravel()]
# Plotting
plt.plot(parameters, p_tiv, marker='x', c='teal', linestyle='dashed', linewidth=1)
plt.ylabel('TIV probability')
plt.xlabel('Parameter Index')
plt.title('Time Invariance Probability by Parameter')
if save_plots:
    plt.savefig('TIV_SMSS.pdf', format='pdf')
if show_plots:
    plt.show()