import numpy as np
import matplotlib.pyplot as plt
from tvpVAR.utils.ineff_factor import ineff_factor
#import tvpVAR.utils.settings as settings


#settings.init()

# Load the relevant np.ndarrays from MCMC sampler results file saved in .npz format
data = np.load('resultsMCMC_v2.npz')
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

# MCM mixing analysis and beta plots
ef_om_st, _ = ineff_factor(s_om_st)
ef_beta, _ = ineff_factor(np.reshape(s_beta, (nk * t, svsims), order='F'))

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

