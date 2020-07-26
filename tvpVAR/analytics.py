import numpy as np
import matplotlib.pyplot as plt
from tvpVAR.utils.ir_vecm_sv import ir_vecm_sv
from tvpVAR.utils.ir_plots import ir_plots
#import tvpVAR.utils.settings as settings TESTING

#settings.init() TESTING

# Load the relevant np.ndarrays from MCMC sampler results file saved in .npz format
data = np.load('resultsMCMC_v2.npz')
s_beta = data['s_beta']
s_Sig = data['s_Sig']
cidx = data['cidx']
bidx = data['bidx']
s_om = data['s_om']
svsims = data['svsims'].item()
dscale = data['dscale'].reshape((-1, 1))
dscale_surp = data['dscale_surp'].reshape((-1, 1))
p = data['p'].item()

#s_beta = settings.beta_s # TESTING
#s_Sig = settings.sig_s # TESTING

# Plot the time invariant/constant param probabilities
p_tiv0 = np.sum(s_om == 0, axis=1)/ svsims
p_tiv = np.zeros(p_tiv0.shape)
p_tiv[:np.count_nonzero(cidx)] = p_tiv0[cidx.ravel()]
p_tiv[np.count_nonzero(cidx):] = p_tiv0[bidx.ravel()]
# Plotting
plt.plot(p_tiv, marker='x', c='teal', linestyle='dashed', linewidth=1)
plt.ylabel('TIV probability')
plt.xlabel('Parameter Index')
plt.title('Time Invariance Probability by Parameter')
plt.show()

# Impulse responses
scale_adj = dscale * (1 / dscale.T)
per = np.arange(1959.5, 2011.75, 0.25)
t_start = np.argwhere(per == 2000).ravel()
s = 20
ab2, c, svars, ir, err = ir_vecm_sv(s_beta, s_Sig, cidx, t_start, s, 2.08 * dscale[2] / dscale[0], p,
                                    np.diag((1 / dscale_surp).ravel()) @ np.array([[1, -1, 0], [0, 1, -1]]) @ np.diag(dscale.ravel()))
nerr = np.count_nonzero(err)
bands = (np.floor(np.array([0.16, 0.5, 0.85]) * (svsims - nerr))).astype(int)
ir_sort = np.sort(ir[:, :, :, ~err.astype('bool').ravel()], 3)

""" Generate Impulse Response Plots """
# Plot using a function
#ir_plots(ir_sort, bands, scale_adj, s)
#plt.show()

# make ir plots
conf1 = (ir_sort[0, 1, :, bands[[0, 2]]] * scale_adj[0, 1]).T
plt.plot(conf1, c='cadetblue', linewidth=2,
         linestyle='dashed')
plt.fill_between(np.arange(s+1), conf1[:, 0], conf1[:, 1], color='cadetblue', alpha=0.2)
plt.plot((ir_sort[0, 1, :, bands[[1]]] * scale_adj[0, 1]).T, c='teal', linewidth=3)
plt.title('Impulse Response of Variable x to y')
plt.ylabel('Impulse Response')
plt.xlabel('Time (in quarters)')
plt.show()

conf2 = (ir_sort[1, 1, :, bands[[0, 2]]] * scale_adj[1, 1]).T
plt.plot(conf2, c='cadetblue', linewidth=2,
         linestyle='dashed')
plt.fill_between(np.arange(s+1), conf2[:, 0], conf2[:, 1], color='cadetblue', alpha=0.2)
plt.plot((ir_sort[1, 1, :, bands[[1]]] * scale_adj[1, 1]).T, c='teal', linewidth=3)
plt.title('Impulse Response of variable y to y')
plt.ylabel('Impulse Response')
plt.xlabel('Time (in quarters)')
plt.show()

conf3 = (ir_sort[2, 1, :, bands[[0, 2]]] * scale_adj[2, 1]).T
plt.plot(conf3, c='cadetblue', linewidth=2,
         linestyle='dashed')
plt.fill_between(np.arange(s+1), conf3[:, 0], conf3[:, 1], color='cadetblue', alpha=0.2)
plt.plot((ir_sort[2, 1, :, bands[[1]]] * scale_adj[2, 1]).T, c='teal', linewidth=3)
plt.title('Impulse Response of variable z to y')
plt.ylabel('Impulse Response')
plt.xlabel('Time (in quarters)')
plt.show()

plt.plot((ir_sort[0, 1, :, bands[[1]]] * scale_adj[0, 1]).T, c='cadetblue', linewidth=2,
         linestyle='dashed')
plt.plot((ir_sort[1, 1, :, bands[[1]]] * scale_adj[1, 1]).T, c='teal', linewidth=3)
plt.title('Impulse Response of variable y to y, different confidence bands')
plt.ylabel('Impulse Response')
plt.xlabel('Time (in quarters)')
plt.show()
