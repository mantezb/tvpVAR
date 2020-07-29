import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
from tvpVAR.utils.ir_vecm_sv import ir_vecm_sv
from tvpVAR.utils.ir_var_sv import ir_var_sv

""" User Settings """
output = 'resultsMCMC_diag_lasso_alpha.npz'
model = 'ir_vecm_sv' #'ir_var_sv' for lower triangular identification as per Primiceri (2005), monetary policy shocks
                     # 'ir_vecm_sv' for identification schene as per Blanchard (2002), fiscal policy shocks
models = ['ir_vecm_sv', 'ir_var_sv']
variables = ['inflation', 'unemployment', 'interest rate']
start_date = 1959.50  # start date of the data
end_date = 2011.75  # end date of the data
step = 0.25  # i.e. 0.25 for quarterly data
ir_dates = [2000, 2000, 2000.00]
ir_dates_names = ['1975:Q1', '1981: Q3', '1996: Q1']
burnin_sims = 0  # extra burnin to be used
ir_output = 'ir_diag_lasso_alpha.npz'
quantiles = [0.16, 0.5, 0.84]

""" Data Load """
# Load the relevant np.ndarrays from MCMC sampler results file saved in .npz format
data = np.load(output)
s_beta = data['s_beta'][:, :, burnin_sims:]
s_Sig = data['s_Sig'][:, :, burnin_sims:]
cidx = data['cidx']
bidx = data['bidx']
s_om = data['s_om'][:, burnin_sims:]
svsims = data['svsims'].item()
dscale = data['dscale'].reshape((-1, 1))
if model=='ir_vecm_sv':
    dscale_surp = data['dscale_surp'].reshape((-1, 1))
p = data['p'].item()
scale_adj = dscale * (1 / dscale.T)

# Dates set up
d_1 = date(int(ir_dates[0]), 3 + int(12*(ir_dates[0]-int(ir_dates[0]))), 1)
d_2 = date(int(ir_dates[1]), 3 + int(12*(ir_dates[1]-int(ir_dates[1]))), 1)
d_3 = date(int(ir_dates[2]), 3 + int(12*(ir_dates[2]-int(ir_dates[2]))), 1)
date_range_1 = [d_1 + relativedelta(months=int(3*i)) for i in range(21)]
date_range_2 = [d_2 + relativedelta(months=int(3*i)) for i in range(21)]
date_range_3 = [d_3 + relativedelta(months=int(3*i)) for i in range(21)]
date_sig = date(int(start_date), 3 + int(12*(start_date-int(start_date))), 1)
date_range_sig = [date_sig + relativedelta(months=int(3*i)) for i in range(s_Sig.shape[1])]

""" Time invariant/constant parameter probabilities"""
# Plot the time invariant/constant param probabilities
p_tiv0 = np.sum(s_om == 0, axis=1)/svsims
p_tiv = np.zeros(p_tiv0.shape)
p_tiv[:np.count_nonzero(cidx)] = p_tiv0[cidx.ravel()]
p_tiv[np.count_nonzero(cidx):] = p_tiv0[bidx.ravel()]
# Plotting
plt.plot(p_tiv, marker='x', c='teal', linestyle='dashed', linewidth=1)
plt.ylabel('TIV probability')
plt.xlabel('Parameter Index')
plt.title('Time Invariance Probability by Parameter')
plt.show()

""" Standard Deviations of Residuals"""
sig = np.quantile(s_Sig, quantiles, axis=2)

for i in range(3):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date_range_sig, (np.sqrt(sig[1, i, :])).T, c='teal', label='Mean std')
    ax.fill_between(date_range_sig, np.sqrt(sig[0, i, :]).T, np.sqrt(sig[2, i, :]).T,
                    color='cadetblue', alpha=0.1, label='HPD interval')
    ax.grid(alpha=0.1)
    ax.set_title(f'Posterior mean of the standard deviation of residuals in {variables[i]} equation')
    ax.set_xlabel('Time')
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.show()

""" Generation of Impulse Responses """
# Impulse responses
per = np.arange(start_date, end_date+step, step)
ir_dates = np.array(ir_dates)
t_start = np.where(np.in1d(per, ir_dates))[0]

s = 20

# Structural identification and impulse response simulation:

if model == 'ir_vecm_sv':
    ab2, c, svars, ir, err = ir_vecm_sv(s_beta, s_Sig, cidx, t_start[0], s, 2.08 * dscale[2] / dscale[0], p,
                                    np.diag((1 / dscale_surp).ravel()) @ np.array([[1, -1, 0], [0, 1, -1]]) @ np.diag(dscale.ravel()))
    nerr = np.count_nonzero(err)
elif model == 'ir_var_sv':
    ir1, ir2, ir3 = ir_var_sv(s_beta, s_Sig, cidx, t_start, s, p)
    np.savez(ir_output, ir1=ir1, ir2=ir2, ir3=ir3)
else:
    print('Identification for model type', model ,'is not available. Please choose from available models:', models)


""" Impulse Response Plots """

if model=='ir_var_sv':

    ir1XY = np.quantile(ir1, quantiles, axis=2)
    ir2XY = np.quantile(ir2, quantiles, axis=2)
    ir3XY = np.quantile(ir3, quantiles, axis=2)
    ir_list = [ir1XY, ir2XY, ir3XY]
    date_list = [date_range_1, date_range_2, date_range_3]

    for i in range(3):

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(date_list[i], (ir_list[i][1, 0, :]*scale_adj[0, 2]).T, c='teal', label='Mean IR')
        ax.fill_between(date_list[i], (ir_list[i][0, 0, :] * scale_adj[0, 2]).T, (ir_list[i][2, 0, :] * scale_adj[0, 2]).T, color='cadetblue', alpha=0.1, label='HPD interval')
        ax.grid(alpha=0.1)
        ax.set_title(f'Impulse response of {variables[i]}, {ir_dates_names[i]}')
        ax.set_xlabel('Time')
        ax.legend(loc='upper left')
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(date_list[i], (ir_list[i][1, 1, :] * scale_adj[1, 2]).T, c='teal', label='Mean IR')
        ax.fill_between(date_list[i], (ir_list[i][0, 1, :] * scale_adj[1, 2]).T, (ir_list[i][2, 1, :] * scale_adj[1, 2]).T,
                        color='cadetblue', alpha=0.1, label='HPD interval')
        ax.grid(alpha=0.1)
        ax.set_title(f'Impulse response of {variables[i]}, {ir_dates_names[i]}')
        ax.set_xlabel('Time')
        ax.legend(loc='upper left')
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(date_list[i], (ir_list[i][1, 2, :] * scale_adj[2, 2]).T, c='teal', label='Mean IR')
        ax.fill_between(date_list[i], (ir_list[i][0, 2, :] * scale_adj[2, 2]).T, (ir_list[i][2, 2, :] * scale_adj[2, 2]).T,
                        color='cadetblue', alpha=0.1, label='HPD interval')
        ax.grid(alpha=0.1)
        ax.set_title(f'Impulse response of {variables[i]}, {ir_dates_names[i]}')
        ax.set_xlabel('Time')
        ax.legend(loc='upper left')
        fig.tight_layout()
        plt.show()


if model == 'ir_vecm_sv':
    bands = (np.floor(np.array([0.16, 0.5, 0.85]) * (svsims - nerr))).astype(int)
    ir_sort = np.sort(ir[:, :, :, ~err.astype('bool').ravel()], 3)
    # make ir plots
    conf1 = (ir_sort[0, 1, :, bands[[0, 2]]] * scale_adj[0, 1]).T
    plt.plot(date_range_1, conf1, c='cadetblue', linewidth=2,
         linestyle='dashed')
    plt.fill_between(date_range_1, conf1[:, 0], conf1[:, 1], color='cadetblue', alpha=0.2)
    plt.plot(date_range_1, (ir_sort[0, 1, :, bands[[1]]] * scale_adj[0, 1]).T, c='teal', linewidth=3)
    plt.title('Impulse Response of Variable x to y')
    plt.ylabel('Impulse Response')
    plt.xlabel('Time (in quarters)')
    plt.show()

    conf2 = (ir_sort[1, 1, :, bands[[0, 2]]] * scale_adj[1, 1]).T
    plt.plot(date_range_1, conf2, c='cadetblue', linewidth=2,
         linestyle='dashed')
    plt.fill_between(date_range_1, conf2[:, 0], conf2[:, 1], color='cadetblue', alpha=0.2)
    plt.plot(date_range_1, (ir_sort[1, 1, :, bands[[1]]] * scale_adj[1, 1]).T, c='teal', linewidth=3)
    plt.title('Impulse Response of variable y to y')
    plt.ylabel('Impulse Response')
    plt.xlabel('Time (in quarters)')
    plt.show()

    conf3 = (ir_sort[2, 1, :, bands[[0, 2]]] * scale_adj[2, 1]).T
    plt.plot(date_range_1, conf3, c='cadetblue', linewidth=2,
             linestyle='dashed')
    plt.fill_between(date_range_1, conf3[:, 0], conf3[:, 1], color='cadetblue', alpha=0.2)
    plt.plot(date_range_1, (ir_sort[2, 1, :, bands[[1]]] * scale_adj[2, 1]).T, c='teal', linewidth=3)
    plt.title('Impulse Response of variable z to y')
    plt.ylabel('Impulse Response')
    plt.xlabel('Time (in quarters)')
    plt.show()

    plt.plot(date_range_1, (ir_sort[0, 1, :, bands[[1]]] * scale_adj[0, 1]).T, c='cadetblue', linewidth=2,
             linestyle='dashed')
    plt.plot(date_range_1,(ir_sort[1, 1, :, bands[[1]]] * scale_adj[1, 1]).T, c='teal', linewidth=3)
    plt.title('Impulse Response of variable y to y, different confidence bands')
    plt.ylabel('Impulse Response')
    plt.xlabel('Time (in quarters)')
    plt.show()


