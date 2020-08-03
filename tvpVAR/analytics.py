import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta
from tvpVAR.utils.ir_vecm_sv import ir_vecm_sv
from tvpVAR.utils.ir_var_sv import ir_var_sv

""" User Settings """
output = 'resultsMCMC_AWM_full_4vars_conv_2lags_25k_1970_v2.npz'
model = 'ir_var_sv' #'ir_var_sv' for lower triangular identification as per Primiceri (2005), monetary policy shocks
                     # 'ir_vecm_sv' for identification schene as per Blanchard (2002), fiscal policy shocks
models = ['ir_vecm_sv', 'ir_var_sv']
variables = ['Real output', 'Prices', 'Interest rate', 'Exchange Rate']
#variables = ['Commodity prices', 'Real output', 'Prices', 'Interest rate']
policy_equation = ['Interest rate']
start_date = 1970.25  # start date of the data
end_date = 2017.75  # end date of the data
step = 0.25  # i.e. 0.25 for quarterly data
ir_dates = [1980.00, 1990.00, 1995.00, 2000.00, 2005.00, 2010.00]  #.00 corresponds to Q1, .25 Q2, 0.5 Q3 and 0.74 Q4
burnin_sims = 0  # extra burnin to be used
ir_output = 'ir_AWM_diag_2lag_conv_4var_25k_1970_v2.npz'
quantiles = [0.16, 0.5, 0.84]
horizon_sim = 12
horizon_plot = 12
write_ir = True # write impulse response results in "ir output" file

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
save_plots = True
show_plots = False
# Dates set up
date_dict = {0: 'Q1', 0.25: 'Q2', 0.5: 'Q3', 0.75: 'Q4'}
ir_dates_names = [f'{int(ir_dates[0])}-{date_dict[np.mod(ir_dates[0],1)]}', f'{int(ir_dates[1])}-{date_dict[np.mod(ir_dates[1],1)]}',
                  f'{int(ir_dates[2])}-{date_dict[np.mod(ir_dates[2],1)]}', f'{int(ir_dates[3])}-{date_dict[np.mod(ir_dates[3],1)]}',
                  f'{int(ir_dates[4])}-{date_dict[np.mod(ir_dates[4],1)]}', f'{int(ir_dates[5])}-{date_dict[np.mod(ir_dates[5],1)]}']
d_1 = date(int(ir_dates[0]), 3 + int(12*(ir_dates[0]-int(ir_dates[0]))), 1)
d_2 = date(int(ir_dates[1]), 3 + int(12*(ir_dates[1]-int(ir_dates[1]))), 1)
d_3 = date(int(ir_dates[2]), 3 + int(12*(ir_dates[2]-int(ir_dates[2]))), 1)
d_4 = date(int(ir_dates[3]), 3 + int(12*(ir_dates[3]-int(ir_dates[3]))), 1)
d_5 = date(int(ir_dates[4]), 3 + int(12*(ir_dates[4]-int(ir_dates[4]))), 1)
d_6 = date(int(ir_dates[5]), 3 + int(12*(ir_dates[5]-int(ir_dates[5]))), 1)
date_range_1 = [d_1 + relativedelta(months=int(3*i)) for i in range(horizon_plot+1)]
date_range_2 = [d_2 + relativedelta(months=int(3*i)) for i in range(horizon_plot+1)]
date_range_3 = [d_3 + relativedelta(months=int(3*i)) for i in range(horizon_plot+1)]
date_range_4 = [d_4 + relativedelta(months=int(3*i)) for i in range(horizon_plot+1)]
date_range_5 = [d_5 + relativedelta(months=int(3*i)) for i in range(horizon_plot+1)]
date_range_6 = [d_6 + relativedelta(months=int(3*i)) for i in range(horizon_plot+1)]

date_sig = date(int(start_date), 3 + int(12*(start_date-int(start_date))), 1)
date_range_sig = [date_sig + relativedelta(months=int(3*i)) for i in range(s_Sig.shape[1])]

# Create a list with the names of the parameters
parameters = []
for i in range(int(np.sum(cidx.ravel()))):
    parameters.append(f'b{i+1}')

for n in range(len(variables)):
    parameters.append(f'c{n + 1}')
    for m in range(p):
        for j in range(len(variables)):
            parameters.append(f'y{n+1}{j+1}{m+1}') #y equation, variable, lag

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
    plt.savefig('TIV_SMSS',format='pdf')
if show_plots:
    plt.show()

""" Standard Deviations of Residuals"""
sig = np.quantile(s_Sig, quantiles, axis=2)

for i in range(len(variables)):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(date_range_sig, (np.sqrt(sig[1, i, :])).T * dscale[i], c='teal', label='Mean std')
    ax.fill_between(date_range_sig, np.sqrt(sig[0, i, :]).T * dscale[i], np.sqrt(sig[2, i, :]).T * dscale[i],
                        color='cadetblue', alpha=0.1, label='HPD interval')
    ax.grid(alpha=0.1)
    ax.set_title(f'Posterior mean of the standard deviation of residuals in {variables[i]} equation')
    ax.set_xlabel('Time')
    ax.legend(loc='upper left')
    fig.tight_layout()
    if save_plots:
        plt.savefig(f'Resid_{variables[i]}_SMSS',format='pdf')
    if show_plots:
        plt.show()


""" Generation of Impulse Responses """

pol_eq = np.where(np.in1d(variables, policy_equation))[0].item()
per = np.arange(start_date, end_date+step, step)
ir_dates = np.array(ir_dates)
t_start = np.where(np.in1d(per, ir_dates))[0]

# Structural identification and impulse response simulation:

if model == 'ir_vecm_sv':
    ab2, c, svars, ir, err = ir_vecm_sv(s_beta, s_Sig, cidx, t_start[0], horizon_sim, 2.08 * dscale[2] / dscale[0], p,
                                    np.diag((1 / dscale_surp).ravel()) @ np.array([[1, -1, 0], [0, 1, -1]]) @ np.diag(dscale.ravel()))
    nerr = np.count_nonzero(err)
elif model == 'ir_var_sv':
    ir1, ir2, ir3, ir4, ir5, ir6 = ir_var_sv(s_beta, s_Sig, cidx, t_start, horizon_sim, p, pol_eq)
    if write_ir:
        np.savez(ir_output, ir1=ir1, ir2=ir2, ir3=ir3, ir4=ir4, ir5=ir5, ir6=ir6)
else:
    print('Identification for model type ', model,' is not available. Please choose from available models:', models)


""" Impulse Response Plots """

if model == 'ir_var_sv':

    ir1XY = np.quantile(ir1, quantiles, axis=2)[:, :, :horizon_plot+1]
    ir2XY = np.quantile(ir2, quantiles, axis=2)[:, :, :horizon_plot+1]
    ir3XY = np.quantile(ir3, quantiles, axis=2)[:, :, :horizon_plot+1]
    ir4XY = np.quantile(ir4, quantiles, axis=2)[:, :, :horizon_plot+1]
    ir5XY = np.quantile(ir5, quantiles, axis=2)[:, :, :horizon_plot+1]
    ir6XY = np.quantile(ir6, quantiles, axis=2)[:, :, :horizon_plot+1]

    ir_list = [ir1XY, ir2XY, ir3XY, ir4XY, ir5XY, ir6XY]
    date_list = [date_range_1, date_range_2, date_range_3, date_range_4, date_range_5, date_range_6]

    for i in range(len(date_list)):

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(date_list[i], (ir_list[i][1, 0, :]*scale_adj[0, pol_eq]).T, c='teal', label='Mean IR')
        ax.fill_between(date_list[i], (ir_list[i][0, 0, :] * scale_adj[0, pol_eq]).T, (ir_list[i][2, 0, :] * scale_adj[0, pol_eq]).T, color='cadetblue', alpha=0.1, label='HPD interval')
        ax.grid(alpha=0.1)
        ax.set_title(f'Impulse response of {variables[0]}, {ir_dates_names[i]}')
        ax.set_xlabel('Time')
        ax.legend(loc='upper left')
        ax.set_ylim(bottom=-0.015, top=0.06)
        fig.tight_layout()
        if save_plots:
            plt.savefig(f'{variables[0]}_{ir_dates_names[i]}_SMSS.pdf', format='pdf')
        if show_plots:
            plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(date_list[i], (ir_list[i][1, 1, :] * scale_adj[1, pol_eq]).T, c='teal', label='Mean IR')
        ax.fill_between(date_list[i], (ir_list[i][0, 1, :] * scale_adj[1, pol_eq]).T, (ir_list[i][2, 1, :] * scale_adj[1, pol_eq]).T,
                        color='cadetblue', alpha=0.1, label='HPD interval')
        ax.grid(alpha=0.1)
        ax.set_title(f'Impulse response of {variables[1]}, {ir_dates_names[i]}')
        ax.set_xlabel('Time')
        ax.legend(loc='upper left')
        ax.set_ylim(bottom=-0.002, top=0.01)
        fig.tight_layout()
        if save_plots:
            plt.savefig(f'{variables[1]}_{ir_dates_names[i]}_SMSS.pdf', format='pdf')
        if show_plots:
            plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(date_list[i], (ir_list[i][1, 2, :] * scale_adj[2, pol_eq]).T, c='teal', label='Mean IR')
        ax.fill_between(date_list[i], (ir_list[i][0, 2, :] * scale_adj[2, pol_eq]).T, (ir_list[i][2, 2, :] * scale_adj[2, pol_eq]).T,
                        color='cadetblue', alpha=0.1, label='HPD interval')
        ax.grid(alpha=0.1)
        ax.set_title(f'Impulse response of {variables[2]}, {ir_dates_names[i]}')
        ax.set_xlabel('Time')
        ax.legend(loc='upper left')
        ax.set_ylim(bottom=-0.5, top=3.5)
        fig.tight_layout()
        if save_plots:
            plt.savefig(f'{variables[2]}_{ir_dates_names[i]}_SMSS.pdf', format='pdf')
        if show_plots:
            plt.show()

        if len(variables)==4:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(date_list[i], (ir_list[i][1, 3, :] * scale_adj[3, pol_eq]).T, c='teal', label='Mean IR')
            ax.fill_between(date_list[i], (ir_list[i][0, 3, :] * scale_adj[3, pol_eq]).T, (ir_list[i][2, 3, :] * scale_adj[3, pol_eq]).T,
                            color='cadetblue', alpha=0.1, label='HPD interval')
            ax.grid(alpha=0.1)
            ax.set_title(f'Impulse response of {variables[3]}, {ir_dates_names[i]}')
            ax.set_xlabel('Time')
            ax.legend(loc='upper left')
            ax.set_ylim(bottom=-0.1, top=0.2)
            fig.tight_layout()
            plt.savefig(f'{variables[3]}_{ir_dates_names[i]}_SMSS.pdf', format='pdf')
            if save_plots:
                plt.savefig(f'{variables[3]}_{ir_dates_names[i]}_SMSS.pdf', format='pdf')
            if show_plots:
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


