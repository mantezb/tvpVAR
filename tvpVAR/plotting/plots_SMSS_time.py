import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from dateutil.relativedelta import relativedelta

variables = ['Commodity Prices', 'Real output','Prices','Interest rate', 'Exchange Rate']
ir_dates = [1980.00, 1990.00, 1996.00, 1999.00, 2009.00, 1980.00]
ir_dates2 = [1990.00, 1996.00, 1999.00, 2009.00,2015.00,1999.00]
ir_dates3 = [1996.00, 1999.00, 2009.00,2015.00,1980.00,1990.00]
ir_dates4 = [1999.00, 2009.00,2015.00,1980.00,1990.00, 1996.00]
ir_dates5 = [2009.00,2015.00,1980.00,1990.00, 1996.00, 1999.00]
plots = 2

horizon_plot = 12
scaling = False
policy_equation = ['Interest rate']
save_plots = True
show_plots = False
ax_adj = False
ax1_adj = False
model = 'Full_5var_2lag_1970'
ir_file = 'ir_AWM_full_5vars_conv_2lags_25k_1970_v5_new.npz'
#ir_file2 ='ir_AWM_diag_4vars_conv_2lags_25k_1970_v5.npz'
data = np.load(ir_file)
#data2 = np.load(ir_file2)
quantiles = [0.16, 0.5, 0.84]
ax1=[-0.015, 0.06]
ax2=[-0.002, 0.01]
ax3=[-0.5, 3.5]
ax4=[-0.1, 0.2]
ax5=[-0.1, 0.2]

horizon=[0,1,2,3,4,5,6,7,8,9,10,11,12]

pol_eq = np.where(np.in1d(variables, policy_equation))[0].item()

date_dict = {0: 'Q1', 0.25: 'Q2', 0.5: 'Q3', 0.75: 'Q4'}
ir_dates_names = [f'{int(ir_dates[0])}-{date_dict[np.mod(ir_dates[0],1)]}', f'{int(ir_dates[1])}-{date_dict[np.mod(ir_dates[1],1)]}',
                  f'{int(ir_dates[2])}-{date_dict[np.mod(ir_dates[2],1)]}', f'{int(ir_dates[3])}-{date_dict[np.mod(ir_dates[3],1)]}',
                  f'{int(ir_dates[4])}-{date_dict[np.mod(ir_dates[4],1)]}', f'{int(ir_dates[5])}-{date_dict[np.mod(ir_dates[5],1)]}']

ir_dates2_names = [f'{int(ir_dates2[0])}-{date_dict[np.mod(ir_dates2[0],1)]}', f'{int(ir_dates2[1])}-{date_dict[np.mod(ir_dates2[1],1)]}',
                  f'{int(ir_dates2[2])}-{date_dict[np.mod(ir_dates2[2],1)]}', f'{int(ir_dates2[3])}-{date_dict[np.mod(ir_dates2[3],1)]}',
                  f'{int(ir_dates2[4])}-{date_dict[np.mod(ir_dates2[4],1)]}', f'{int(ir_dates2[5])}-{date_dict[np.mod(ir_dates2[5],1)]}']

ir_dates3_names = [f'{int(ir_dates3[0])}-{date_dict[np.mod(ir_dates3[0],1)]}', f'{int(ir_dates3[1])}-{date_dict[np.mod(ir_dates3[1],1)]}',
                  f'{int(ir_dates3[2])}-{date_dict[np.mod(ir_dates3[2],1)]}', f'{int(ir_dates3[3])}-{date_dict[np.mod(ir_dates3[3],1)]}',
                  f'{int(ir_dates3[4])}-{date_dict[np.mod(ir_dates3[4],1)]}', f'{int(ir_dates3[5])}-{date_dict[np.mod(ir_dates3[5],1)]}']

ir_dates4_names = [f'{int(ir_dates4[0])}-{date_dict[np.mod(ir_dates4[0],1)]}', f'{int(ir_dates4[1])}-{date_dict[np.mod(ir_dates4[1],1)]}',
                  f'{int(ir_dates4[2])}-{date_dict[np.mod(ir_dates4[2],1)]}', f'{int(ir_dates4[3])}-{date_dict[np.mod(ir_dates4[3],1)]}',
                  f'{int(ir_dates4[4])}-{date_dict[np.mod(ir_dates4[4],1)]}', f'{int(ir_dates4[5])}-{date_dict[np.mod(ir_dates4[5],1)]}']

ir_dates5_names = [f'{int(ir_dates5[0])}-{date_dict[np.mod(ir_dates5[0],1)]}', f'{int(ir_dates5[1])}-{date_dict[np.mod(ir_dates5[1],1)]}',
                  f'{int(ir_dates5[2])}-{date_dict[np.mod(ir_dates5[2],1)]}', f'{int(ir_dates5[3])}-{date_dict[np.mod(ir_dates5[3],1)]}',
                  f'{int(ir_dates5[4])}-{date_dict[np.mod(ir_dates5[4],1)]}', f'{int(ir_dates5[5])}-{date_dict[np.mod(ir_dates5[5],1)]}']

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

### SMSS 1 ###

scale_adj=data['scale_adj']
ir1 = data['ir1']
ir2 = data['ir2']
ir3 = data['ir3']
ir4 = data['ir4']
ir5 = data['ir5']
ir6 = data['ir6']
ir1XY = np.quantile(ir1, quantiles, axis=2)[:, :, :horizon_plot+1]
ir2XY = np.quantile(ir2, quantiles, axis=2)[:, :, :horizon_plot+1]
ir3XY = np.quantile(ir3, quantiles, axis=2)[:, :, :horizon_plot+1]
ir4XY = np.quantile(ir4, quantiles, axis=2)[:, :, :horizon_plot+1]
ir5XY = np.quantile(ir5, quantiles, axis=2)[:, :, :horizon_plot+1]
ir6XY = np.quantile(ir6, quantiles, axis=2)[:, :, :horizon_plot+1]

ir_list = [ir1XY, ir2XY, ir3XY, ir4XY, ir5XY, ir1XY]

ir_list1 = [ir2XY, ir3XY, ir4XY, ir5XY, ir6XY, ir3XY]

ir_list2 = [ir3XY, ir4XY, ir5XY, ir6XY, ir1XY,ir2XY]

date_list = [date_range_1, date_range_2, date_range_3, date_range_4, date_range_5, date_range_6] 

color01='teal'
color02='cadetblue'

fig1, ax1 = plt.subplots(nrows=len(variables),ncols=int(len(ir_list)), figsize=(10*len(variables), 30))


for a in ax1.ravel():
    a.tick_params(axis='x', rotation=-45)
    
for i in range(int(len(date_list))):
        
    ax1[0,i].plot(date_list[i], (ir_list[i][1, 0, :] * scale_adj[0, pol_eq]).T, c=color01, label='IRF: TVP-VAR-SV')
    ax1[0,i].fill_between(date_list[i], (ir_list[i][0, 0, :] * scale_adj[0,pol_eq]).T, (ir_list[i][2, 0, :] * scale_adj[0, pol_eq]).T,
                       color=color02, alpha=0.1, label='HPD int: TVP-VAR-SV')
    ax1[0,i].grid(alpha=0.1)
    ax1[0,i].set_title(f'{ir_dates_names[i]}')
    if i == 0:
        ax1[0,i].set_ylabel(f'{variables[0]}')
    if ax1_adj:
        ax1[0,i].set_ylim(bottom=ax1[0], top=ax1[1])
            
    ax1[1,i].plot(date_list[i], (ir_list[i][1, 1, :] * scale_adj[1, pol_eq]).T, c=color01, label='IRF: TVP-VAR-SV')
    ax1[1,i].fill_between(date_list[i], (ir_list[i][0, 1, :] * scale_adj[1,pol_eq]).T, (ir_list[i][2, 1, :] * scale_adj[1, pol_eq]).T,
                    color=color02, alpha=0.1, label='HPD int: TVP-VAR-SV')
    ax1[1,i].grid(alpha=0.1)
    if i== 0:
        ax1[1,i].set_ylabel(f'{variables[1]}')
    if ax1_adj:
        ax1[1,i].set_ylim(bottom=ax2[0], top=ax2[1])
    
    ax1[2,i].plot(date_list[i], (ir_list[i][1, 2, :] * scale_adj[2, pol_eq]).T, c=color01, label='IRF: TVP-VAR-SV')
    ax1[2,i].fill_between(date_list[i], (ir_list[i][0, 2, :] * scale_adj[2, pol_eq]).T, (ir_list[i][2, 2, :] * scale_adj[2, pol_eq]).T,
                          color=color02, alpha=0.1, label='HPD int: TVP-VAR-SV')
    ax1[2,i].grid(alpha=0.1)
    if i == 0:
        ax1[2,i].set_ylabel(f'{variables[2]}')
    if ax1_adj:
        ax1[2,i].set_ylim(bottom=ax3[0], top=ax3[1])
    
    if len(variables) >= 4:
        ax1[3,i].plot(date_list[i], (ir_list[i][1, 3, :] * scale_adj[3, pol_eq]).T, c=color01, label='IRF: TVP-VAR-SV')
        ax1[3,i].fill_between(date_list[i], (ir_list[i][0, 3, :] * scale_adj[3, pol_eq]).T,
                        (ir_list[i][2, 3, :] * scale_adj[3, pol_eq]).T,
                        color=color02, alpha=0.1, label='HPD int: TVP-VAR-SV')
        ax1[3,i].grid(alpha=0.1)
        if i== 0:
            ax1[3,i].set_ylabel(f'{variables[3]}')
        if len(variables)==4:
            ax1[3,i].set_xlabel('Time')
        if len(variables) == 4 and i==0:
            ax1[3,i].legend(loc='upper right')
        if ax1_adj:
            ax1[3,i].set_ylim(bottom=ax4[0], top=ax4[1])
    
    if len(variables) >= 5:
        ax1[4,i].plot(date_list[i], (ir_list[i][1, 4, :] * scale_adj[4, pol_eq]).T, c=color01, label='IRF: TVP-VAR-SV')
        ax1[4,i].fill_between(date_list[i], (ir_list[i][0, 4, :] * scale_adj[4, pol_eq]).T,
                           (ir_list[i][2, 4, :] * scale_adj[4, pol_eq]).T,
                           color=color02, alpha=0.1, label='HPD int: TVP-VAR-SV')
        ax1[4,i].grid(alpha=0.1)
        if i == 0:
            ax1[4,i].set_ylabel(f'{variables[4]}')
        if len(variables)==5:
            ax1[4,i].set_xlabel('Time')
        if len(variables) == 5 and i==0:
            ax1[4,i].legend(loc='upper right')
        if ax1_adj:
            ax1[4,i].set_ylim(bottom=ax5[0], top=ax5[1])   
    

fig1.tight_layout()
    
if save_plots:
       fig1.savefig(f'Plot_SMSS_ALL.pdf', format='pdf')
if show_plots:
       plt.show()  




date_list = [date_range_1, date_range_2, date_range_3, date_range_4, date_range_5, date_range_6]  # date_range_3, date_range_4, date_range_5, date_range_6

fig, ax = plt.subplots(nrows=len(variables),ncols=int(len(ir_list)/3), figsize=(10*len(variables), 20))

color01='teal'
color02='cadetblue'

for j in range(3):
    
    fig, ax = plt.subplots(nrows=len(variables),ncols=int(len(ir_list)/3), figsize=(18, 20)) # figsize=(15, 15)
    
    for i in range(j*2,int(len(date_list)/3*(j+1))):
        
        ax[0,i-2*j].plot(horizon, (ir_list[i][1, 0, :] * scale_adj[0, pol_eq]).T, c=color01, label=f'IRF: {ir_dates_names[i]}')
        ax[0,i-2*j].fill_between(horizon, (ir_list[i][0, 0, :] * scale_adj[0,pol_eq]).T, (ir_list[i][2, 0, :] * scale_adj[0, pol_eq]).T,
                        color=color02, alpha=0.1, label=f'HPD int: {ir_dates_names[i]}')
        ax[0,i-2*j].grid(alpha=0.1)
        ax[0,i-2*j].set_title(f'{ir_dates_names[i]} vs {ir_dates2_names[i]} ')
        if i-2*j == 0:
            ax[0,i-2*j].set_ylabel(f'{variables[0]}')
        if ax_adj:
            ax[0,i-2*j].set_ylim(bottom=ax1[0], top=ax1[1])
            
        ax[1,i-2*j].plot(horizon, (ir_list[i][1, 1, :] * scale_adj[1, pol_eq]).T, c=color01, label=f'IRF: {ir_dates_names[i]}')
        ax[1,i-2*j].fill_between(horizon, (ir_list[i][0, 1, :] * scale_adj[1,pol_eq]).T, (ir_list[i][2, 1, :] * scale_adj[1, pol_eq]).T,
                        color=color02, alpha=0.1, label=f'HPD int: {ir_dates_names[i]}')
        ax[1,i-2*j].grid(alpha=0.1)
        if i-2*j == 0:
            ax[1,i-2*j].set_ylabel(f'{variables[1]}')
        if ax_adj:
            ax[1,i-2*j].set_ylim(bottom=ax2[0], top=ax2[1])
    
        ax[2,i-2*j].plot(horizon, (ir_list[i][1, 2, :] * scale_adj[2, pol_eq]).T, c=color01, label=f'IRF: {ir_dates_names[i]}')
        ax[2,i-2*j].fill_between(horizon, (ir_list[i][0, 2, :] * scale_adj[2, pol_eq]).T, (ir_list[i][2, 2, :] * scale_adj[2, pol_eq]).T,
                        color=color02, alpha=0.1, label=f'HPD int: {ir_dates_names[i]}')
        ax[2,i-2*j].grid(alpha=0.1)
        if i-2*j == 0:
            ax[2,i-2*j].set_ylabel(f'{variables[2]}')
        if ax_adj:
            ax[2,i-2*j].set_ylim(bottom=ax3[0], top=ax3[1])
    
        if len(variables) >= 4:
            ax[3,i-2*j].plot(horizon, (ir_list[i][1, 3, :] * scale_adj[3, pol_eq]).T, c=color01, label=f'IRF: {ir_dates_names[i]}')
            ax[3,i-2*j].fill_between(horizon, (ir_list[i][0, 3, :] * scale_adj[3, pol_eq]).T,
                            (ir_list[i][2, 3, :] * scale_adj[3, pol_eq]).T,
                            color=color02, alpha=0.1, label=f'HPD int: {ir_dates_names[i]}')
            ax[3,i-2*j].grid(alpha=0.1)
            if i-2*j == 0:
                ax[3,i-2*j].set_ylabel(f'{variables[3]}')
            if len(variables)==4:
                ax[3,i-2*j].set_xlabel('IR horizon (quarters)')
            if len(variables) == 4:
                ax[3,i-2*j].legend(loc='best')
            if ax_adj:
                ax[3,i-2*j].set_ylim(bottom=ax4[0], top=ax4[1])
    
        if len(variables) >= 5:
            ax[4,i-2*j].plot(horizon, (ir_list[i][1, 4, :] * scale_adj[4, pol_eq]).T, c=color01, label=f'IRF: {ir_dates_names[i]}')
            ax[4,i-2*j].fill_between(horizon, (ir_list[i][0, 4, :] * scale_adj[4, pol_eq]).T,
                            (ir_list[i][2, 4, :] * scale_adj[4, pol_eq]).T,
                            color=color02, alpha=0.1, label=f'HPD int: {ir_dates_names[i]}')
            ax[4,i-2*j].grid(alpha=0.1)
            if i-2*j == 0:
                ax[4,i-2*j].set_ylabel(f'{variables[4]}')
            if len(variables)==5:
                ax[4,i-2*j].set_xlabel('Time')
            if len(variables) == 5:
                ax[4,i-2*j].legend(loc='best')
            if ax_adj:
                ax[4,i-2*j].set_ylim(bottom=ax5[0], top=ax5[1])

    color1= "orangered"
    color2="coral"

    for i in range(j*2,int(len(date_list)/3*(j+1))):
        ax[0,i-2*j].plot(horizon, (ir_list1[i][1, 0, :]*scale_adj[0, pol_eq]).T, c=color1, label=f'IRF: {ir_dates2_names[i]}')
        ax[0,i-2*j].fill_between(horizon, (ir_list1[i][0, 0, :] * scale_adj[0, pol_eq]).T, (ir_list1[i][2, 0, :] * scale_adj[0, pol_eq]).T, 
                             color=color2, alpha=0.1, label=f'HPD int: {ir_dates2_names[i]}')
        ax[0,i-2*j].grid(alpha=0.1)
        if ax_adj:
            ax[0,i-2*j].set_ylim(bottom=ax1[0], top=ax1[1])
    
    
        ax[1,i-2*j].plot(horizon, (ir_list1[i][1, 1, :] * scale_adj[1, pol_eq]).T, c=color1, label=f'IRF: {ir_dates2_names[i]}')
        ax[1,i-2*j].fill_between(horizon, (ir_list1[i][0, 1, :] * scale_adj[1, pol_eq]).T, (ir_list1[i][2, 1, :] * scale_adj[1, pol_eq]).T,
                        color=color2, alpha=0.1, label=f'HPD int: {ir_dates2_names[i]}')
        ax[1,i-2*j].grid(alpha=0.1)
        if ax_adj:
            ax[1,i-2*j].set_ylim(bottom=ax2[0], top=ax2[1])
    
        ax[2,i-2*j].plot(horizon, (ir_list1[i][1, 2, :] * scale_adj[2, pol_eq]).T, c=color1, label=f'IRF: {ir_dates2_names[i]}')
        ax[2,i-2*j].fill_between(horizon, (ir_list1[i][0, 2, :] * scale_adj[2, pol_eq]).T, (ir_list1[i][2, 2, :] * scale_adj[2, pol_eq]).T,
                        color=color2, alpha=0.1, label=f'HPD int: {ir_dates2_names[i]}')
        ax[2,i-2*j].grid(alpha=0.1)
        if ax_adj:
               ax[2,i-2*j].set_ylim(bottom=ax3[0], top=ax3[1])
               
        if len(variables) >= 4:
            
            ax[3,i-2*j].plot(horizon, (ir_list1[i][1, 3, :] * scale_adj[3, pol_eq]).T, c=color1, label=f'IRF: {ir_dates2_names[i]}')
            ax[3,i-2*j].fill_between(horizon, (ir_list1[i][0, 3, :] * scale_adj[3, pol_eq]).T, (ir_list1[i][2, 3, :] * scale_adj[3, pol_eq]).T,
                            color=color2, alpha=0.1, label=f'HPD int: {ir_dates2_names[i]}')
            ax[3,i-2*j].grid(alpha=0.1)
            if len(variables) == 4:
                ax[3,i-2*j].legend(loc='best')
            if ax_adj:
                ax[3,i-2*j].set_ylim(bottom=ax4[0], top=ax4[1])
    
        if len(variables) >= 5:
            ax[4,i-2*j].plot(horizon, (ir_list1[i][1, 4, :] * scale_adj[4, pol_eq]).T, c=color1, label=f'IRF: {ir_dates2_names[i]}')
            ax[4,i-2*j].fill_between(horizon, (ir_list1[i][0, 4, :] * scale_adj[4, pol_eq]).T, (ir_list1[i][2, 4, :] * scale_adj[4, pol_eq]).T,
                            color=color2, alpha=0.1, label=f'HPD int: {ir_dates2_names[i]}')
            ax[4,i-2*j].grid(alpha=0.1)
            if len(variables) == 5:
                ax[4,i-2*j].legend(loc='best')
            if ax_adj:
                ax[4,i-2*j].set_ylim(bottom=ax5[0], top=ax5[1])

    color12="orchid"
    color11= "mediumvioletred"

    if plots==3:
        
        for i in range(j*2,int(len(date_list)/3*(j+1))):
            ax[0,i-2*j].plot(horizon, (ir_list2[i][1, 0, :]*scale_adj[0, pol_eq]).T, c=color11, label=f'IRF: {ir_dates3_names[i]}')
            ax[0,i-2*j].fill_between(horizon, (ir_list2[i][0, 0, :] * scale_adj[0, pol_eq]).T, (ir_list2[i][2, 0, :] * scale_adj[0, pol_eq]).T, 
                                 color=color12, alpha=0.1, label=f'HPD int: {ir_dates3_names[i]}')
            ax[0,i-2*j].grid(alpha=0.1)
            if ax_adj:
                ax[0,i-2*j].set_ylim(bottom=ax1[0], top=ax1[1])
        
        
            ax[1,i-2*j].plot(horizon, (ir_list2[i][1, 1, :] * scale_adj[1, pol_eq]).T, c=color11, label=f'IRF: {ir_dates3_names[i]}')
            ax[1,i-2*j].fill_between(horizon, (ir_list2[i][0, 1, :] * scale_adj[1, pol_eq]).T, (ir_list2[i][2, 1, :] * scale_adj[1, pol_eq]).T,
                            color=color12, alpha=0.1, label=f'HPD int: {ir_dates3_names[i]}')
            ax[1,i-2*j].grid(alpha=0.1)
            if ax_adj:
                ax[1,i-2*j].set_ylim(bottom=ax2[0], top=ax2[1])
        
            ax[2,i-2*j].plot(horizon, (ir_list2[i][1, 2, :] * scale_adj[2, pol_eq]).T, c=color11, label=f'IRF: {ir_dates3_names[i]}')
            ax[2,i-2*j].fill_between(horizon, (ir_list2[i][0, 2, :] * scale_adj[2, pol_eq]).T, (ir_list2[i][2, 2, :] * scale_adj[2, pol_eq]).T,
                            color=color12, alpha=0.1, label=f'HPD int: {ir_dates3_names[i]}')
            ax[2,i-2*j].grid(alpha=0.1)
            if ax_adj:
                   ax[2,i-2*j].set_ylim(bottom=ax3[0], top=ax3[1])
                   
            if len(variables) >= 4:
                
                ax[3,i-2*j].plot(horizon, (ir_list2[i][1, 3, :] * scale_adj[3, pol_eq]).T, c=color11, label=f'IRF: {ir_dates3_names[i]}')
                ax[3,i-2*j].fill_between(horizon, (ir_list2[i][0, 3, :] * scale_adj[3, pol_eq]).T, (ir_list2[i][2, 3, :] * scale_adj[3, pol_eq]).T,
                                color=color12, alpha=0.1, label=f'HPD int: {ir_dates3_names[i]}')
                ax[3,i-2*j].grid(alpha=0.1)
                if len(variables) == 4:
                    ax[3,i-2*j].legend(loc='best')
                if ax_adj:
                    ax[3,i-2*j].set_ylim(bottom=ax4[0], top=ax4[1])
        
            if len(variables) >= 5:
                ax[4,i-2*j].plot(horizon, (ir_list2[i][1, 4, :] * scale_adj[4, pol_eq]).T, c=color11, label=f'IRF: {ir_dates3_names[i]}')
                ax[4,i-2*j].fill_between(horizon, (ir_list2[i][0, 4, :] * scale_adj[4, pol_eq]).T, (ir_list2[i][2, 4, :] * scale_adj[4, pol_eq]).T,
                                color=color12, alpha=0.1, label=f'HPD int: {ir_dates3_names[i]}')
                ax[4,i-2*j].grid(alpha=0.1)
                if len(variables) == 5:
                    ax[4,i-2*j].legend(loc='best')
                if ax_adj:
                    ax[4,i-2*j].set_ylim(bottom=ax5[0], top=ax5[1])

    fig.tight_layout()
    
    if save_plots:
       fig.savefig(f'Plot_SMSS_{model}_{j+1}.pdf', format='pdf')
    if show_plots:
       plt.show()  


