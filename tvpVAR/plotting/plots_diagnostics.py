import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({'font.size': 14})

""" User Settings """
output1 = 'resultsMCMC_AWM_full_4vars_conv_2lags_25k_1970_v5.npz'
output2 = 'resultsMCMC_AWM_diag_4vars_conv_2lags_25k_1970_v7.npz'
output3= 'resultsMCMC_AWM_full_4vars_conv_3lags_25k_1970_v2.npz'
output4 ='resultsMCMC_AWM_full_4vars_conv_2lags_50k_1970.npz'
output5 = 'resultsMCMC_AWM_full_5vars_conv_2lags_25k_1970_v5.npz'
output6 = 'resultsMCMC_AWM_diag_5vars_conv_2lags_25k_1970.npz'
output7 = 'resultsMCMC_AWM_full_5vars_conv_3lags_25k_1970.npz'
output8 = 'resultsMCMC_AWM_full_5vars_conv_2lags_50k_1970.npz'


output = [output1,output2,output3,output4,output5,output6,output7,output8]
models = ['4 vars, 2 lags, 25k','4 vars, 2 lags, 25k','4 vars, 3 lags, 25k','4 vars, 2 lags, 50k',
          '5 vars, 2 lags, 25k','5 vars, 2 lags, 25k','5 vars, 3 lags, 25k','5 vars, 2 lags, 50k']

models_rep = ['full: 4 vars, 2 lags, 25k','diag: 4 vars, 2 lags, 25k','full: 4 vars, 3 lags, 25k','full: 4 vars, 2 lags, 50k',
          'full: 5 vars, 2 lags, 25k','diag: 5 vars, 2 lags, 25k','full: 5 vars, 3 lags, 25k','full: 5 vars, 2 lags, 50k']


burnin_sims = 0  # extra simulations to be burned for the below calculations based on diagnostics
vars = [4,4,4,4,5,5,5,5]
save_plots = True  # save plots as pdf
show_plots = True  # show plots
ineff_file1 = "ineff_full_4vars_conv_2lags_25k_1970_v5.npz" 
ineff_file2 = "ineff_diag_4vars_conv_2lags_25k_1970_v7.npz"
ineff_file3 = "ineff_full_4vars_conv_3lags_25k_1970_v2.npz"
ineff_file4 = "ineff_diag_4vars_conv_2lags_50k_1970.npz"
ineff_file5 = "ineff_full_5vars_conv_2lags_25k_1970_v5.npz"
ineff_file6 = "ineff_diag_5vars_conv_2lags_25k_1970.npz"
ineff_file7 = "ineff_full_5vars_conv_3lags_25k_1970.npz"
ineff_file8 = "ineff_full_5vars_conv_2lags_50k_1970.npz"

""" Data Load: inefficiency factors """
data_ineff = np.load(ineff_file1)
ef_om_st1 = data_ineff['ef_om_st']
ef_beta1 = data_ineff['ef_beta']
data_ineff = np.load(ineff_file2)
ef_om_st2 = data_ineff['ef_om_st']
ef_beta2 = data_ineff['ef_beta']
data_ineff = np.load(ineff_file3)
ef_om_st3 = data_ineff['ef_om_st']
ef_beta3 = data_ineff['ef_beta']
data_ineff = np.load(ineff_file4)
ef_om_st4 = data_ineff['ef_om_st']
ef_beta4 = data_ineff['ef_beta']
data_ineff = np.load(ineff_file5)
ef_om_st5 = data_ineff['ef_om_st']
ef_beta5 = data_ineff['ef_beta']
data_ineff =np.load(ineff_file6)
ef_om_st6 = data_ineff['ef_om_st']
ef_beta6 = data_ineff['ef_beta']
data_ineff =np.load(ineff_file7)
ef_om_st7 = data_ineff['ef_om_st']
ef_beta7 = data_ineff['ef_beta']
data_ineff =np.load(ineff_file8)
ef_om_st8 = data_ineff['ef_om_st']
ef_beta8 = data_ineff['ef_beta']

ef_om_st_ls = [ef_om_st1[:,0],ef_om_st2[:,0],ef_om_st3[:,0],ef_om_st4[:,0],ef_om_st5[:,0],ef_om_st6[:,0],ef_om_st7[:,0],ef_om_st8[:,0]]
ef_beta_ls = [ef_beta1[:,0],ef_beta2[:,0],ef_beta3[:,0],ef_beta4[:,0],ef_beta5[:,0],ef_beta6[:,0],ef_beta7[:,0],ef_beta8[:,0]]

""" Ineff Boxplots """

# Estimate outliers
if True:
    om_df=pd.DataFrame(ef_om_st_ls)
    beta_df=pd.DataFrame(ef_beta_ls)
    all_om=np.sum(om_df>0,axis=1)
    all_beta=np.sum(beta_df>0,axis=1)
    om_150=np.sum(om_df<125, axis=1)
    beta_150=np.sum(beta_df<80, axis=1)
    om_per=om_150/all_om
    beta_per=beta_150/all_beta
    print(om_per)
    print(beta_per)

median_om_st = []
for i in range(len(ef_om_st_ls)):
    median_om_st.append(round(np.median(ef_om_st_ls[i]),1))
                        
median_beta = []
for i in range(len(ef_beta_ls)):
    median_beta.append(round(np.median(ef_beta_ls[i]),1)) 
                        
fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(16,12)) 

for a in ax.ravel():
    a.tick_params(axis='x', rotation=-60)

flierprops = dict(marker='_', markerfacecolor='slategrey', markersize=4, linestyle='none', markeredgecolor='slategrey', label='Outliers')
medianprops = dict(color='darkred', label='Median')
meanprops = dict(color='darkorange', label='Mean')     

bplot1=ax[0].boxplot(ef_om_st_ls, labels=models, flierprops=flierprops,
            medianprops=medianprops, meanprops=meanprops, showmeans=True, meanline=True, widths = 0.4, patch_artist=True)
for i in range(len(ef_om_st_ls)):
    ax[0].text((i+1.23),median_om_st[i], median_om_st[i],color='darkred',fontsize=11)
ax[0].set_xlabel('Model')
ax[0].set_ylabel('Inefficiency Factors')
ax[0].set_title(r'Boxplots of Inefficiency Factors for $\mathbf{\omega^*}$')
ax[0].set_ylim(0,125)

bplot2=ax[1].boxplot(ef_beta_ls, labels=models, flierprops=flierprops,
            medianprops=medianprops, meanprops=meanprops, showmeans=True, meanline=True, widths = 0.4,patch_artist=True)
for i in range(len(ef_beta_ls)):
    ax[1].text((i+1.23),median_beta[i], median_beta[i],color='darkred',fontsize=11)
ax[1].set_xlabel('Model')
#ax[0,1].ylabel('Inefficiency Factors')
ax[1].set_title(r'Boxplots of Inefficiency Factors for $\mathbf{\beta}$')
ax[1].set_ylim(0,80)

colors = ['cadetblue', 'coral', 'cadetblue','cadetblue','cadetblue', 'coral','cadetblue','cadetblue']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

ax[0].legend([bplot1['boxes'][0],bplot2['boxes'][1],bplot1['medians'][0],bplot1['means'][0],bplot1['fliers'][0]],['SMSS full', 'SMSS diag','median','mean','outlier'],loc='upper right')

fig.tight_layout()

if save_plots:
    fig.savefig('Ineff_factors.pdf', format='pdf')
if show_plots:
    plt.show()

""" Data Load """

# Load the relevant np.ndarrays from MCMC sampler results file saved in .npz format

p_tiv_b_all = []
p_tiv1_all = []
p_tiv2_all = []
p_tiv3_all = []
p_tiv4_all = []
p_tiv5_all = []
p_tiv_b_all = []
mbeta_b_all = []
mbeta1_all = []
mbeta2_all = []
mbeta3_all = []
mbeta4_all = []
mbeta5_all = []

for i in range(1,len(output)+1):

    data = np.load(output[i-1])
    s_beta = data['s_beta']
    cidx = data['cidx']
    bidx = data['bidx']
    nk = data['nk'].item()
    t = data['t'].item()
    svsims = data['svsims'].item()
    p = data['p'].item()
    s_om=data['s_om']

    ''' MTV '''

    mbeta0 = np.mean(s_beta, axis=2).T
    mbeta = np.zeros(mbeta0.shape)
    mbeta[:, :np.count_nonzero(cidx)] = mbeta0[:, cidx.ravel()]
    mbeta[:, np.count_nonzero(cidx):] = mbeta0[:, bidx.ravel()]

    """ Time invariant/constant parameter probabilities"""
    # Plot the time invariant/constant param probabilities
    p_tiv0 = np.sum(s_om == 0, axis=1)/svsims
    p_tiv = np.zeros(p_tiv0.shape)
    p_tiv[:np.count_nonzero(cidx)] = p_tiv0[cidx.ravel()]
    p_tiv[np.count_nonzero(cidx):] = p_tiv0[bidx.ravel()]

    if vars[i-1] == 5:
        
        mbeta_b_all.append(mbeta[:, :np.count_nonzero(cidx)])
        mbeta1_all.append(mbeta[:, np.count_nonzero(cidx):(np.count_nonzero(cidx)+(vars[i-1]*p+1))])
        mbeta2_all.append(mbeta[:, (np.count_nonzero(cidx) + (vars[i - 1] * p + 1)):(np.count_nonzero(cidx) + (vars[i - 1] * p + 1)*2)])
        mbeta3_all.append(mbeta[:, (np.count_nonzero(cidx) + (vars[i - 1] * p + 1)*2 ):(
                    np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 3 )])
        mbeta4_all.append(mbeta[:, (np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 3 ):(
                np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 4 )])
        mbeta5_all.append(mbeta[:, (np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 4):(
                np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 5)])
    
        p_tiv_b_all.append(p_tiv[:np.count_nonzero(cidx)])
        p_tiv1_all.append(p_tiv[np.count_nonzero(cidx):(np.count_nonzero(cidx) + (vars[i - 1] * p + 1))])
        p_tiv2_all.append(p_tiv[(np.count_nonzero(cidx) + (vars[i - 1] * p + 1)):(
                    np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 2)])
        p_tiv3_all.append(p_tiv[(np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 2 ):(
                np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 3)])
        p_tiv4_all.append(p_tiv[(np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 3):(
                np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 4 )])
        p_tiv5_all.append(p_tiv[(np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 4):(
                np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 5)])

    if vars[i-1] == 4:
        
        mbeta_b_all.append(mbeta[:, :np.count_nonzero(cidx)])
        mbeta1_all.append(np.empty((mbeta.shape[0],(vars[i-1]*p+1)),)*np.nan)
        mbeta2_all.append(mbeta[:, np.count_nonzero(cidx):(np.count_nonzero(cidx)+(vars[i-1]*p+1))])
        mbeta3_all.append(mbeta[:, (np.count_nonzero(cidx) + (vars[i - 1] * p + 1)):(np.count_nonzero(cidx) + (vars[i - 1] * p + 1)*2)])
        mbeta4_all.append(mbeta[:, (np.count_nonzero(cidx) + (vars[i - 1] * p + 1)*2 ):(
                    np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 3 )])
        mbeta5_all.append(mbeta[:, (np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 3 ):(
                np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 4 )])
    
        p_tiv_b_all.append(p_tiv[:np.count_nonzero(cidx)])
        p_tiv1_all.append(np.empty(((vars[i-1]*p+1)),)*np.nan)
        p_tiv2_all.append(p_tiv[np.count_nonzero(cidx):(np.count_nonzero(cidx) + (vars[i - 1] * p + 1))])
        p_tiv3_all.append(p_tiv[(np.count_nonzero(cidx) + (vars[i - 1] * p + 1)):(
                    np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 2)])
        p_tiv4_all.append(p_tiv[(np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 2 ):(
                np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 3)])
        p_tiv5_all.append(p_tiv[(np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 3):(
                np.count_nonzero(cidx) + (vars[i - 1] * p + 1) * 4 )])

# Names of variables
b4=[r'$b_1$',r'$b_2$',r'$b_3$',r'$b_4$',r'$b_5$',r'$b_6$']
b5=[r'$b_1$',r'$b_2$',r'$b_3$',r'$b_4$',r'$b_5$',r'$b_6$',r'$b_7$',r'$b_8$',r'$b_9$',r'$b_{10}$']

beta4_2=[r'$\beta_0$', r'$YER_{1}$',r'$YED_{1}$',r'$STN_{1}$',r'$EEN_{1}$', r'$YER_{2}$',r'$YED_{2}$',r'$STN_{2}$',r'$EEN_{2}$']
beta4_3=[r'$\beta_0$', r'$YER_{1}$',r'$YED_{1}$',r'$STN_{1}$',r'$EEN_{1}$', r'$YER_{2}$',r'$YED_{2}$',r'$STN_{2}$',r'$EEN_{2}$', r'$YER_{3}$',r'$YED_{3}$',r'$STN_{3}$',r'$EEN_{3}$']
beta5_2=[r'$\beta_0$',r'$COMPR_{1}$', r'$YER_{1}$',r'$YED_{1}$',r'$STN_{1}$',r'$EEN_{1}$',r'$COMPR_{2}$', r'$YER_{2}$',r'$YED_{2}$',r'$STN_{2}$',r'$EEN_{2}$']
beta5_3=[r'$\beta_0$',r'$COMPR_{1}$', r'$YER_{1}$',r'$YED_{1}$',r'$STN_{1}$',r'$EEN_{1}$',r'$COMPR_{2}$', r'$YER_{2}$',r'$YED_{2}$',r'$STN_{2}$',r'$EEN_{2}$',r'$COMPR_{3}$', r'$YER_{3}$',r'$YED_{3}$',r'$STN_{3}$',r'$EEN_{3}$']

colnames=[r'$\beta_0$',r'$COMPR_{1}$', r'$YER_{1}$',r'$YED_{1}$',r'$STN_{1}$',r'$EEN_{1}$',r'$COMPR_{2}$', r'$YER_{2}$',r'$YED_{2}$',r'$STN_{2}$',r'$EEN_{2}$',r'$COMPR_{3}$', r'$YER_{3}$',r'$YED_{3}$',r'$STN_{3}$',r'$EEN_{3}$']
bnames=[b4,b4,b4,b4,b5,b5,b5,b5]
betanames=[beta4_2,beta4_2,beta4_3,beta4_2,beta5_2,beta5_2,beta5_3,beta5_2]

linst='dashed'
linw=1
mark='x'

df_ls0=[]
df_ls1=[]
df_ls2=[]
df_ls3=[]
df_ls4=[]
df_ls5=[]
df_ls00=[]
df_ls11=[]
df_ls22=[]
df_ls33=[]
df_ls44=[]
df_ls55=[]

for i in range(len(output)):
   df_ls0.append(pd.DataFrame((np.max(mbeta_b_all[i], axis=0) - np.min(mbeta_b_all[i],axis=0)).reshape(-1,1).T,columns=bnames[i]))
   df_ls1.append(pd.DataFrame((np.max(mbeta1_all[i], axis=0) - np.min(mbeta1_all[i],axis=0)).reshape(-1,1).T,columns=betanames[i]))
   df_ls2.append(pd.DataFrame((np.max(mbeta2_all[i], axis=0) - np.min(mbeta2_all[i],axis=0)).reshape(-1,1).T,columns=betanames[i]))
   df_ls3.append(pd.DataFrame((np.max(mbeta3_all[i], axis=0) - np.min(mbeta3_all[i],axis=0)).reshape(-1,1).T,columns=betanames[i]))
   df_ls4.append(pd.DataFrame((np.max(mbeta4_all[i], axis=0) - np.min(mbeta4_all[i],axis=0)).reshape(-1,1).T,columns=betanames[i]))
   df_ls5.append(pd.DataFrame((np.max(mbeta5_all[i], axis=0) - np.min(mbeta5_all[i],axis=0)).reshape(-1,1).T,columns=betanames[i]))
   df_ls00.append(pd.DataFrame(p_tiv_b_all[i].reshape(-1,1).T,columns=bnames[i]))
   df_ls11.append(pd.DataFrame(p_tiv1_all[i].reshape(-1,1).T,columns=betanames[i]))
   df_ls22.append(pd.DataFrame(p_tiv2_all[i].reshape(-1,1).T,columns=betanames[i]))
   df_ls33.append(pd.DataFrame(p_tiv3_all[i].reshape(-1,1).T,columns=betanames[i]))
   df_ls44.append(pd.DataFrame(p_tiv4_all[i].reshape(-1,1).T,columns=betanames[i]))
   df_ls55.append(pd.DataFrame(p_tiv5_all[i].reshape(-1,1).T,columns=betanames[i]))

df0=pd.concat(df_ls0).T
df0.columns=models_rep
df1=pd.concat(df_ls1)
df1=df1[colnames].T
df1.columns=models_rep
df2=pd.concat(df_ls2)
df2=df2[colnames].T
df2.columns=models_rep
df3=pd.concat(df_ls3)
df3=df3[colnames].T
df3.columns=models_rep
df4=pd.concat(df_ls4)
df4=df4[colnames].T
df4.columns=models_rep
df5=pd.concat(df_ls5)
df5=df5[colnames].T
df5.columns=models_rep
df00=pd.concat(df_ls00).T
df00.columns=models_rep
df11=pd.concat(df_ls11)
df11=df11[colnames].T
df11.columns=models_rep
df22=pd.concat(df_ls22)
df22=df22[colnames].T
df22.columns=models_rep
df33=pd.concat(df_ls33)
df33=df33[colnames].T
df33.columns=models_rep
df44=pd.concat(df_ls44)
df44=df44[colnames].T
df44.columns=models_rep
df55=pd.concat(df_ls55)
df55=df55[colnames].T
df55.columns=models_rep

colors2 = ['teal', 'tab:red','olivedrab','darkorange','steelblue','midnightblue','mediumvioletred','grey']

# MTV #
w = 0.8
alpha = float(0.8)

fig1, ax1 = plt.subplots(nrows=6, ncols=1, figsize=(19,21)) 

df0.plot.bar(color=colors2, linestyle=linst, linewidth=linw, ax=ax1[0],width=w,alpha=alpha,legend=False)
df1.plot.bar(color=colors2, linestyle=linst, linewidth=linw,  ax=ax1[1],width=w,alpha=alpha,legend=False) 
df2.plot.bar(color=colors2, linestyle=linst, linewidth=linw,ax=ax1[2],width=w,alpha=alpha,legend=False) 
df3.plot.bar(color=colors2, linestyle=linst, linewidth=linw, ax=ax1[3],width=w,alpha=alpha,legend=True) 
df4.plot.bar(color=colors2, linestyle=linst, linewidth=linw,ax=ax1[4],width=w,alpha=alpha,legend=False) 
df5.plot.bar(color=colors2, linestyle=linst, linewidth=linw,  ax=ax1[5],width=w,alpha=alpha,legend=False) 
ax1[0].set_xticks(np.arange(0,len(df0.index.to_list())))
ax1[0].set_xticklabels(df0.index.to_list())
ax1[1].set_xticks(np.arange(0,len(df1.index.to_list())))
ax1[1].set_xticklabels(df1.index.to_list())
ax1[2].set_xticks(np.arange(0,len(df2.index.to_list())))
ax1[2].set_xticklabels(df2.index.to_list())
ax1[3].set_xticks(np.arange(0,len(df3.index.to_list())))
ax1[3].set_xticklabels(df3.index.to_list())
ax1[4].set_xticks(np.arange(0,len(df4.index.to_list())))
ax1[4].set_xticklabels(df4.index.to_list())
ax1[5].set_xticks(np.arange(0,len(df5.index.to_list())))
ax1[5].set_xticklabels(df5.index.to_list())
ax1[3].legend(loc='lower right')

ax1[0].set_title(r'MTV of Coefficients: Max Avg $\beta_{i}$ - Min Avg $\beta_{i}$')
ax1[5].set_xlabel('Coefficient')
ax1[0].set_ylabel(r'$\mathbf{B_{0,t}}$')
ax1[1].set_ylabel(r'$COMPR_t$ Equation')
ax1[2].set_ylabel(r'$YER_t$ Equation')
ax1[3].set_ylabel(r'$YED_t$ Equation')
ax1[4].set_ylabel(r'$STN_t$ Equation')
ax1[5].set_ylabel(r'$EEN_t$ Equation')

for a in ax1.ravel():
   a.tick_params(axis='x', rotation=0)
   a.set_ylim()
   a.ticklabel_format(axis='y', style='', scilimits=(-3,3))
   a.grid(alpha=0.1)

fig1.tight_layout()

if save_plots:
    fig1.savefig('MTV_beta_bar.pdf', format='pdf')
if show_plots:
    plt.show()

fig2, ax2 = plt.subplots(nrows=6, ncols=1, figsize=(19,21)) 

df00.plot.bar(color=colors2, linestyle=linst, linewidth=linw, ax=ax2[0],width=w,alpha=alpha,legend=False) 
df11.plot.bar(color=colors2, linestyle=linst, linewidth=linw,  ax=ax2[1],width=w,alpha=alpha,legend=False) 
df22.plot.bar(color=colors2, linestyle=linst, linewidth=linw,ax=ax2[2],width=w,alpha=alpha,legend=False) 
df33.plot.bar(color=colors2, linestyle=linst, linewidth=linw, ax=ax2[3],width=w,alpha=alpha,legend=True) 
df44.plot.bar(color=colors2, linestyle=linst, linewidth=linw,ax=ax2[4],width=w,alpha=alpha,legend=False) 
df55.plot.bar(color=colors2, linestyle=linst, linewidth=linw,  ax=ax2[5],width=w,alpha=alpha,legend=False) 
ax2[0].set_xticks(np.arange(0,len(df00.index.to_list())))
ax2[0].set_xticklabels(df00.index.to_list())
ax2[1].set_xticks(np.arange(0,len(df11.index.to_list())))
ax2[1].set_xticklabels(df11.index.to_list())
ax2[2].set_xticks(np.arange(0,len(df22.index.to_list())))
ax2[2].set_xticklabels(df22.index.to_list())
ax2[3].set_xticks(np.arange(0,len(df33.index.to_list())))
ax2[3].set_xticklabels(df33.index.to_list())
ax2[4].set_xticks(np.arange(0,len(df44.index.to_list())))
ax2[4].set_xticklabels(df44.index.to_list())
ax2[5].set_xticks(np.arange(0,len(df55.index.to_list())))
ax2[5].set_xticklabels(df55.index.to_list())
ax1[3].legend(loc='lower right')
ax2[0].set_title(r'TIP of Coefficients')
ax2[5].set_xlabel('Coefficient')
ax2[0].set_ylabel(r'$\mathbf{B_{0,t}}$')
ax2[1].set_ylabel(r'$COMPR_t$ Equation')
ax2[2].set_ylabel(r'$YER_t$ Equation')
ax2[3].set_ylabel(r'$YED_t$ Equation')
ax2[4].set_ylabel(r'$STN_t$ Equation')
ax2[5].set_ylabel(r'$EEN_t Equation$')

for a in ax2.ravel():
    a.tick_params(axis='x', rotation=0)
    a.set_ylim((0,1))
    a.grid(alpha=0.1)
    
fig2.tight_layout()

if save_plots:
    fig2.savefig('TIV_beta_bar.pdf', format='pdf')
if show_plots:
    plt.show()
