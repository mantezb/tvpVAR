import timeit
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from tvpVAR.utils.coint_johansen import coint_johansen

# Specification of directories
base_path = path.dirname(__file__)  # Location of the main.py
data_path = path.abspath(path.join(base_path, 'data'))  # The path where the data is stored

""" User Settings """
# Data specification
filename = 'AWM_5vars.csv'
output = 'AWM_5vars_conv_n.csv'
vars = 5
variables = ['Commodity Prices', 'Real output', 'Prices', 'Interest rate', 'Exchange Rate']
scale_data = True
center_data = True

data = pd.read_csv(path.join(data_path, filename), header=None)
y_data = data.to_numpy()[:, :int(vars)]

for i in range(vars):
    plt.plot(y_data[:, i])
    plt.show()

corr = np.corrcoef(y_data.T)
print(corr)

y_data_conv = np.empty((y_data.shape[0]-1, y_data.shape[1]))
y_data_conv[:, 0] = [(np.log(y_data[i+1, 0])-np.log(y_data[i, 0])) for i in range(y_data.shape[0]-1)]
y_data_conv[:, 1] = [(np.log(y_data[i+1, 1])-np.log(y_data[i, 1])) for i in range(y_data.shape[0]-1)]
y_data_conv[:, 2] = [(np.log(y_data[i+1, 2])-np.log(y_data[i, 2])) for i in range(y_data.shape[0]-1)]
#y_data_conv[:, 2] = [(y_data[i+1, 2]-y_data[i, 2]) for i in range(y_data.shape[0]-1)]
y_data_conv[:, 3] = y_data[1:, 3]
y_data_conv[:, 4] = [(np.log(y_data[i+1, 4])-np.log(y_data[i, 4])) for i in range(y_data.shape[0]-1)]

for i in range(vars):
    plt.plot(y_data_conv[:,i])
    plt.show()

corr_conv = np.corrcoef(y_data_conv.T)
print(corr_conv)

new_data = pd.DataFrame(y_data_conv)
new_data.to_csv(output)

dscale1 = 1 + scale_data * (np.std(y_data, axis=0) - 1)
dcenter1 = center_data * np.mean(y_data, axis=0)
y_data_stand = (y_data - dcenter1 )/ dscale1
dscale2 = 1 + scale_data * (np.std(y_data_conv, axis=0) - 1)
dcenter2 = center_data * np.mean(y_data_conv, axis=0)
y_data_conv_stand = (y_data_conv - dcenter2) / dscale2

data1=pd.DataFrame(y_data_stand)
data2=pd.DataFrame(y_data_conv_stand)


for i in range(vars):
    plt.plot(y_data_stand[:, i])
    plt.show()
for i in range(vars):
    plt.plot(y_data_conv_stand[:,i])
    plt.show()
    
# Augmented Dickey-Fuller test of unit roots

for i in range(vars):
    data1[i].hist()
    plt.show()
    Y = data1[i].values
    result = adfuller(Y)
    print(f'{variables[i]}')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

for i in range(vars):
    data2[i].hist()
    plt.show()
    Y_conv = data2[i].values
    result = adfuller(Y_conv)
    print(f'{variables[i]}')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))



#  Johansen test of cointegration - data in levels!
print(f'Johnsen Test results at Lag length 1')
coint_johansen(data, 0, 1)
print(f'Johnsen Test results at Lag length 2')
coint_johansen(data, 0, 2)
print(f'Johnsen Test results at Lag length 3')
coint_johansen(data, 0, 3)
print(f'Johnsen Test results at Lag length 4')
coint_johansen(data, 0, 4)
