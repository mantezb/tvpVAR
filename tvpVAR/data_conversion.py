import timeit
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specification of directories
base_path = path.dirname(__file__)  # Location of the main.py
data_path = path.abspath(path.join(base_path, 'data'))  # The path where the data is stored

""" User Settings """
# Data specification
filename = 'AWM_5vars.csv'
output = 'AWM_5vars_conv.csv'
vars = 5

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
y_data_conv[:, 3] = y_data[1:, 3]
y_data_conv[:, 4] = [(np.log(y_data[i+1, 4])-np.log(y_data[i, 4])) for i in range(y_data.shape[0]-1)]

for i in range(vars):
    plt.plot(y_data_conv[:,i])
    plt.show()

corr_conv = np.corrcoef(y_data_conv.T)
print(corr_conv)

new_data = pd.DataFrame(y_data_conv)
new_data.to_csv(output)
