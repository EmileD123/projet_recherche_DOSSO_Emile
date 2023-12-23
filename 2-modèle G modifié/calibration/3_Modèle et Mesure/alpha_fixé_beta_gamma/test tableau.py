from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi

def test_func(gamma,beta) :
    likelihood=[]
    for i in range(len(gamma)):
        likelihood.append([])
    for j in range(len(gamma)):
        for k in range(len(beta)):
            calcul = gamma[j][j] - beta[k][k]  
            likelihood[j].append(calcul)
    return likelihood

delta = 0.5
gamma_range = np.arange(-10,5.5, delta) 
beta_range = np.arange(-10, 5.5, delta)                #gamma_range.copy().T 
gamma_range, beta_range = np.meshgrid(gamma_range, beta_range,indexing='ij')
gamma_range = gamma_range.astype(float);  beta_range = beta_range.astype(float)
test = test_func(gamma_range,beta_range)

min = test[0][0]
for i in range(len(test)) :
    for j in range(len(test[i])) :
        if min >= test[i][j] and test[i][j] != 0 and test[i][j] != nan:
            min = test[i][j] ; arg_min_gamma = gamma_range[i][j] ; arg_min_beta = beta_range[i][j]


print("gammma range = \n",gamma_range,"beta range = \n", beta_range)
print("likelihood = \n",test)
print(test[4][8])
print(test[8][4])

print("gamma= ",arg_min_gamma,"; beta= ", arg_min_beta, "; min =", min)