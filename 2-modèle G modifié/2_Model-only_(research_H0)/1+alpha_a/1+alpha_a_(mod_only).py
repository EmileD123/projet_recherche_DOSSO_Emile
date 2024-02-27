from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as spi
import time
from time import time_ns
import os
file1 = '../Pantheon+SH0ES_STAT+SYS.txt'
file2 = '../Pantheon+Shoes data.txt'


#obtention of covariance matrix
with open(file1) as file:
    data = [line.strip() for line in file]
data = np.array(data) ; data = data.astype(float)
matcov_SN_Cepheid = [data[i:i+1701] for i in range(0, 1701**2, 1701)] #peut importe si on range ligne par ligne ou colonne par colonne car la matrice de covariance est symétrique
matcov_SN_Cepheid_1ere_moitie = [data[i:i+849] for i in range(0, 849**2, 849)]
matcov_SN_Cepheid_2nde_moitie = [data[851+i:850+i+850] for i in range(0, 849**2, 849)]



#obtention of redshift and modulus distances
zHD = []
CEPH_DIST = []      #distances determined thanks to Cepheids anchors
MU_SHOES = []
with open(file2) as file:
    for line in file:
        columns = line.strip().split(' ')
        if len(columns) >= 2:
            zHD.append(columns[2])
            CEPH_DIST.append(columns[12])
            MU_SHOES.append(columns[10])
CEPH_DIST.pop(0); zHD.pop(0); MU_SHOES.pop(0)
CEPH_DIST = np.array(CEPH_DIST); zHD = np.array(zHD); MU_SHOES = np.array(MU_SHOES)
CEPH_DIST = CEPH_DIST.astype(float); zHD = zHD.astype(float); MU_SHOES = MU_SHOES.astype(float)

#we define the function which computes the likelihood
def likelihood_func(H0,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for j in range(len(H0)):
        DeltaD = np.empty(len(mat_cov))
        for i in range(len(mat_cov)):
            mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
            def f(x):
                return (1/(((1+0.18*(1/(1+x)))**(1/2))*H0[j]*(((0.334)*((1+x)**3)+(0.666))**(1/2))))*(3*(10**5))*(10**6)  
                #↑we use the value of OmegaM = 0.334 optimized for the flat ΛCDM ; alpha = 0.18
            result = spi.quad(f,0,zHD[i])                                           
            mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)


            if CEPH_DIST[i] == -9.0 : #we check whether the measurement is related to the measurement of a distance with a Cepheid (CEPH_DIST[i] == -9.0 means that this is not the case)
                mu = mu_shoes - mu_theory 
                DeltaD[i]=mu
            else :
                mu = mu_shoes - mu_cepheid 
                DeltaD[i]=mu
        DeltaD_transpose = np.transpose(DeltaD)
        A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
        likelihood.append(np.dot(A,DeltaD)) 
    return likelihood


delta = 0.001
H0 = np.arange(67.7, 68.101, delta)                
H0 = H0.astype(float)
tps1 = time_ns()/1e9
Chi2 = likelihood_func(H0,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)     
tps2 = time_ns()/1e9
print("duration of computation for Chi2 = ", tps2 - tps1, " s")
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)

min = Chi2[0] 
CI_1σ = [] ; CI_2σ= [] 

for i in range(len(Chi2)) :
    if min >= Chi2[i] and Chi2[i]!= 0 and Chi2[i]!= nan:
        min = Chi2[i] ; arg_min_H0 = H0[i]
print("min =", min)
print("H0= ",arg_min_H0)


for i in range(len(Chi2)) :
    if min <= Chi2[i] and Chi2[i]<=min+4 and Chi2[i]!= nan: 
        CI_2σ.append(H0[i])

    if min <= Chi2[i] and Chi2[i]<=min+1 and Chi2[i] != nan:
        CI_1σ.append(H0[i])
        
print("gamma - 1σ = ",CI_1σ[0], "; gamma + 1σ = ",CI_1σ[-1])
#print("CI_2σ=", CI_2σ)
#print("nb d'éléments CI_2σ=",len(CI_2σ)/2)
#print("CI_1σ =", CI_1σ)
#print("nb d'éléments CI_1σ =",len(CI_1σ)/2)


 
