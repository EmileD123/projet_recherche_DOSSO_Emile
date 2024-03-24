from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi
from merge_sort import merge_sort as merge_sort
from time import time

file1 = '..\Pantheon+SH0ES_STAT+SYS.txt'
file2 = '..\Pantheon+Shoes data.txt'

#obtention of redshift and modulus distances
with open(file1) as file:
    data = [line.strip() for line in file]
data = np.array(data) ; data = data.astype(float)
matcov_SN_Cepheid = [data[i:i+1701] for i in range(0, 1701**2, 1701)] 
matcov_SN_Cepheid_diag = np.zeros((1701,1701))
for i in range(1701):
    matcov_SN_Cepheid_diag[i][i] = matcov_SN_Cepheid[i][i]




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
def likelihood_func(alpha,gamma,H0,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for m in range(len(gamma)):
        likelihood.append([])
    for j in range(len(gamma)):
        for k in range(len(H0)):                
            DeltaD = np.empty(1701)                            
            for i in range(1701):
                mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                def f(x):
                    return (1/(((1+alpha*np.exp(-10*(1-(1/(1+x)))/(1/(1+x))))**(1/2))*H0[k][k]*(((0.333)*((1+x)**3)+(0.667))**(1/2))))*(3*(10**5))*(10**6)
                    #↑we use the value of OmegaM = 0.333 optimized for the flat ΛCDM
                result = spi.quad(f,0,zHD[i])
                mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                    

                if CEPH_DIST[i] == -9.0 : #we check whether the measurement is related to the measurement of a distance with a Cepheid (CEPH_DIST[i] == -9.0 means that this is not the case)
                    mu = (mu_shoes + (5/2)*gamma[j][j]*np.log10((1+alpha*np.exp(-10*(1-(1/(1+zHD[i])))/(1/(1+zHD[i]))))/(1+alpha)))-mu_theory #alpha = 0.18 ; alpha_brout = 0.192
                    DeltaD[i]=mu 
                else :
                    mu = (mu_shoes + (5/2)*gamma[j][j]*np.log10((1+alpha*np.exp(-10*(1-(1/(1+zHD[i])))/(1/(1+zHD[i]))))/(1+alpha)))-mu_cepheid #alpha = 0.18 ; alpha_brout = 0.192
                    DeltaD[i]=mu
            DeltaD_transpose = np.transpose(DeltaD)
            A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
            likelihood[j].append(np.dot(A,DeltaD)) 
    return likelihood





#Set-up entries of likelihood_func
alpha_riess = 0.18 ; alpha_brout_f_lcdm = 0.192
delta = 0.01
gamma = np.arange(-1.9,0,delta) 
H0 = np.arange(66.0, 67.9, delta) ; H0 = H0[:-1]               
gamma , H0 = np.meshgrid(gamma, H0,indexing='ij')
gamma = gamma.astype(float);  H0 = H0.astype(float)
print(gamma);print(H0)
tps1 = time()
Chi2 = likelihood_func(alpha_riess ,gamma,H0,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
tps2 = time()
print("temps de calcul Chi2 = ", tps2-tps1)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)  
print(Chi2)

np.save('last_Chi2_save',Chi2)


#we search the optimized parameter


min = Chi2[0][1]
CI_1σ = [] ; CI_2σ = [] ; CI_1σ_gamma = []; CI_1σ_H0 = []

for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min >= Chi2[i][j] and Chi2[i][j] != 0 and Chi2[i][j] != nan:
            min = Chi2[i][j] ; arg_min_gamma = gamma[i][j] ; arg_min_H0 =H0[i][j]
print("gamma= ", arg_min_gamma, "; H0= ", arg_min_H0, "; min =", min)


for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        #if min <= Chi2[i][j] and Chi2[i][j]<=min+6.17 and Chi2[i][j] != nan:
        #    CI_2σ.append([Chi2[i][j]]);CI_2σ.append(gamma[i][j]);CI_2σ.append(H0[i][j])
            

        if min <= Chi2[i][j] and Chi2[i][j]<=min+2.3 and Chi2[i][j] != nan:
           # CI_1σ.append([Chi2[i][j]]);CI_1σ.append(gamma[i][j]);CI_1σ.append(H0[i][j])
            CI_1σ_gamma.append(gamma[i][j])
            CI_1σ_H0.append(H0[i][j])

merge_sort(CI_1σ_gamma);merge_sort(CI_1σ_H0)

#print("CI_2σ =", CI_2σ)
#print("nb d'éléments CI_2σ =",len(CI_2σ)/3)
#print("CI_1σ =", CI_1σ)
#print("nb d'éléments CI_1σ =",len(CI_1σ)/3)
print("gamma-1σ =",CI_1σ_gamma[0]) ; print("; gamma+1σ =",CI_1σ_gamma[-1])#print("CI_1σ_gamma =",CI_1σ_gamma)
print("H0-1σ =",CI_1σ_H0[0]) ; print("; H0+1σ =",CI_1σ_H0[-1])#print("CI_1σ_H0=",CI_1σ_H0)


#we draw the confidence contours

 
fig, ax = plt.subplots()
im = ax.imshow(Chi2, interpolation ='bilinear',
               origin ='lower',
               cmap ="bone",extent=(66.0,67.9,-1.9,0.0))
  
levels = [min+2.3,min+6.7]
CS = ax.contour(Chi2, levels, 
                origin ='lower',
                cmap ='Greens',
                linewidths = 2,extent=(66.0,67.9,-1.9,0.0))

ax.set_xlabel('H0', fontsize=12)  
ax.set_ylabel('gamma', fontsize=12)   
ax.clabel(CS, levels,
          inline = 1, 
          fmt ='% 1.1f',
          fontsize = 9)

plt.savefig('confidence_contour_H0_gamma_[66.0;67.9]_[-1.9;0.0].jpg')


plt.show()



