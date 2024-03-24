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

#we tackle the covariance matrix
with open('Pantheon+SH0ES_STAT+SYS.txt') as file:
    data = [line.strip() for line in file]
data = np.array(data) ; data = data.astype(float)
matcov_SN_Cepheid = [data[i:i+1701] for i in range(0, 1701**2, 1701)] #it doesn't matter whether you arrange row by row or column by column, because the covariance matrix is symmetrical
matcov_SN_Cepheid_diag = np.zeros((1701,1701))
for i in range(1701):
    matcov_SN_Cepheid_diag[i][i] = matcov_SN_Cepheid[i][i]




zHD = []
CEPH_DIST = []      #distances calculated using the presence of Cepheids
MU_SHOES = []
# Open the text file for reading
with open('Pantheon+Shoes data.txt') as file:
    # Loop through each line in the file
    for line in file:
        # Split the line into columns based on a tab separator
        columns = line.strip().split(' ')
            
        # Check if there are at least two columns
        if len(columns) >= 2:
            # Append the values from the first and second columns to their respective lists
            zHD.append(columns[2])
            CEPH_DIST.append(columns[12])
            MU_SHOES.append(columns[10])
CEPH_DIST.pop(0); zHD.pop(0); MU_SHOES.pop(0)
CEPH_DIST = np.array(CEPH_DIST); zHD = np.array(zHD); MU_SHOES = np.array(MU_SHOES)
CEPH_DIST = CEPH_DIST.astype(float); zHD = zHD.astype(float); MU_SHOES = MU_SHOES.astype(float)




#we define the function that calculates the likelihood
def likelihood_func(OmegaMatter,H0,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[] 
    for i in range(len(OmegaMatter)):
        likelihood.append([])
    for j in range(len(OmegaMatter)):
        for k in range(len(H0)):
            DeltaD = np.empty(1701)                            
            for i in range(1701):
                mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                def f(x):
                    return (1/(H0[k][k]*(((OmegaMatter[j][j]/100)*((1+x)**3)+(1-(OmegaMatter[j][j]/100)))**(1/2))))*(3*(10**5))*(10**6)  #🔴Omegam divided by 100 here ! ; calculation of the luminous distance with cosmological parameters (OmegaLambda corresponding to the flat ΛCDM in Brout et al. 2022 = Analysis on cosmological constraints)
                result = spi.quad(f,0,zHD[i])                                               #idem
                mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

                if CEPH_DIST[i] == -9.0 : #we check whether the measurement is related to the measurement of a distance with a Cepheid (CEPH_DIST[i] == -9.0 means that this is not the case)
                    mu = mu_shoes-mu_theory
                    DeltaD[i]=mu
                else :
                    mu = mu_shoes-mu_cepheid
                    DeltaD[i]=mu
                #we calculate the transpose
            DeltaD_transpose = np.transpose(DeltaD)
                #we calculate the likelihood 
            A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
            likelihood[j].append(np.dot(A,DeltaD)) 
    return likelihood






#Now we're going to try to draw the (Omegam,Omegalambda) diagram by varying these parameters and find the minimum likelihood
delta = 0.05
OmegaM = np.arange(25, 38.1, delta) #; OmegaM = OmegaM[:-1]
H0 = np.arange(67, 80.1, delta)
OmegaM, H0 = np.meshgrid(OmegaM, H0,indexing='ij')
OmegaM = OmegaM.astype(float);  H0 = H0.astype(float)
tps1 = time()
Chi2 = likelihood_func(OmegaM,H0,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
tps2 = time()
print("time of computation of Chi2 = ", tps2 - tps1)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)
print(Chi2)


#we search the optimized parameter

min = Chi2[0][1]
CI_1σ = [] ; CI_2σ = [] ; CI_1σ_Omegam = []; CI_1σ_H0 = []

for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min >= Chi2[i][j] and Chi2[i][j] != 0 and Chi2[i][j] != nan:
            min = Chi2[i][j] ; arg_min_Om = OmegaM[i][j] ; arg_min_H0 = H0[i][j]
print("Om= ", arg_min_Om, "; H0= ", arg_min_H0, "; min =", min)
            


for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min <= Chi2[i][j] and Chi2[i][j]<=min+6.17 and Chi2[i][j] != nan:
            CI_2σ.append([Chi2[i][j]]);CI_2σ.append(OmegaM[i][j]);CI_2σ.append(H0[i][j])
            

        if min <= Chi2[i][j] and Chi2[i][j]<=min+2.3 and Chi2[i][j] != nan:
            CI_1σ.append([Chi2[i][j]]);CI_1σ.append(OmegaM[i][j]);CI_1σ.append(H0[i][j])
            CI_1σ_Omegam.append(OmegaM[i][j])
            CI_1σ_H0.append(H0[i][j])

merge_sort(CI_1σ_Omegam);merge_sort(CI_1σ_H0)


print("OmegaM-1σ =",CI_1σ_Omegam[0],"; OmegaM+1σ =",CI_1σ_Omegam[-1]) #print("CI_1σ_gamma =",CI_1σ_gamma)
print("H0-1σ =",CI_1σ_H0[0],"; H0+1σ =",CI_1σ_H0[-1]) #print("CI_1σ_H0=",CI_1σ_H0)


#we draw the confidence contours

fig, ax = plt.subplots()
im = ax.imshow(Chi2, interpolation ='bilinear',
               origin ='lower',
               cmap ="bone",extent=(65,80,25,40))
  
levels = [min+2.3,min+6.7]
CS = ax.contour(Chi2, levels, 
                origin ='lower',
                cmap ='Greens',
                linewidths = 2,extent=(65,80,25,40))

ax.set_xlabel('H0', fontsize=12)  
ax.set_ylabel('OmegaM', fontsize=12)   
ax.clabel(CS, levels,
          inline = 1, 
          fmt ='% 1.1f',
          fontsize = 9)
  


plt.show()
 
