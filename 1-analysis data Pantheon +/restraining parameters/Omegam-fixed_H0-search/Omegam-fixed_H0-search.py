from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi
from time import time

file1 = '..\Pantheon+SH0ES_STAT+SYS.txt'
file2 = '..\Pantheon+Shoes data.txt'

#we draw the confidence contours
with open(file1) as file:
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
with open(file2) as file:
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
def likelihood_func(H0,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for j in range(len(H0)):
        DeltaD = np.empty(1701)                            
        for i in range(1701):
            mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
            def f(x):
                return (1/(H0[j]*((0.334*((1+x)**3)+(0.666))**(1/2))))*(3*(10**5))*(10**6) #🔴H0 divisé par 100 ici ! ; calcul de la distance lumineuse avec les paramètres cosmologiques (OmegaLambda correspondant au flat ΛCDM dans Brout et al. 2022 = Analysis on cosmological constraints)
            result = spi.quad(f,0,zHD[i])                                               #idem
            mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

            if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                mu = mu_shoes-mu_theory
                DeltaD[i]=mu
            else :
                mu = mu_shoes-mu_cepheid
                DeltaD[i]=mu
                #print(DeltaD)
                #on dispose du deltaD de la formule de la likelihood (formule (14) Brout et al. 2022) -> on peut essayer de voir si le résultat n'est pas aberrant en affichant les distance residuals en fonction du redshift et en comparant avec la fig 4 (il faudrait trouver un autre moyen de vérifier)
                """
                plt.scatter(zHD,DeltaD)
                plt.xscale('log')
                plt.ylabel('modulus distance residuals')
                plt.xlabel('redshift')
                plt.legend()
                plt.show()
                """
        #on calcule la transposée
        DeltaD_transpose = np.transpose(DeltaD)
        #on calcule la likelihood 
        A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
        likelihood.append(np.dot(A,DeltaD)) 
    return likelihood






#Now we're going to try to draw the (Omegam,Omegalambda) diagram by varying these parameters and find the minimum likelihood
delta = 0.1
H0 = np.arange(70,75.1, delta) 
H0 = H0.astype(float);  
tps1 = time()
Chi2 = likelihood_func(H0,matcov_SN_Cepheid_diag,zHD,CEPH_DIST,MU_SHOES)
tps2 = time()
print("temps de calcul Chi2 = ", tps2-tps1)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)

#we search the optimized parameter

min = Chi2[0]
CI_1σ = [] ; CI_2σ = []

for i in range(len(Chi2)) :
    if min >= Chi2[i] and Chi2[i] != 0 and Chi2[i] != nan:
        min = Chi2[i] ; arg_min_H0 = H0[i] 
print("H0= ", arg_min_H0, "; min =", min)



for i in range(len(Chi2)) :
        if min <= Chi2[i] and Chi2[i]<=min+3.841 and Chi2[i] != nan:
            CI_2σ.append([Chi2[i]]);CI_2σ.append(H0[i])
            

        if min <= Chi2[i] and Chi2[i]<=min+1 and Chi2[i] != nan:
            CI_1σ.append(Chi2[i]);CI_1σ.append(H0[i])
            



print("CI_2σ =", CI_2σ)
print("nb d'éléments CI_2σ =",len(CI_2σ)/2)
print("CI_1σ =", CI_1σ)
print("nb d'éléments CI_1σ =",len(CI_1σ)/2)


plt.show()
 
