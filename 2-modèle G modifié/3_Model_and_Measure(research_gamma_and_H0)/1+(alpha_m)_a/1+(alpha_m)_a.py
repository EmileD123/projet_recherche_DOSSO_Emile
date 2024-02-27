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



file1_path = '../Pantheon+SH0ES_STAT+SYS.txt'
file2_path = '../Pantheon+Shoes data.txt'

#on s'attaque à la matrice de covariance
with open(file1_path) as file:
    data = [line.strip() for line in file]
data = np.array(data) ; data = data.astype(float)
matcov_SN_Cepheid = [data[i:i+1701] for i in range(0, 1701**2, 1701)] #peut importe si on range ligne par ligne ou colonne par colonne car la matrice de covariance est symétrique
matcov_SN_Cepheid_diag = np.zeros((1701,1701))
for i in range(1701):
    matcov_SN_Cepheid_diag[i][i] = matcov_SN_Cepheid[i][i]




zHD = []
CEPH_DIST = []      #les distances calculées à l'aide de la présence de Cepheids
MU_SHOES = []
# Open the text file for reading
with open(file2_path) as file:
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

#on définit la fonction qui calcule la likelihood
def likelihood_func(gamma,H0,m,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for i in range(len(H0)):
        likelihood.append([])
        for j in range(len(m)):
            likelihood[i].append([])

    for j in range(len(gamma)):
        for k in range(len(H0)):
            for l in range(len(m)):
                DeltaD = np.empty(1701)                            
                for i in range(1701):
                    mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                    def f(x):
                        return (1/(((1+((73.3/67.4)**(2/m[l][l][l])-1)*(1/(1+x)))**(m[l][l][l]/2))*H0[k][k][k]*(((0.334)*((1+x)**3)+(0.666))**(1/2))))*(3*(10**5))*(10**6) #alpha_m = ((H0riess/H0planck)**(2/m)-1) = ((73.3/67.4)**(2/m)-1)
                    result = spi.quad(f,0,zHD[i])                                               #idem
                    mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

                    if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                        mu = (mu_shoes + (5/2)*m[l][l][l]*gamma[j][j][j]*np.log10((1+((73.3/67.4)**(2/m[l][l][l])-1)*(1/(1+zHD[i])))/(1+((73.3/67.4)**(2/m[l][l][l])-1))))-mu_theory #alpha_m = ((H0riess/H0planck)**(2/m)-1) = ((73.3/67.4)**(2/m)-1)
                        DeltaD[i]=mu
                    else :
                        mu = (mu_shoes + (5/2)*m[l][l][l]*gamma[j][j][j]*np.log10((1+((73.3/67.4)**(2/m[l][l][l])-1)*(1/(1+zHD[i])))/(1+((73.3/67.4)**(2/m[l][l][l])-1))))-mu_cepheid
                        DeltaD[i]=mu
                    #on calcule la transposée
                    DeltaD_transpose = np.transpose(DeltaD)
                    #on calcule la likelihood 
                A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
                likelihood[k][l].append(np.dot(A,DeltaD)) 
    return likelihood

delta = 2
gamma = np.arange(-10,11, delta) ; H0 = np.arange(65,76,delta/2) ; m = np.arange(1,12,delta/2) #gamma = np.arange(-0.1, 0.601, delta) 
gamma, H0, m = np.meshgrid(gamma,H0,m,indexing='ij')
#print(gamma);print(H0);print(m)
gamma = gamma.astype(float) ; H0 = H0.astype(float) ; m = m.astype(float)
Chi2 = likelihood_func(gamma,H0,m,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2)
Chi2 = Chi2.astype(float)
print("matrix result Chi2 = ",Chi2)



min = Chi2[0][0][0]
CI_1σ = []
CI_2σ = []
CI_1σ_gamma = [] ; CI_1σ_H0 = [] ; CI_1σ_m = []
for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        for k in range(len(Chi2[i][j])) :
            if min >= Chi2[i][j][k] and Chi2[i][j][k]!= 0 and Chi2[i][j][k]!= nan:
                min = Chi2[i][j][k] ; arg_min_gamma = gamma[i][j][k] ; arg_min_H0 = H0[i][j][k] ; arg_min_m = m[i][j][k]
print("gamma= ",arg_min_gamma,"; H0= ",arg_min_H0,"; m= ",arg_min_m,"; min =", min)



for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        for k in range(len(Chi2[i][j])) :
            if Chi2[i][j][k]<=min+8.02 and Chi2[i][j][k]!= nan :
                CI_2σ.append(gamma[i][j][k]) ; CI_2σ.append(H0[i][j][k]) ; CI_2σ.append(m[i][j][k])
                #CI_2σ.append([Chi2[i]])
            if Chi2[i][j][k]<=min+3.5 and Chi2[i][j][k]!= nan:
                CI_1σ.append(gamma[i][j][k]) ; CI_1σ.append(H0[i][j][k]) ; CI_1σ.append(m[i][j][k])
                #CI_1σ.append([Chi2[i]])
                CI_1σ_gamma.append(gamma[i][j][k]);CI_1σ_H0.append(H0[i][j][k]);CI_1σ_m.append(m[i][j][k])
        

merge_sort(CI_1σ_gamma);merge_sort(CI_1σ_H0);merge_sort(CI_1σ_m)

print("gamma - 1σ = ",CI_1σ_gamma[0],"; gamma + 1σ = ",CI_1σ_gamma[-1])
print("m - 1σ = ",CI_1σ_m[0],"; m + 1σ = ",CI_1σ_m[-1])
print("H0 - 1σ = ",CI_1σ_H0[0],"; H0 + 1σ = ",CI_1σ_H0[-1])

#print("CI_2σ =", CI_2σ)
#print("nb d'éléments CI_2σ =",len(CI_2σ)/4)
#print("CI_1σ =", CI_1σ)
#print("nb d'éléments CI_1σ =",len(CI_1σ)/4)
#print('CI_1σ_gamma',CI_1σ_gamma)
#print('CI_1σ_H0',CI_1σ_H0)
#print('CI_1σ_m',CI_1σ_m)