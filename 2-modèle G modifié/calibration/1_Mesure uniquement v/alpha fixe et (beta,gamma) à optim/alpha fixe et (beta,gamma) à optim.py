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



#on s'attaque à la matrice de covariance
with open('Pantheon+SH0ES_STAT+SYS.txt') as file:
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




#on définit la fonction qui calcule la likelihood
def likelihood_func(gamma,beta,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for i in range(len(gamma)):
        likelihood.append([])
    for j in range(len(gamma)):
        for k in range(len(beta)):
            DeltaD = np.empty(1701)                            
            for i in range(1701):
                mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                def f(x):
                    return (1/(73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) #🔴calcul de la distance lumineuse avec les paramètres cosmologiques FlatLambdaCDM dans Brout et al. 2022 = Analysis on cosmological constraints
                result = spi.quad(f,0,zHD[i])                                               
                mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

                if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                    mu = mu_shoes + (5/2)*gamma[j][j]*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))/(beta[k][k])))/(1+0.18)) - mu_theory
                    DeltaD[i]=mu
                else :
                    mu = mu_shoes + (5/2)*gamma[j][j]*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))/(beta[k][k])))/(1+0.18)) - mu_cepheid
                    DeltaD[i]=mu
            DeltaD_transpose = np.transpose(DeltaD)
                #on calcule la likelihood 
            A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
            likelihood[j].append(np.dot(A,DeltaD)) 
    return likelihood
#print(likelihood_func(0.309,0.691,matcov_SN_Cepheid_diag))





#maintenant on va tenter de tracer le diagramme (gamma_range,beta) en faisant varier ces paramètres et trouver le minimum de la likelihood
delta = 0.1
gamma_range = np.arange(-5, 3.1, delta) 
beta_range = np.arange(-3, 5.1, delta)                #gamma_range.copy().T 
gamma_range, beta_range = np.meshgrid(gamma_range, beta_range)
gamma_range = gamma_range.astype(float);  beta_range = beta_range.astype(float)
Chi2 = likelihood_func(gamma_range,beta_range,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)
#Chi2[0][10] = 0 on supprme un outlier (valeur à 10**8)
#Chi2[0][0] = 10000
print(Chi2) ; print(gamma_range, beta_range)



min = Chi2[0][0]
CI_1σ = []
CI_1σ_beta = []
CI_1σ_gamma = []
CI_2σ = []
for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min >= Chi2[i][j] and Chi2[i][j] != 0 and Chi2[i][j] != nan:
            min = Chi2[i][j] ; arg_min_gamma = gamma_range[i][j] ; arg_min_beta = beta_range[i][j]
print("gamma= ",arg_min_gamma,"; beta= ", arg_min_beta, "; min =", min)
             #/(len(gamma_range)-1)                 #/(len(beta_range)-1)


for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if  Chi2[i][j]<=min+6.17 and Chi2[i][j] != nan:
            CI_2σ.append([Chi2[i][j]])
            CI_2σ.append(gamma_range[i][j])
            CI_2σ.append(beta_range[i][j])


        if Chi2[i][j]<=min+2.3 and Chi2[i][j] != nan:
            CI_1σ.append([Chi2[i][j]])
            CI_1σ.append(gamma_range[i][j])
            CI_1σ.append(beta_range[i][j])
            CI_1σ_beta.append(beta_range[i][j])
            CI_1σ_gamma.append(gamma_range[i][j])


merge_sort(CI_1σ_gamma) ; merge_sort(CI_1σ_beta)


print("Cl_95 =", CI_2σ)
print("nb d'éléments CI_2σ =",len(CI_2σ)/3)
print("CI_1σ =", CI_1σ)
print("nb d'éléments CI_1σ =",len(CI_1σ)/3)
print("CI_1σ_beta =", CI_1σ_beta)
print("CI_1σ_gamma =", CI_1σ_gamma)



