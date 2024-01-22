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


file1 = '..\Pantheon+SH0ES_STAT+SYS.txt'
file2 = '..\Pantheon+Shoes data.txt'

#on s'attaque à la matrice de covariance
with open(file1) as file:
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




#on définit la fonction qui calcule la likelihood
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
                    return (1/(((1+0.18*(x/(1+x)))**(1/2))*H0[k][k]*((((OmegaMatter[j][j])/100)*((1+x)**3)+(1-((OmegaMatter[j][j])/100)))**(1/2))))*(3*(10**5))*(10**6) #🔴Omegam divisé par 100 ici ! ; calcul de la distance lumineuse avec les paramètres cosmologiques (OmegaLambda correspondant au flat ΛCDM dans Brout et al. 2022 = Analysis on cosmological constraints)
                result = spi.quad(f,0,zHD[i])                                               #idem
                mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

                if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                    mu = mu_shoes-mu_theory
                    DeltaD[i]=mu
                else :
                    mu = mu_shoes-mu_cepheid
                    DeltaD[i]=mu
            DeltaD_transpose = np.transpose(DeltaD)
                #on calcule la likelihood 
            A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
            likelihood[j].append(np.dot(A,DeltaD)) 
    return likelihood






#maintenant on va tenter de tracer le diagramme (Omegam,Omegalambda) en faisant varier ces paramètres et trouver le minimum de la likelihood
delta = 0.1
OmegaM = np.arange(25, 35.1, delta) ; OmegaM = OmegaM[:-1]
H0 = np.arange(65, 75.1, delta)                #OmegaM.copy().T 
OmegaM, H0 = np.meshgrid(OmegaM, H0,indexing='ij')
OmegaM = OmegaM.astype(float);  H0 = H0.astype(float)
Chi2 = likelihood_func(OmegaM,H0,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)  
#Chi2[0][10] = 0 on supprme un outlier (valeur à 10**8)
print(Chi2)



min = Chi2[0][1]
CI_1σ = [] ; CI_2σ = [] ; CI_1σ_Omegam = []; CI_1σ_H0 = []

for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min >= Chi2[i][j] and Chi2[i][j] != 0 and Chi2[i][j] != nan:
            min = Chi2[i][j] ; arg_min_Om = OmegaM[i][j] ; arg_min_H0 =H0[i][j]
print("Om= ", arg_min_Om/100, "; H0= ", arg_min_H0, "; min =", min)


for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min <= Chi2[i][j] and Chi2[i][j]<=min+6.17 and Chi2[i][j] != nan:
            CI_2σ.append([Chi2[i][j]]);CI_2σ.append(OmegaM[i][j]/100);CI_2σ.append(H0[i][j])
            

        if min <= Chi2[i][j] and Chi2[i][j]<=min+2.3 and Chi2[i][j] != nan:
            CI_1σ.append([Chi2[i][j]]);CI_1σ.append(OmegaM[i][j]/100);CI_1σ.append(H0[i][j])
            CI_1σ_Omegam.append(OmegaM[i][j]/100)
            CI_1σ_H0.append(H0[i][j])

merge_sort(CI_1σ_Omegam);merge_sort(CI_1σ_H0)

print("CI_2σ =", CI_2σ)
print("nb d'éléments CI_2σ =",len(CI_2σ)/3)
print("CI_1σ =", CI_1σ)
print("nb d'éléments CI_1σ =",len(CI_1σ)/3)
print("CI_1σ_Omegam =",CI_1σ_Omegam)
print("CI_1σ_H0=",CI_1σ_H0)

 
