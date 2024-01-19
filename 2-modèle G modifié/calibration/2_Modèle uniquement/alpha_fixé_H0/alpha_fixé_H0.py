from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi

file1 = '../Pantheon+SH0ES_STAT+SYS.txt'
file2 = '../Pantheon+Shoes data.txt'

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
def likelihood_func(H0,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for j in range(len(H0)):
        DeltaD = np.empty(1701)                            
        for i in range(1701):
            mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
            def f(x):
                return (1/(((1+0.192*(1/(1+x)))**(1/2))*H0[j]*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) # alpha = ((H0riess/H0planck)^2)-1 ≈ 0.18 ; alpha_brout_f_lcdm = ((H0brout_f_lcdm /H0planck)^2)-1 ≈ 0.192 
            #🔴calcul de la distance lumineuse avec les paramètres cosmologiques FlatLambdaCDM dans Brout et al. 2022 = Analysis on cosmological constraints
            result = spi.quad(f,0,zHD[i])                                               
            mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

            if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                    mu = mu_shoes-mu_theory
                    DeltaD[i]=mu
            else :
                mu = mu_shoes-mu_cepheid
                DeltaD[i]=mu
                #on calcule la transposée
        DeltaD_transpose = np.transpose(DeltaD)
        #on calcule la likelihood 
        A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
        likelihood.append(np.dot(A,DeltaD)) 
    return likelihood





delta = 0.001
H0 = np.arange(66.5, 74.5, delta) 
H0 = H0.astype(float)
Chi2 = likelihood_func(H0,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)




min = Chi2[0]
CL_68 = [] ; CL_95 = [] 

for i in range(len(Chi2)) :
    if min >= Chi2[i] and Chi2[i]!= 0 and Chi2[i]!= nan:
        min = Chi2[i] ; arg_min_H0 = H0[i]
print("H0= ",arg_min_H0,"; min =", min)
             #/(len(H0)-1)                 #/(len(H0)-1)


for i in range(len(Chi2)) :
    if min <= Chi2[i] and Chi2[i]<=min+4 and Chi2[i]!= nan:
        CL_95.append(H0[i])
        CL_95.append([Chi2[i]])

    if min <= Chi2[i] and Chi2[i]<=min+1 and Chi2[i] != nan:
        CL_68.append(H0[i])
        CL_68.append([Chi2[i]])
        

print("Cl_95 =", CL_95)
print("nb d'éléments CL_95 =",len(CL_95)/2)
print("CL_68 =", CL_68)
print("nb d'éléments CL_68 =",len(CL_68)/2)


 
