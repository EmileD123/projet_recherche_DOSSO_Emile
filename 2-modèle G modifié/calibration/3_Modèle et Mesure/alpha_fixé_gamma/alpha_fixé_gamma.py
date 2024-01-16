from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi


file1_path = '../../../Pantheon+SH0ES_STAT+SYS.txt'
file2_path = '../../../Pantheon+Shoes data.txt'

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
def likelihood_func(gamma,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for j in range(len(gamma)):
        DeltaD = np.empty(1701)                            
        for i in range(1701):
            mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
            def f(x):
                return (1/(((1+0.18*(1/(1+x)))**(1/2))*73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) #🔴calcul de la distance lumineuse avec les paramètres cosmologiques (H0=73.5 pour le flat wCDM dans Brout et al. 2022 = Analysis on cosmological constraints)
            result = spi.quad(f,0,zHD[i])                                               #idem
            mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

            if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                mu = (mu_shoes + (5/2)*gamma[j]*np.log10((1+0.18*(1/(1+zHD[i])))/(1+0.18)))-mu_theory #alpha = 0.18
                DeltaD[i]=mu
            else :
                mu = (mu_shoes + (5/2)*gamma[j]*np.log10((1+0.18*(1/(1+zHD[i])))/(1+0.18)))-mu_cepheid
                DeltaD[i]=mu
                #on calcule la transposée
        DeltaD_transpose = np.transpose(DeltaD)
        #on calcule la likelihood 
        A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
        likelihood.append(np.dot(A,DeltaD)) 
    return likelihood

delta = 0.001
gamma = np.arange(4.5,5.5, delta) #gamma = np.arange(-0.1, 0.601, delta) 
gamma = gamma.astype(float)
Z = likelihood_func(gamma,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Z = np.array(Z) ; Z = Z.astype(float)
#Z[0][10] = 0 on supprime un outlier (valeur à 10**8)
#Z[0][0] = 10000
print(Z) ; print(gamma)




min = Z[0]
CL_68 = []
CL_95 = []
for i in range(len(Z)) :
    if min >= Z[i] and Z[i]!= 0 and Z[i]!= nan:
        min = Z[i] ; arg_min_gamma = gamma[i]
print("gamma= ",arg_min_gamma,"; min =", min)
             #/(len(gamma)-1)                 #/(len(gamma)-1)


for i in range(len(Z)) :
    if Z[i]<=min+3.841 and Z[i]!= nan:
        CL_95.append(gamma[i])
        CL_95.append([Z[i]])
        CL_95.append(i)
    if Z[i]<=min+1 and Z[i]!= nan:
        CL_68.append(gamma[i])
        CL_68.append([Z[i]])
        CL_68.append(i)
        

print("Cl_95 =", CL_95)
print("nb d'éléments CL_95 =",len(CL_95)/3)
print("CL_68 =", CL_68)
print("nb d'éléments CL_68 =",len(CL_68)/3)
"""
fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation ='bilinear',
               origin ='lower',
               cmap ="bone",extent=(gamma[0],gamma[len(gamma)-1],gamma[0],gamma[len(gamma)-1])) #marche pas -> à changer !
  
levels = [min+2.3,min+6.7]
CS = ax.contour(Z, levels, 
                origin ='lower',
                cmap ='Greens',
                linewidths = 2,extent=(gamma[0],gamma[len(gamma)-1],gamma[0],gamma[len(gamma)-1]))

ax.set_xlabel('OmegaM', fontsize=12)  
ax.set_ylabel('OmegaLambda', fontsize=12)

ax.clabel(CS, levels,
          inline = 1, 
          fmt ='% 1.1f',
          fontsize = 9)

plt.show()
"""