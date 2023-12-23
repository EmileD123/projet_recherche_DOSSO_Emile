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
def likelihood_func(alpha,beta,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for i in range(len(alpha)):
        likelihood.append([])
    for j in range(len(alpha)):
        for k in range(len(beta)):
            DeltaD = np.empty(1701)                            
            for i in range(1701):
                mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                def f(x):
                    return (1/(((1+alpha[j][j]*np.exp(x/((1+x)*beta[k][k])))**(1/2))*(73.6)*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) #🔴calcul de la distance lumineuse avec les paramètres cosmologiques FlatLambdaCDM dans Brout et al. 2022 = Analysis on cosmological constraints
                result = spi.quad(f,0,zHD[i])                                               
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
#print(likelihood_func(0.309,0.691,matcov_SN_Cepheid_diag))





#maintenant on va tenter de tracer le diagramme (alpha_range,beta) en faisant varier ces paramètres et trouver le minimum de la likelihood
delta = 0.01
alpha_range = np.arange(-0.25, 0.26, delta) 
beta_range = np.arange(-0.25, 0.26, delta)                #alpha_range.copy().T 
alpha_range, beta_range = np.meshgrid(alpha_range, beta_range)
alpha_range = alpha_range.astype(float);  beta_range = beta_range.astype(float)
Z = likelihood_func(alpha_range,beta_range,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Z = np.array(Z) ; Z = Z.astype(float)
#Z[0][10] = 0 on supprme un outlier (valeur à 10**8)
#Z[0][0] = 10000
print(Z) ; print(alpha_range, beta_range)



min = Z[0][0]
CL_68 = [] ; CL_95 = []; CL_68_beta = [] ; CL_68_alpha = []


for i in range(len(Z)) :
    for j in range(len(Z[i])) :
        if min >= Z[i][j] and Z[i][j] != 0 and Z[i][j] != nan:
            min = Z[i][j] ; arg_min_alpha = alpha_range[i][j] ; arg_min_beta = beta_range[i][j]
print("alpha= ",arg_min_alpha,"; beta= ", arg_min_beta, "; min =", min)
             #/(len(alpha_range)-1)                 #/(len(beta_range)-1)


for i in range(len(Z)) :
    for j in range(len(Z[i])) :
        if min <= Z[i][j] and Z[i][j]<=min+6.17 and Z[i][j] != nan:
            CL_95.append([Z[i][j]])
            CL_95.append(alpha_range[i][j])
            CL_95.append(beta_range[i][j])

        if min <= Z[i][j] and Z[i][j]<=min+2.3 and Z[i][j] != nan:
            CL_68.append([Z[i][j]])
            CL_68.append(alpha_range[i][j])
            CL_68.append(beta_range[i][j])
            CL_68_alpha.append(alpha_range[i][j])
            CL_68_beta.append(beta_range[i][j])


merge_sort(CL_68_alpha) ; merge_sort(CL_68_beta)

print("Cl_95 =", CL_95)
print("nb d'éléments CL_95 =",len(CL_95)/3)
print("CL_68 =", CL_68)
print("nb d'éléments CL_68 =",len(CL_68)/3)
print("CL_68_alpha =", CL_68_alpha)
print("CL_68_beta =",CL_68_beta)






fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation ='bilinear',
               origin ='lower',
               cmap ="bone",extent=(-0.25,0.26,-0.25,0.26)) 
  
levels = [min+2.3,min+6.7]
CS = ax.contour(Z, levels, 
                origin ='lower',
                cmap ='Greens',
                linewidths = 2,extent=(-0.25,0.26,-0.25,0.26))

ax.set_xlabel('beta', fontsize=12)  
ax.set_ylabel('alpha', fontsize=12)

ax.clabel(CS, levels,
          inline = 1, 
          fmt ='% 1.1f',
          fontsize = 9)
  



plt.show()
 
