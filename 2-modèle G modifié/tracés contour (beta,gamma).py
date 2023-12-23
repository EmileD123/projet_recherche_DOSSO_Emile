from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi


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

#introduction α ; param dans G/Gn = 1+α*exp((1-a)/beta)
H0riess=73.0 ;H0planck=67.4
alpha = ((H0riess/H0planck)**2)-1


#on définit la fonction qui calcule la likelihood
def likelihood_func(beta,gamma,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for i in range(len(beta)):
        likelihood.append([])
    for j in range(len(beta)):
        for k in range(len(gamma)):
            DeltaD = np.empty(1701)                            
            for i in range(1701):
                mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                def f(x):
                    return 1/(H0planck*((1+(((H0riess/H0planck)**2)-1)*np.exp((x/(1+x))/beta[j][j]))**(1/2)))*(3*(10**5))*(10**6) #🔴calcul de la distance lumineuse modifié en prenant H0planck pour calibrer (donne H(z=0)=H0riess) -> Sûr?
                result = spi.quad(f,0,zHD[i])                                               #idem
                mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

                if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                    mu = (mu_shoes+(5/2)*gamma[k][k]*np.log10(1+alpha*np.exp((zHD[i]/(1+zHD[i]))/beta[j][j])))-mu_theory
                    DeltaD[i]=mu
                else :
                    mu = (mu_shoes+(5/2)*gamma[k][k]*np.log10(1+alpha*np.exp((zHD[i]/(1+zHD[i]))/beta[j][j])))-mu_cepheid
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
            likelihood[j].append(np.dot(A,DeltaD)) 
    return likelihood

#maintenant on va tenter de tracer le diagramme (beta,gamma) en faisant varier ces paramètres et trouver le minimum de la likelihood
delta = 0.01
beta = np.arange(2.25, 2.76, delta) 
gamma = np.arange(0.5, 1.01, delta)                #beta.copy().T 
beta, gamma = np.meshgrid(beta, gamma)
beta = beta.astype(float);  gamma = gamma.astype(float)
Z = likelihood_func(beta,gamma,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Z = np.array(Z) ; Z = Z.astype(float)
#Z[0][10] = 0 on supprme un outlier (valeur à 10**8)
#Z[0][0] = 10000
print(Z) ; print(beta, gamma)


min = Z[1][1]
CL_68 = []
CL_95 = []
for i in range(len(Z)) :
    for j in range(len(Z[i])) :
        if min >= Z[i][j] and Z[i][j] != 0 and Z[i][j] != nan :
            min = Z[i][j] ; arg_min_beta = beta[i][i] ; arg_min_gamma = gamma[j][j]
print("beta= ",arg_min_beta,"; gamma= ", arg_min_gamma, "; min =", min)

for i in range(len(Z)) :
    for j in range(len(Z[i])) :
        if min <= Z[i][j] and Z[i][j]<=min+6.17 and Z[i][j] != nan:
            CL_95.append([Z[i][j]])
            CL_95.append(i)
            CL_95.append(j)
        if min <= Z[i][j] and Z[i][j]<=min+2.3 and Z[i][j] != nan:
            CL_68.append([Z[i][j]])
            CL_68.append(i)
            CL_68.append(j)

print("Cl_95 =", CL_95)
print("nb d'éléments CL_95 =",len(CL_95)/3)
print("CL_68 =", CL_68)
print("nb d'éléments CL_68 =",len(CL_68)/3)





fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation ='bilinear',
               origin ='lower',
               cmap ="bone",extent=(0.5,1,2.25,2.75)) 

levels = [min+2.3,min+6.7]
CS = ax.contour(Z, levels, 
                origin ='lower',
                cmap ='Greens',
                linewidths = 2,extent=(0.5,1,2.25,2.75))

ax.set_xlabel('gamma', fontsize=12)  
ax.set_ylabel('beta', fontsize=12)  #si on échange pas les axes -> sinon mauvaise position du minimum... : explication ? (même propblème que pour les autres contours)

ax.clabel(CS, levels,
          inline = 1, 
          fmt ='% 1.1f',
          fontsize = 9)

plt.show()