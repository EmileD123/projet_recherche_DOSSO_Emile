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
def likelihood_func(OmegaMatter,OmegaLambda,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for i in range(len(OmegaMatter)):
        likelihood.append([])
    for j in range(len(OmegaMatter)):
        for k in range(len(OmegaLambda)):
            DeltaD = np.empty(1701)                            
            for i in range(1701):
                mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                def f(x):
                    return (1/(73.6*((OmegaMatter[j][j]*((1+x)**3)+OmegaLambda[k][k])**(1/2))))*(3*(10**5))*(10**6) #🔴calcul de la distance lumineuse avec les paramètres cosmologiques (H0=73.6 pour le flat LCDM dans Brout et al. 2022 = Analysis on cosmological constraints)
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
            likelihood[j].append(np.dot(A,DeltaD)) 
    return likelihood
#print(likelihood_func(0.309,0.691,matcov_SN_Cepheid_diag))





#maintenant on va tenter de tracer le diagramme (Omegam,Omegalambda) en faisant varier ces paramètres et trouver le minimum de la likelihood
delta = 0.001
OmegaM = np.arange(0.25, 0.451, delta) 
OmegaL = np.arange(0.55, 0.751, delta)              #OmegaM.copy().T 
OmegaL, OmegaM = np.meshgrid(OmegaL, OmegaM,indexing='ij')
OmegaM = OmegaM.astype(float);  OmegaL = OmegaL.astype(float)
Chi2 = likelihood_func(OmegaM,OmegaL,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)
#Chi2[0][10] = 0 on supprme un outlier (valeur à 10**8)
#Chi2[0][0] = 10000
print(Chi2) ; print(OmegaM, OmegaL)



min = Chi2[0][1]; indexOm = 0 ; indexOl = 0
CI_1σ= []
CI_1σ_Omega_M = []
CI_1σ_Omega_L = []
CI_2σ = []

for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min >= Chi2[i][j] and Chi2[i][j] != 0 and Chi2[i][j] != nan:
            min = Chi2[i][j] ; arg_min_Om = OmegaM[i][j] ; arg_min_Ol = OmegaL[i][j]
print("Om= ",arg_min_Om,"; Ol= ", arg_min_Ol, "; min =", min)
             #/(len(OmegaM)-1)                 #/(len(OmegaL)-1)



for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min <= Chi2[i][j] and Chi2[i][j]<=min+6.17 and Chi2[i][j] != nan:
            CI_2σ.append(OmegaM[i][j])
            CI_2σ.append(OmegaL[i][j])
            CI_2σ.append([Chi2[i][j]])

        if min <= Chi2[i][j] and Chi2[i][j]<=min+2.3 and Chi2[i][j] != nan:
            CI_1σ_Omega_M.append(OmegaM[i][j]);CI_1σ_Omega_L.append(OmegaL[i][j])
            CI_1σ.append(OmegaM[i][j]);CI_1σ.append(OmegaL[i][j])
            CI_1σ.append([Chi2[i][j]])
   
merge_sort(CI_1σ_Omega_L);merge_sort(CI_1σ_Omega_M)

print("Cl_95 =", CI_2σ)
print("nb d'éléments CI_2σ =",len(CI_2σ)/3)
print("CI_1σ=", CI_1σ)
print("nb d'éléments CI_1σ=",len(CI_1σ)/3)
print("CI_1σ_Omega_M=", CI_1σ_Omega_M)
print("CI_1σ_Omega_L=", CI_1σ_Omega_L)



fig, ax = plt.subplots()
im = ax.imshow(Chi2, interpolation ='bilinear',
               origin ='lower',
               cmap ="bone",extent=(OmegaM[0][0],OmegaM[len(OmegaM)-1][len(OmegaM)-1],OmegaL[0][0],OmegaL[len(OmegaL)-1][len(OmegaL)-1])) #marche pas -> à changer !
  
levels = [min+2.3,min+6.7]
CS = ax.contour(Chi2, levels, 
                origin ='lower',
                cmap ='Greens',
                linewidths = 2,extent=(OmegaM[0][0],OmegaM[len(OmegaM)-1][len(OmegaM)-1],OmegaL[0][0],OmegaL[len(OmegaL)-1][len(OmegaL)-1]))

ax.set_xlabel('OmegaM', fontsize=12)  
ax.set_ylabel('OmegaLambda', fontsize=12)

ax.clabel(CS, levels,
          inline = 1, 
          fmt ='% 1.1f',
          fontsize = 9)
  


"""
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the viewing angle (adjust these values as needed)
ax.view_init(elev=20, azim=45)  # Change elev and azim to set the view angle

# Plot the 3D surface
surface = ax.plot_surface(OmegaM, OmegaL, Chi2, cmap='viridis')

# Add color bar to show values
fig.colorbar(surface, shrink=0.5, aspect=5)

# Label axes (optional)
ax.set_xlabel('OmegaM')
ax.set_ylabel('OmegaL')
ax.set_zlabel('Likelihood')


CS1 = plt.contour(OmegaM, OmegaL, Chi2)
   
fmt = {}
strs = ['1', '2', '3', '4', '5', '6', '7']
for l, s in zip(CS1.levels, strs):
    fmt[l] = s
plt.clabel(CS1, CS1.levels, inline = True,
           fmt = fmt, fontsize = 10)
  
plt.title('matplotlib.pyplot.contour()')

"""


plt.show()
 
