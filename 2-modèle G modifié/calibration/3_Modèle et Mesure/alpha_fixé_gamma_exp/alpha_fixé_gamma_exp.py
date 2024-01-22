from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi

"""
Implémentation tiré de :
- alpha_fixé_gamma_exp dans le dossier "Mesure uniquement"
"""

file1_path = "../Pantheon+SH0ES_STAT+SYS.txt"
file2_path = "../Pantheon+Shoes data.txt"


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
                return (1/(((1+(0.18)*np.exp(x/((1+x))))**(1/2))*(73.6)*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) #🔴calcul de la distance lumineuse avec les paramètres cosmologiques FlatLambdaCDM dans Brout et al. 2022 = Analysis on cosmological constraints
            result = spi.quad(f,0,zHD[i])                                               #idem
            mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

            if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                mu = mu_shoes + (5/2)*(gamma[j])*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))))/(1+0.18)) - mu_theory #alpha = 0.18
                DeltaD[i]=mu
            else :
                mu = mu_shoes + (5/2)*(gamma[j])*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))))/(1+0.18)) - mu_cepheid
                DeltaD[i]=mu
                #on calcule la transposée
        DeltaD_transpose = np.transpose(DeltaD)
        #on calcule la likelihood 
        A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
        likelihood.append(np.dot(A,DeltaD)) 
    return likelihood
#print(likelihood_func(0.309,0.691,matcov_SN_Cepheid_diag))






delta = 0.001
gamma = np.arange(-5.45,-5.149,delta) 
gamma = gamma.astype(float)
Chi2 = likelihood_func(gamma,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)
#Chi2[0][10] = 0 on supprime un outlier (valeur à 10**8)
#Chi2[0][0] = 10000
print(Chi2) ; print(gamma)




min = Chi2[20]
CI_1σ = []
CI_2σ = []
for i in range(len(Chi2)) :
    if min >= Chi2[i] and Chi2[i]!= 0 and Chi2[i]!= nan:
        min = Chi2[i] ; arg_min_gamma = gamma[i]
print("gamma= ",arg_min_gamma,"; min =", min)
             #/(len(gamma)-1)                 #/(len(gamma)-1)


for i in range(len(Chi2)) :
    if Chi2[i]<=min+4 and Chi2[i]!= nan:
        CI_2σ.append(gamma[i])
        CI_2σ.append([Chi2[i]])
        #CI_2σ.append(i)
    if Chi2[i]<=min+1 and Chi2[i]!= nan:
        CI_1σ.append(gamma[i])
        CI_1σ.append([Chi2[i]])
        #CI_1σ.append(i)
        

print("CI_2σ =", CI_2σ)
print("nb d'éléments CI_2σ =",len(CI_2σ)/2)
print("CI_1σ =", CI_1σ)
print("nb d'éléments CI_1σ =",len(CI_1σ)/2)




"""
fig, ax = plt.subplots()
im = ax.imshow(Chi2, interpolation ='bilinear',
               origin ='lower',
               cmap ="bone",extent=(0.65,0.75,0.25,0.35)) 
  
levels = [min+2.3,min+6.7]
CS = ax.contour(Chi2, levels, 
                origin ='lower',
                cmap ='Greens',
                linewidths = 2,extent=(0.65,0.75,0.25,0.35))

ax.set_xlabel('OmegaLambda', fontsize=12)  
ax.set_ylabel('OmegaM', fontsize=12)

ax.clabel(CS, levels,
          inline = 1, 
          fmt ='% 1.1f',
          fontsize = 9)
  



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
 
