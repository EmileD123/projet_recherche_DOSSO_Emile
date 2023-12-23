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




#on définit la fonction qui calcule la likelihood
def likelihood_func(beta,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for j in range(len(beta)):
        DeltaD = np.empty(1701)                            
        for i in range(1701):
            mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
            def f(x):
                return (1/(73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) #🔴calcul de la distance lumineuse avec les paramètres cosmologiques (H0=73.5 pour le flat wCDM dans Brout et al. 2022 = Analysis on cosmological constraints)
            result = spi.quad(f,0,zHD[i])                                               #idem
            mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

            if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                mu = mu_shoes + (5/2)*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))/(beta[j])))/(1+0.18)) - mu_theory #alpha = 0.18
                DeltaD[i]=mu
            else :
                mu = mu_shoes + (5/2)*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))/(beta[j])))/(1+0.18)) - mu_cepheid
                DeltaD[i]=mu
                #on calcule la transposée
        DeltaD_transpose = np.transpose(DeltaD)
        #on calcule la likelihood 
        A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
        likelihood.append(np.dot(A,DeltaD)) 
    return likelihood
#print(likelihood_func(0.309,0.691,matcov_SN_Cepheid_diag))






delta = 0.1
beta = np.arange(-15, -1.5, delta) 
beta = beta.astype(float)
Z = likelihood_func(beta,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Z = np.array(Z) ; Z = Z.astype(float)
#Z[0][10] = 0 on supprime un outlier (valeur à 10**8)
#Z[0][0] = 10000
print(Z) ; print(beta)




min = Z[20]
CL_68 = []
CL_95 = []
for i in range(len(Z)) :
    if min >= Z[i] and Z[i]!= 0 and Z[i]!= nan:
        min = Z[i] ; arg_min_beta = beta[i]
print("beta= ",arg_min_beta,"; min =", min)
             #/(len(beta)-1)                 #/(len(beta)-1)


for i in range(len(Z)) :
    if Z[i]<=(min+3.841) and Z[i]!= nan:
        CL_95.append(beta[i])
        CL_95.append([Z[i]])
        #CL_95.append(i)
    if Z[i]<=(min+1.0) and Z[i]!= nan:
        CL_68.append(beta[i])
        CL_68.append([Z[i]])
        #CL_68.append(i)
        

print("Cl_95 =", CL_95)
print("nb d'éléments CL_95 =",len(CL_95)/2)
print("CL_68 =", CL_68)
print("nb d'éléments CL_68 =",len(CL_68)/2)




"""
fig, ax = plt.subplots()
im = ax.imshow(Z, interpolation ='bilinear',
               origin ='lower',
               cmap ="bone",extent=(0.65,0.75,0.25,0.35)) 
  
levels = [min+2.3,min+6.7]
CS = ax.contour(Z, levels, 
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
surface = ax.plot_surface(OmegaM, OmegaL, Z, cmap='viridis')

# Add color bar to show values
fig.colorbar(surface, shrink=0.5, aspect=5)

# Label axes (optional)
ax.set_xlabel('OmegaM')
ax.set_ylabel('OmegaL')
ax.set_zlabel('Likelihood')


CS1 = plt.contour(OmegaM, OmegaL, Z)
   
fmt = {}
strs = ['1', '2', '3', '4', '5', '6', '7']
for l, s in zip(CS1.levels, strs):
    fmt[l] = s
plt.clabel(CS1, CS1.levels, inline = True,
           fmt = fmt, fontsize = 10)
  
plt.title('matplotlib.pyplot.contour()')

"""


plt.show()
 
