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
matcov_SN_Cepheid = [data[i:i+1701] for i in range(0, 1701**2, 1701)] #on a les colonnes (ou les lignes) de notre matrice de covariance -> Comment faire pour savoir si un élément de l'array représente
                                                                        #une ligne ou une colonne ? -> pose problème pour le calcul matriciel par la suite ; Pour éviter ce problème, pour l'instant on ne 
                                                                        #récupère que la diagonale
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
def likelihood_func(OmegaMatter, OmegaLambda, mat_cov, zHD, CEPH_DIST, MU_SHOES):

    likelihood = [
        [
            np.dot(
                np.transpose([
                    MU_SHOES[i] - (5 * np.log10(((1 + zHD[i]) * spi.quad(
                        lambda x: (1 / (73.5 * ((OmegaMatter[j][j] * ((1 + x) ** 3) + OmegaLambda[k][k]) ** 0.5))) * (3e5) * 1e6, 0, zHD[i])[0]) / 10))
                    if CEPH_DIST[i] == -9.0
                    else (MU_SHOES[i] - CEPH_DIST[i])
                    for i in range(1701)
                ]),
                np.linalg.inv(mat_cov)
            )
            for k in range(len(OmegaLambda))
        ]
        for j in range(len(OmegaMatter))
    ]

    return likelihood



#print(likelihood_func(0.309,0.691,matcov_SN_Cepheid_diag))






#maintenant on va tenter de tracer le diagramme (Omegam,Omegalambda) en faisant varier ces paramètres et trouver le minimum de la likelihood
delta = 0.1
OmegaM = np.arange(0, 1.1, delta) 
OmegaL = OmegaM.copy().T 
OmegaM, OmegaL = np.meshgrid(OmegaM, OmegaL)
OmegaM = OmegaM.astype(float);  OmegaL = OmegaL.astype(float)
Z = likelihood_func(OmegaM,OmegaL,matcov_SN_Cepheid_diag,zHD,CEPH_DIST,MU_SHOES)

#Z[0][10] = 0 on supprme un outlier (valeur à 10**8)
print(Z)


"""
min = Z[0][1]; indexOm = 0 ; indexOl = 0
for i in range(len(Z)) :
    for j in range(len(Z[i])) :
        if min >= Z[i][j] and Z[i][j] != 0 and Z[i][j] != nan:
            min = Z[i][j] ; indexOm = i ; indexOl = j
print("Om= ",indexOm/(len(OmegaM)-1),"; Ol= ", indexOl/(len(OmegaL)-1), "; min =", min)




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




plt.show()

"""




