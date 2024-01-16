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

"""
Implémentation tiré de :
- "alpha_fixé et (beta,gamma) à optim" dans le dossier "Mesure uniquement"
- "alpha_fixé_beta_exp" dans le dossier "Modèle uniquement"
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
def likelihood_func(gamma,alpha,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for i in range(len(gamma)):
        likelihood.append([])
    for j in range(len(gamma)):
        for k in range(len(alpha)):
            DeltaD = np.empty(1701)                            
            for i in range(1701):
                mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                def f(x):
                    return (1/(((1+alpha[k][k]*(1/(1+x)))**(1/2))*(73.6)*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) #🔴calcul de la distance lumineuse avec les paramètres cosmologiques FlatLambdaCDM dans Brout et al. 2022 = Analysis on cosmological constraints
                result = spi.quad(f,0,zHD[i])                                               
                mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

                if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                    mu = (mu_shoes + (5/2)*gamma[j][j]*np.log10((1+alpha[k][k]*(1/(1+zHD[i])))/(1+alpha[k][k])))-mu_theory #alpha = 0.18
                    DeltaD[i]=mu
                else :
                    mu = (mu_shoes + (5/2)*gamma[j][j]*np.log10((1+alpha[k][k]*(1/(1+zHD[i])))/(1+alpha[k][k])))-mu_cepheid #alpha = 0.18
                    DeltaD[i]=mu
            DeltaD_transpose = np.transpose(DeltaD)
                #on calcule la likelihood 
            A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
            likelihood[j].append(np.dot(A,DeltaD)) 
    return likelihood
#print(likelihood_func(0.309,0.691,matcov_SN_Cepheid_diag))





#maintenant on va tenter de tracer le diagramme (gamma_range,alpha) en faisant varier ces paramètres et trouver le minimum de la likelihood
delta = 0.1
gamma_range = np.arange(0,5.1, delta) 
alpha_range = np.arange(0.18,5.28, delta)        
gamma_range, alpha_range = np.meshgrid(gamma_range, alpha_range,indexing='ij')
gamma_range = gamma_range.astype(float);  alpha_range = alpha_range.astype(float)
Chi2 = likelihood_func(gamma_range,alpha_range,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)




min = Chi2[0][0]
CL_68 = []
CL_68_alpha = []
CL_68_gamma = []
CL_95 = []

for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min >= Chi2[i][j] and Chi2[i][j] != 0 and Chi2[i][j] != nan:
            min = Chi2[i][j] ; arg_min_gamma = gamma_range[i][j] ; arg_min_alpha = alpha_range[i][j]
print(" alpha= ", arg_min_alpha,"; gamma= ",arg_min_gamma, "; min =", min)
             #/(len(gamma_range)-1)                 #/(len(alpha_range)-1)


for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if  Chi2[i][j]<=min+6.17 and Chi2[i][j] != nan:
            CL_95.append([Chi2[i][j]])
            CL_95.append(gamma_range[i][j])
            CL_95.append(alpha_range[i][j])


        if Chi2[i][j]<=min+2.3 and Chi2[i][j] != nan:
            CL_68.append([Chi2[i][j]])
            CL_68.append(gamma_range[i][j])
            CL_68.append(alpha_range[i][j])
            CL_68_alpha.append(alpha_range[i][j])
            CL_68_gamma.append(gamma_range[i][j])



merge_sort(CL_68_gamma) ; merge_sort(CL_68_alpha)


print("Cl_95 =", CL_95)
print("nb d'éléments CL_95 =",len(CL_95)/3)
print("CL_68 =", CL_68)
print("nb d'éléments CL_68 =",len(CL_68)/3)
print("CL_68_gamma =", CL_68_gamma)
print("CL_68_alpha =", CL_68_alpha)

#print("Chi2[10][10] = ",Chi2[10][10],"gamma_range[10][10] = ",gamma_range[10][10],"alpha_range[10][10] = ",alpha_range[10][10])

"""
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set the viewing angle (adjust these values as needed)
ax.view_init(elev=20, azim=45)  # Change elev and azim to set the view angle

# Plot the 3D surface
surface = ax.plot_surface(gamma_range, alpha_range, Chi2, cmap='viridis')

# Add color bar to show values
fig.colorbar(surface, shrink=0.5, aspect=5)

# Label axes (optional)
ax.set_xlabel('gamma')
ax.set_ylabel('alpha')
ax.set_zlabel('Likelihood')


CS1 = plt.contour(gamma_range, alpha_range , Chi2)
   
fmt = {}
strs = ['1', '2', '3', '4', '5', '6', '7']
for l, s in zip(CS1.levels, strs):
    fmt[l] = s
plt.clabel(CS1, CS1.levels, inline = True,
           fmt = fmt, fontsize = 10)
  
plt.title('matplotlib.pyplot.contour()')




plt.show()
"""