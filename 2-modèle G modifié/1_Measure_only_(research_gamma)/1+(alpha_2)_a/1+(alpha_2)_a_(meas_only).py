from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi

file1 = '..\Pantheon+SH0ES_STAT+SYS.txt'
file2 = '..\Pantheon+Shoes data.txt'

#obtention of covariance matrix
with open(file1) as file:
    data = [line.strip() for line in file]
data = np.array(data) ; data = data.astype(float)
matcov_SN_Cepheid = [data[i:i+1701] for i in range(0, 1701**2, 1701)] 
matcov_SN_Cepheid_diag = np.zeros((1701,1701))
for i in range(1701):
    matcov_SN_Cepheid_diag[i][i] = matcov_SN_Cepheid[i][i]



#obtention of redshift and modulus distances
zHD = []
CEPH_DIST = []      #distances determined thanks to Cepheids anchors
MU_SHOES = []
with open(file2) as file:
    for line in file:
        columns = line.strip().split(' ')
            
        if len(columns) >= 2:
            zHD.append(columns[2])
            CEPH_DIST.append(columns[12])
            MU_SHOES.append(columns[10])
CEPH_DIST.pop(0); zHD.pop(0); MU_SHOES.pop(0)
CEPH_DIST = np.array(CEPH_DIST); zHD = np.array(zHD); MU_SHOES = np.array(MU_SHOES)
CEPH_DIST = CEPH_DIST.astype(float); zHD = zHD.astype(float); MU_SHOES = MU_SHOES.astype(float)




#we define the function which computes the likelihood
def likelihood_func(gamma,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for j in range(len(gamma)):
        DeltaD = np.empty(1701)                            
        for i in range(1701):
            mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
            def f(x):
                return (1/(73.3*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
                #↑we use the values of H0 = 73.3 and OmegaM = 0.334 optimized for the flat ΛCDM 
            result = spi.quad(f,0,zHD[i])                                              
            mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

            if CEPH_DIST[i] == -9.0 : #we check whether the measurement is related to the measurement of a distance with a Cepheid (CEPH_DIST[i] == -9.0 means that this is not the case)
                mu = (mu_shoes + (5/2)*2*gamma[j]*np.log10((1+0.088*(1/(1+zHD[i])))/(1+0.088)))-mu_theory #alpha_2 = 0.088
                DeltaD[i]=mu
            else :
                mu = (mu_shoes + (5/2)*2*gamma[j]*np.log10((1+0.088*(1/(1+zHD[i])))/(1+0.088)))-mu_cepheid
                DeltaD[i]=mu
        DeltaD_transpose = np.transpose(DeltaD) 
        A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
        likelihood.append(np.dot(A,DeltaD)) 
    return likelihood


delta = 0.01
gamma = np.arange(-0.5, 0.51, delta) 
gamma = gamma.astype(float)
Chi2 = likelihood_func(gamma,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)


min = Chi2[0]
CI_1σ = []
CI_2σ = []
for i in range(len(Chi2)) :
    if min >= Chi2[i] and Chi2[i]!= 0 and Chi2[i]!= nan:
        min = Chi2[i] ; arg_min_gamma = gamma[i]
print("gamma= ",arg_min_gamma,"; min =", min)



for i in range(len(Chi2)) :
    if Chi2[i]<=min+4 and Chi2[i]!= nan:
        CI_2σ.append(gamma[i])
    if Chi2[i]<=min+1 and Chi2[i]!= nan:
        CI_1σ.append(gamma[i])
        

print("gamma - 1σ = ",CI_1σ[0], "; gamma + 1σ = ",CI_1σ[-1])
#print("CI_2σ =", CI_2σ)
#print("nb d'éléments CI_2σ =",len(CI_2σ)/3)
#print("CI_1σ =", CI_1σ)
#print("nb d'éléments CI_1σ =",len(CI_1σ)/3)


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

plt.show()
"""
