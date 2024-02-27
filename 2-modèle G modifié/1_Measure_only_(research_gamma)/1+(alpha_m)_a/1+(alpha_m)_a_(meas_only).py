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
def likelihood_func(m,gamma,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for i in range(len(gamma)):
        likelihood.append([])
    for j in range(len(gamma)):
        for k in range(len(m)):
            DeltaD = np.empty(1701)                            
            for i in range(1701):
                mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
                def f(x):
                    return (1/(73.3 *((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) 
                    #↑we use the values of H0 = 73.3 and OmegaM = 0.334 optimized for the flat ΛCDM 
                result = spi.quad(f,0,zHD[i])                                              
                mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

                if CEPH_DIST[i] == -9.0 : #we check whether the measurement is related to the measurement of a distance with a Cepheid (CEPH_DIST[i] == -9.0 means that this is not the case)
                    mu = (mu_shoes + (5/2)*m[k][k]*gamma[j][j]*np.log10((1+((73.3/67.4)**(2/m[k][k])-1)*(1/(1+zHD[i])))/(1+((73.3/67.4)**(2/m[k][k])-1))))-mu_theory #alpha_m = ((H0riess/H0planck)**(2/m)-1) = ((73.3/67.4)**(2/m)-1)
                    DeltaD[i]=mu
                else :
                    mu = (mu_shoes + (5/2)*m[k][k]*gamma[j][j]*np.log10((1+((73.3/67.4)**(2/m[k][k])-1)*(1/(1+zHD[i])))/(1+((73.3/67.4)**(2/m[k][k])-1))))-mu_cepheid
                    DeltaD[i]=mu
            DeltaD_transpose = np.transpose(DeltaD) 
            A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
            likelihood[j].append(np.dot(A,DeltaD)) 
    return likelihood


delta = 1
gamma = np.arange(-5,5,delta/2) 
m = np.arange(-100,100,delta*10)
gamma , m = np.meshgrid(gamma, m,indexing='ij')
gamma = gamma.astype(float); m = m.astype(float)
Chi2 = likelihood_func(m,gamma,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)


min = Chi2[0][1]
CI_1σ = [] ; CI_2σ = [] ; CI_1σ_gamma = []; CI_1σ_m = []; CI_1σ_m_chi2 = [] 

for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        if min >= Chi2[i][j] and Chi2[i][j] != 0 and Chi2[i][j] != nan:
            min = Chi2[i][j] ; arg_min_gamma = gamma[i][j] ; arg_min_m =m[i][j]
print("gamma= ", arg_min_gamma, "; m= ", arg_min_m, "; min =", min)


for i in range(len(Chi2)) :
    for j in range(len(Chi2[i])) :
        #if min <= Chi2[i][j] and Chi2[i][j]<=min+6.17 and Chi2[i][j] != nan:
        #CI_2σ.append([Chi2[i][j]]);CI_2σ.append(gamma[i][j]);CI_2σ.append(m[i][j])
            
        if min <= Chi2[i][j] and Chi2[i][j]<=min+2.3 and Chi2[i][j] != nan:
           # CI_1σ.append([Chi2[i][j]]);CI_1σ.append(gamma[i][j]);CI_1σ.append(m[i][j])
            CI_1σ_gamma.append(gamma[i][j])
            CI_1σ_m.append(m[i][j])
            CI_1σ_m_chi2.append(Chi2[i][j])

merge_sort(CI_1σ_gamma);merge_sort(CI_1σ_m)

#print("CI_2σ =", CI_2σ)
#print("nb d'éléments CI_2σ =",len(CI_2σ)/3)
#print("CI_1σ =", CI_1σ)
#print("nb d'éléments CI_1σ =",len(CI_1σ)/3)
print("gamma-1σ =",CI_1σ_gamma[0]) ; print("; gamma+1σ =",CI_1σ_gamma[-1])#print("CI_1σ_gamma =",CI_1σ_gamma)
print("m-1σ =",CI_1σ_m[0]) ; print("; m+1σ =",CI_1σ_m[-1])#print("CI_1σ_m=",CI_1σ_m)
print("CI_1σ_m = ", CI_1σ_m)
print("CI_1σ_m_chi2 = ", CI_1σ_m_chi2)



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
