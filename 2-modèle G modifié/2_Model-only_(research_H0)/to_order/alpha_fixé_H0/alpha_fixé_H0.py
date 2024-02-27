from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import scipy.integrate as spi
import time
from time import time_ns
import os
#file1 = '../Pantheon+SH0ES_STAT+SYS.txt'
#file2 = '../Pantheon+Shoes data.txt'


#on s'attaque à la matrice de covariance
with open('Pantheon+SH0ES_STAT+SYS.txt') as file:
    data = [line.strip() for line in file]
data = np.array(data) ; data = data.astype(float)
matcov_SN_Cepheid = [data[i:i+1701] for i in range(0, 1701**2, 1701)] #peut importe si on range ligne par ligne ou colonne par colonne car la matrice de covariance est symétrique
matcov_SN_Cepheid_1ere_moitie = [data[i:i+849] for i in range(0, 849**2, 849)]
matcov_SN_Cepheid_2nde_moitie = [data[851+i:850+i+850] for i in range(0, 849**2, 849)]




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

"""
zHD_1ere_moitie = []
CEPH_DIST_1ere_moitie = []      
MU_SHOES_1ere_moitie = []
with open('Pantheon+Shoes data.txt') as file:
    for i in range(850):
        columns = line.strip().split(' ')
        if len(columns) >= 2:
            zHD_1ere_moitie.append(columns[2])
            CEPH_DIST_1ere_moitie.append(columns[12])
            MU_SHOES_1ere_moitie.append(columns[10])
CEPH_DIST_1ere_moitie.pop(0); zHD_1ere_moitie.pop(0); MU_SHOES_1ere_moitie.pop(0)
CEPH_DIST_1ere_moitie = np.array(CEPH_DIST_1ere_moitie); zHD_1ere_moitie = np.array(zHD_1ere_moitie); MU_SHOES_1ere_moitie = np.array(MU_SHOES_1ere_moitie)
CEPH_DIST_1ere_moitie = CEPH_DIST_1ere_moitie.astype(float); zHD_1ere_moitie = zHD_1ere_moitie.astype(float); MU_SHOES_1ere_moitie = MU_SHOES_1ere_moitie.astype(float)       

zHD_2nde_moitie = []
CEPH_DIST_2nde_moitie = []      
MU_SHOES_2nde_moitie = []
with open('Pantheon+Shoes data.txt') as file:
    for i in range(851,1701):
        columns = line.strip().split(' ')
        if len(columns) >= 2:
            zHD_2nde_moitie.append(columns[2])
            CEPH_DIST_2nde_moitie.append(columns[12])
            MU_SHOES_2nde_moitie.append(columns[10])
CEPH_DIST_2nde_moitie.pop(0); zHD_2nde_moitie.pop(0); MU_SHOES_2nde_moitie.pop(0)
CEPH_DIST_2nde_moitie = np.array(CEPH_DIST_2nde_moitie); zHD_2nde_moitie = np.array(zHD_2nde_moitie); MU_SHOES_2nde_moitie = np.array(MU_SHOES_2nde_moitie)
CEPH_DIST_2nde_moitie = CEPH_DIST_2nde_moitie.astype(float); zHD_2nde_moitie = zHD_2nde_moitie.astype(float); MU_SHOES_2nde_moitie = MU_SHOES_2nde_moitie.astype(float)       
"""

#on définit la fonction qui calcule la likelihood
def likelihood_func(H0,mat_cov,zHD,CEPH_DIST,MU_SHOES) :
    likelihood=[]
    for j in range(len(H0)):
        DeltaD = np.empty(len(mat_cov))
        for i in range(len(mat_cov)):
            mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
            def f(x):
                return (1/(((1+0.192*(1/(1+x)))**(1/2))*H0[j]*(((0.334)*((1+x)**3)+(0.666))**(1/2))))*(3*(10**5))*(10**6) # alpha = ((H0riess/H0planck)^2)-1 ≈ 0.18 ; alpha_brout_f_lcdm = ((H0brout_f_lcdm /H0planck)^2)-1 ≈ 0.192 
            #🔴calcul de la distance lumineuse avec les paramètres cosmologiques FlatLambdaCDM dans Brout et al. 2022 = Analysis on cosmological constraints
            result = spi.quad(f,0,zHD[i])                                           
            mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)


            if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
                mu = mu_shoes - mu_theory #(mu_shoes + (5/2)*0.*np.log10((1+0.192*(1/(1+zHD[i])))/(1+0.192)))
                DeltaD[i]=mu
            else :
                mu = mu_shoes - mu_cepheid #(mu_shoes + (5/2)*0.*np.log10((1+0.192*(1/(1+zHD[i])))/(1+0.192)))
                DeltaD[i]=mu
                #on calcule la transposée
        DeltaD_transpose = np.transpose(DeltaD)
        #on calcule la likelihood 
        A = np.dot(DeltaD_transpose,np.linalg.inv(mat_cov))
        likelihood.append(np.dot(A,DeltaD)) 
    return likelihood





delta = 0.001
H0 = np.arange(67.595, 67.597, delta)                
H0 = H0.astype(float)
tps1 = time_ns()/1e9
Chi2 = likelihood_func(H0,matcov_SN_Cepheid,zHD,CEPH_DIST,MU_SHOES)     
#Chi2_1ere_moitie = likelihood_func(H0,matcov_SN_Cepheid_1ere_moitie,zHD_1ere_moitie,CEPH_DIST_1ere_moitie,MU_SHOES_1ere_moitie)
#Chi2_2nde_moitie = likelihood_func(H0,matcov_SN_Cepheid_2nde_moitie,zHD_2nde_moitie,CEPH_DIST_2nde_moitie,MU_SHOES_2nde_moitie)
tps2 = time_ns()/1e9
print("temps de calcul de Chi2 = ", tps2 - tps1, " s")
Chi2 = np.array(Chi2) ; Chi2 = Chi2.astype(float)
print(Chi2)
#Chi2_1ere_moitie = np.array(Chi2_1ere_moitie) ; Chi2_1ere_moitie = Chi2_1ere_moitie.astype(float)
#Chi2_2nde_moitie = np.array(Chi2_2nde_moitie) ; Chi2_2nde_moitie = Chi2_2nde_moitie.astype(float)




min = Chi2[0] 
#min_1ere_moitie = Chi2_1ere_moitie[0] ; min_2nde_moitie = Chi2_2nde_moitie[0]
CI_1σ = [] ; CI_2σ= [] 

for i in range(len(Chi2)) :
    if min >= Chi2[i] and Chi2[i]!= 0 and Chi2[i]!= nan:
        min = Chi2[i] ; arg_min_H0 = H0[i]
"""
for i in range(len(Chi2_1ere_moitie)) :
    if min_1ere_moitie >= Chi2_1ere_moitie[i] and Chi2_1ere_moitie[i]!= 0 and Chi2_1ere_moitie[i]!= nan:
        min_1ere_moitie = Chi2_1ere_moitie[i] 
for i in range(len(Chi2_2nde_moitie)) :
    if min_2nde_moitie >= Chi2_2nde_moitie[i] and Chi2_2nde_moitie[i]!= 0 and Chi2_2nde_moitie[i]!= nan:
        min_2nde_moitie = Chi2_2nde_moitie[i]
"""
print("H0= ",arg_min_H0,"; min =", min)
#print(" min_1ere_moitie = ", min_1ere_moitie,"; min_2nde_moitie = ", min_2nde_moitie)
             #/(len(H0)-1)                 #/(len(H0)-1)


for i in range(len(Chi2)) :
    if min <= Chi2[i] and Chi2[i]<=min+4 and Chi2[i]!= nan: 
        CI_2σ.append(H0[i])
        CI_2σ.append([Chi2[i]])

    if min <= Chi2[i] and Chi2[i]<=min+1 and Chi2[i] != nan:
        CI_1σ.append(H0[i])
        CI_1σ.append([Chi2[i]])
        

print("CI_2σ=", CI_2σ)
print("nb d'éléments CI_2σ=",len(CI_2σ)/2)
print("CI_1σ =", CI_1σ)
print("nb d'éléments CI_1σ =",len(CI_1σ)/2)


 
