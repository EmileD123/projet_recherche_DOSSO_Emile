import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
import scipy.integrate as spi
import csv
   
plt.figure(1)   
# Initialize two empty lists to store the values from the columns
zHD = []
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
            MU_SHOES.append(columns[10])

zHD.pop(0) ;MU_SHOES.pop(0);zHD = np.array(zHD);MU_SHOES = np.array(MU_SHOES) #passer de list à array permet de faire plus de manipulations par la suite
zHD = zHD.astype(float);MU_SHOES = MU_SHOES.astype(float)
plt.figure(1)
#C'est bon on a nos deux listes : il ne reste plus qu'à tracer le nuages de points
plt.scatter(zHD,MU_SHOES , c='b', marker='.', label='Original Data')
plt.ylabel('modulus distance')
plt.xlabel('redshift')
plt.xscale('log')
plt.title('Hubble diagram of SNe IA')




#on trace la courbe théorique selon le modèle LCDM
MU_theory_flatLCDM = []
for z in zHD:
    def f(x):
        return (1/(73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_theory_flatLCDM.append(result_final)


MU_only_alpha = []
for z in zHD:
    def f(x):
        return (1/(((1+0.18*(1/(1+x)))**(1/2))*73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_only_alpha.append(result_final)


"""
MU_alpha_fixé_exp= [] 
for z in zHD:
    def f(x):
        return (1/(((1+(0.18)*np.exp(x/(1+x)))**(1/2))*73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_alpha_fixé_exp.append(result_final)

MU_alpha_fixé_beta_exp = [] #beta = -0.001
for z in zHD:
    def f(x):
        return (1/(((1+(0.18)*np.exp(x/((1+x)*(-1*10**(-0.001)))))**(1/2))*73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_alpha_fixé_beta_exp.append(result_final)
"""


plt.plot(zHD,MU_theory_flatLCDM,c='green',label='Flat LCDM')
plt.plot(zHD,MU_only_alpha ,c='red',label='only_alpha')
plt.legend()
"""
plt.plot(zHD,MU_alpha_fixé_exp,c='orange',label='alpha_fixé_exp')
plt.plot(zHD,MU_alpha_fixé_beta_exp,c='g',label='alpha_fixé_beta_exp')


residuals = []
for i in range(1701):
    def f(x):
        return (1/(73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) 
    result = spi.quad(f,0,zHD[i])
    result_final = 5*np.log10(((1+zHD[i])*result[0])/10)
    #residuals = ? <- difficulté : associer à chaque entité de MU_SHOES un redshift : pas si simple parce qu'il y a plus de light curves (1701) que de SNE1A (1500 et des bananes) donc à priori certaines instances de MU_SHOES ont des z identiques ...   
    residuals.append(result_final-MU_SHOES[i])

plt.figure(2)

plt.scatter(zHD,residuals,c='black' ,marker='.',label='residuals')
plt.ylabel('modulus distance')
plt.xlabel('redshift')
plt.xscale('log')
plt.legend()

"""
plt.figure(2)
residuals_sans_modif = []
residuals_avec_modif = []

for i in range (1701):
    residuals_sans_modif.append(MU_SHOES[i]-MU_theory_flatLCDM[i])
    residuals_avec_modif.append(MU_SHOES[i]-MU_only_alpha[i])

plt.scatter(zHD,residuals_sans_modif,c='green',label='model non-modified')
plt.scatter(zHD,residuals_avec_modif,c='red',label='model with introduction of alpha')
plt.title('residuals with or without modification')
plt.xscale('log')
plt.xlabel('redshift')
plt.ylabel('residuals')


plt.legend()

plt.show()