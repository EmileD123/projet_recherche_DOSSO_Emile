from math import nan
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi


zHD = []
MU_SHOES = []
with open('Pantheon+Shoes data.txt') as file:
    for line in file:
        columns = line.strip().split(' ')
        
        if len(columns) >= 2:
            zHD.append(columns[2])
            MU_SHOES.append(columns[10])

zHD.pop(0) ;MU_SHOES.pop(0);zHD = np.array(zHD);MU_SHOES = np.array(MU_SHOES) #passer de list à array permet de faire plus de manipulations par la suite
zHD = zHD.astype(float);MU_SHOES = MU_SHOES.astype(float)



#C'est bon on a nos deux listes : il ne reste plus qu'à tracer le nuages de points
plt.figure(1)
plt.scatter(zHD,MU_SHOES , c='b', marker='.', label='Original Data')
plt.ylabel('modulus distance')
plt.xlabel('redshift')
plt.xscale('log')
plt.title('Hubble diagram of SNe IA')

#on trace la courbe théorique selon le modèle LCDM ainsi que la ou les courbes issues de théories modifiées qu'on veut étudier
MU_theory_flatLCDM = []
for z in zHD:
    def f(x):
        return (1/(73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_theory_flatLCDM.append(result_final)

MU_alpha_brout_f_lcdm_fixé_H0 = []
for z in zHD:
    def f(x):
        return (1/(((1+0.192*(1/(1+x)))**(1/2))*67.60*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)  #alpha_brout_f_lcdm = ((H0brout_f_lcdm /H0planck)^2)-1 ≈ 0.192 -> H0 = 67.6
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_alpha_brout_f_lcdm_fixé_H0.append(result_final)



"""
MU_alpha_fixé_HO =[]
for z in zHD:
    def f(x):
        return (1/(((1+0.18*(1/(1+x)))**(1/2))*67.91*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6) # alpha = ((H0riess/H0planck)^2)-1 ≈ 0.18 -> H0 = 67.9
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_alpha_fixé_HO.append(result_final)
MU_alpha_fixé_HO = []

for z in zHD:
    def f(x):
        return (1/(((1+0.192*(1/(1+x)))**(1/2))*67.60*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)  #alpha_brout_f_lcdm = ((H0brout_f_lcdm /H0planck)^2)-1 ≈ 0.192 -> H0 = 67.6
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_alpha_fixé_HO.append(result_final)

MU_only_alpha = []
for z in zHD:
    def f(x):
        return (1/(((1+0.18*(1/(1+x)))**(1/2))*73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_only_alpha.append(result_final)

MU_alpha_fixé_H0_Omegam = []
for z in zHD:
    def f(x):
        return (1/(((1+0.18*(x/(1+x)))**(1/2))*73.3*((0.286*((1+x)**3)+(1-0.286))**(1/2))))*(3*(10**5))*(10**6) #🔴Omegam divisé par 100 ici ! ; calcul de la distance lumineuse avec les paramètres cosmologiques (OmegaLambda correspondant au flat ΛCDM dans Brout et al. 2022 = Analysis on cosmological constraints)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_alpha_fixé_H0_Omegam.append(result_final)

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
plt.plot(zHD,MU_alpha_brout_f_lcdm_fixé_H0,c='red',label='alpha_brout_f_lcdm_fixé_HO')

"""
plt.plot(zHD,MU_alpha_fixé_HO,c='red',label='alpha_fixé_HO')
plt.plot(zHD,MU_alpha_brout_f_lcdm_fixé_H0,c='purple',label='alpha_brout_f_lcdm_fixé_H0')
plt.plot(zHD,MU_alpha_fixé_H0_Omegam,c='red',label='alpha_fixé_H0_Omegam')
#plt.plot(zHD,MU_only_alpha ,c='red',label='only_alpha')
plt.plot(zHD,MU_alpha_fixé_exp,c='orange',label='alpha_fixé_exp')
plt.plot(zHD,MU_alpha_fixé_beta_exp,c='g',label='alpha_fixé_beta_exp')
"""
plt.legend()




#Ensuite on trace les résidus du modèle LCDM classique ainsi que ceux de la ou les théories modifiées qu'on veut étudier
plt.figure(2)

residuals_sans_modif = []
residuals_avec_modif = []


for i in range (1701):
    residuals_sans_modif.append(MU_SHOES[i]-MU_theory_flatLCDM[i])
    residuals_avec_modif.append(MU_SHOES[i]-MU_alpha_brout_f_lcdm_fixé_H0[i])
    #residuals_avec_modif.append(MU_SHOES[i]-MU_alpha_fixé_HO[i])
    #residuals_avec_modif.append(MU_SHOES[i]-MU_only_alpha[i])
    #residuals_avec_modif.append(MU_SHOES[i]-MU_alpha_fixé_H0_Omegam[i])

plt.scatter(zHD,residuals_sans_modif,c='green',label='model non-modified')
plt.scatter(zHD,residuals_avec_modif,c='red',label='model with introduction of alpha_brout_f_lcdm_fixé_HO')
#plt.scatter(zHD,residuals_avec_modif,c='red',label='model with introduction of alpha')
#plt.scatter(zHD,residuals_avec_modif,c='red',label='model with introduction of alpha_fixé_H0_Omegam')
plt.title('residuals with or without modification')
plt.xscale('log')
plt.xlabel('redshift')
plt.ylabel('residuals')

plt.legend()


plt.show()