import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate as spi
import csv
   
    
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

plt.figure(1)

#C'est bon on a nos deux listes : il ne reste plus qu'à tracer le nuages de points
plt.scatter(zHD,MU_SHOES , c='black', marker='.', label='Original Data')
y = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES)
#print(y)
#plt.plot(zHD,39.29971437+2.31618666*np.log(zHD*6.60324733),linestyle ='--'  ,c = 'black', label='Fitted Curve Original Data')


plt.ylabel('modulus distance')
plt.xlabel('redshift')
plt.xscale('log')
plt.title('Hubble diagram of SNe IA')

MU_SHOES_alpha_fixé_et_gamma_à_optimiser = [] ; MU_SHOES_alpha_fixé_et_gamma_à_optim_forme_exp = [] ; MU_SHOES_alpha_fixe_et_beta_gamma_à_optim = [] 



for i in range(1701):
        #alpha_fixé_et_gamma_à_optimiser : gamma = 0.258 (+0.12 -0.12) pour modif modèle uniquement ; gamma = 4.847 (+? -?) pour modifs modèle et données
        mu_alpha_fixé_et_gamma_à_optimiser = MU_SHOES[i] + (5/2)*(4.847)*np.log10((1+0.18*(1/(1+zHD[i])))/(1+0.18))
        MU_SHOES_alpha_fixé_et_gamma_à_optimiser.append(mu_alpha_fixé_et_gamma_à_optimiser)
        """
        #alpha_fixé_et_gamma_à_optim_forme_exp : gamma = -0.23 (+0.108 -0.109)
        mu_alpha_fixé_et_gamma_à_optim_form_exp = MU_SHOES[i] + (5/2)*(-0.023)*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))))/(1+0.18))
        MU_SHOES_alpha_fixé_et_gamma_à_optim_forme_exp.append(mu_alpha_fixé_et_gamma_à_optim_form_exp)
        #alpha_fixe_et_beta_gamma_à_optim : gamma = -2.2 (+4.1 -4.2) , beta = 2.1 (+2.3 -4)
        mu_alpha_fixe_et_beta_gamma_à_optim = MU_SHOES[i] + (5/2)*(-2.2)*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))/(2.1)))/(1+0.18))
        MU_SHOES_alpha_fixe_et_beta_gamma_à_optim.append(mu_alpha_fixe_et_beta_gamma_à_optim)
        """

#alpha_fixé_et_gamma_à_optimiser : 36.41466768 2.31238919 22.91085229
plt.scatter(zHD,MU_SHOES_alpha_fixé_et_gamma_à_optimiser , c='pink', marker='.', label='alpha_fixé_et_gamma_à_optimiser ')
y_alpha_fixé_et_gamma_à_optimiser = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_alpha_fixé_et_gamma_à_optimiser)
#print(y_alpha_fixé_et_gamma_à_optimiser)
#plt.plot(zHD,y_alpha_fixé_et_gamma_à_optimiser[0][0]+y_alpha_fixé_et_gamma_à_optimiser[0][1]*np.log(zHD*y_alpha_fixé_et_gamma_à_optimiser[0][2]), linestyle ='--'  ,c ='r' ,label='Fitted Curve alpha_fixé_et_gamma_à_optimiser')

"""
#alpha_fixé_et_gamma_à_optim_forme_exp 35.16131108 2.31234324 39.39771072
plt.scatter(zHD,MU_SHOES_alpha_fixé_et_gamma_à_optim_forme_exp , c='blue', marker='.', label='alpha_fixé_et_gamma_à_optim_forme_exp')
y_alpha_fixé_et_gamma_à_optim_forme_exp = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_alpha_fixé_et_gamma_à_optim_forme_exp)
#print(y_alpha_fixé_et_gamma_à_optim_forme_exp)
plt.plot(zHD,y_alpha_fixé_et_gamma_à_optim_forme_exp[0][0]+y_alpha_fixé_et_gamma_à_optim_forme_exp[0][1]*np.log(zHD*y_alpha_fixé_et_gamma_à_optim_forme_exp[0][2]), linestyle ='--'  ,c ='orange' ,label='Fitted Curve alpha_fixé_et_gamma_à_optim_forme_exp')

#alpha_fixe_et_beta_gamma_à_optim 38.57879767 2.30005315 8.89739746
plt.scatter(zHD,MU_SHOES_alpha_fixe_et_beta_gamma_à_optim , c='purple', marker='.', label='alpha_fixe_et_beta_gamma_à_optim')
y_alpha_fixe_et_beta_gamma_à_optim = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_alpha_fixe_et_beta_gamma_à_optim)
#print(y_alpha_fixe_et_beta_gamma_à_optim)
plt.plot(zHD,y_alpha_fixe_et_beta_gamma_à_optim[0][0]+y_alpha_fixe_et_beta_gamma_à_optim[0][1]*np.log(zHD*y_alpha_fixe_et_beta_gamma_à_optim[0][2]), linestyle ='--'  ,c ='green' ,label='Fitted Curve alpha_fixe_et_beta_gamma_à_optim')
"""


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
        return (1/(((1+0.18)**(1/2))*73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
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



plt.plot(zHD,MU_theory_flatLCDM,c='black',label='Flat LCDM')
plt.plot(zHD,MU_only_alpha ,c='r',label='only_alpha')
plt.legend()
"""
plt.plot(zHD,MU_alpha_fixé_exp,c='orange',label='alpha_fixé_exp')
plt.plot(zHD,MU_alpha_fixé_beta_exp,c='g',label='alpha_fixé_beta_exp')
"""

plt.figure(2)
plt.ylabel('modulus distance residuals')
plt.xlabel('redshift')
plt.xscale('log')
plt.title('residuals with or without change of G')
residuals_sans_modif = []
for i in range(len(zHD)):
    diff =  MU_SHOES[i] - MU_theory_flatLCDM[i]
    residuals_sans_modif.append(diff)
plt.scatter(zHD,residuals_sans_modif,c='green',label='residuals with models and data non-modified')


"""
plt.figure(3)
plt.ylabel('modulus distance residuals')
plt.xlabel('redshift')
plt.xscale('log')
plt.title('residuals with models and data modified')
"""
residuals_avec_modif = []
for i in range(len(zHD)):
    diff =  MU_SHOES_alpha_fixé_et_gamma_à_optimiser[i] - MU_only_alpha[i] # -0.18 # permet de superoposer une partie des points rouges et verts (seulement aux z plus faibles , ce qui montre que ldécalage entre rouget vert dépend de z)
    residuals_avec_modif.append(diff)
plt.scatter(zHD,residuals_avec_modif,c='red',label='residuals with models and data modified')
plt.legend()
plt.show()


