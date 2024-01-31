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
plt.scatter(zHD,MU_SHOES , c='blue', marker='.', label='Original Data')
"""
y = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES)
#print(y)
plt.plot(zHD,39.29971437+2.31618666*np.log(zHD*6.60324733), c = 'black',label='Fitted Curve Original Data')
"""

plt.ylabel('modulus distance')
plt.xlabel('redshift')
plt.xscale('log')
plt.title('Hubble diagram of SNe IA')


MU_SHOES_only_alpha = [] ; MU_SHOES_only_alpha_form_exp = [] ; MU_SHOES_alpha_fixé_et_gamma_à_optimiser = [] 
MU_SHOES_alpha_fixé_et_gamma_à_optim_forme_exp = [] ; MU_SHOES_alpha_fixé_et_beta_à_optimiser = [] ; MU_SHOES_alpha_fixe_et_beta_gamma_à_optim = [] 

MU_theory_flatLCDM = []
for z in zHD:
    def f(x):
        return (1/(73.6*((0.334*((1+x)**3)+0.666)**(1/2))))*(3*(10**5))*(10**6)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    #residuals = ? <- difficulté : associer à chaque entité de MU_SHOES un redshift : pas si simple parce qu'il y a plus de light curves (1701) que de SNE1A (1500 et des bananes) donc à priori certaines instances de MU_SHOES ont des z identiques ...   
    MU_theory_flatLCDM.append(result_final)

plt.plot(zHD,MU_theory_flatLCDM,c='red',label='Theory flat LCDM')

for i in range(1701):
        """
        #only_alpha : alpha = 0.04 (+0.02 -0.01)
        mu_only_alpha = MU_SHOES[i] + ((5/2)*np.log10((1+0.04*(1/(1+zHD[i])))/(1+0.04)))
        MU_SHOES_only_alpha.append(mu_only_alpha)
        #only_alpha_exp : alpha = -0.032 (+0.014 -0.014)
        mu__only_alpha_form_exp = mu = MU_SHOES[i] + (5/2)*np.log10((1+(-0.032)*np.exp((zHD[i]/(1+zHD[i]))))/(1+(-0.032)))
        MU_SHOES_only_alpha_form_exp.append(mu__only_alpha_form_exp)
        """
        #alpha_fixé_et_gamma_à_optimiser : gamma = 0.258 (+0.12 -0.12)
        mu_alpha_fixé_et_gamma_à_optimiser = MU_SHOES[i] + (5/2)*(0.258)*np.log10((1+(0.18*(1/(1+zHD[i]))))/(1+0.18))
        MU_SHOES_alpha_fixé_et_gamma_à_optimiser.append(mu_alpha_fixé_et_gamma_à_optimiser)
        """
        #alpha_fixé_et_gamma_à_optim_forme_exp : gamma = -0.23 (+0.108 -0.109)
        mu_alpha_fixé_et_gamma_à_optim_form_exp = MU_SHOES[i] + (5/2)*(-0.023)*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))))/(1+0.18))
        MU_SHOES_alpha_fixé_et_gamma_à_optim_forme_exp.append(mu_alpha_fixé_et_gamma_à_optim_form_exp)
        #alpha_fixé_et_beta_à_optimiser : beta = -3.6 (+1.1 -3.3)
        mu_alpha_fixé_et_beta_à_optimiser = MU_SHOES[i] + (5/2)*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))/(-3.6)))/(1+0.18))
        MU_SHOES_alpha_fixé_et_beta_à_optimiser.append(mu_alpha_fixé_et_beta_à_optimiser)
        #alpha_fixe_et_beta_gamma_à_optim : gamma = -2.2 (+4.1 -4.2) , beta = 2.1 (+2.3 -4)
        mu_alpha_fixe_et_beta_gamma_à_optim = MU_SHOES[i] + (5/2)*(-2.2)*np.log10((1+0.18*np.exp((zHD[i]/(1+zHD[i]))/(0.021)))/(1+0.18))
        MU_SHOES_alpha_fixe_et_beta_gamma_à_optim.append(mu_alpha_fixe_et_beta_gamma_à_optim)
        """


"""
#only_alpha : 38.5979981 2.31255411 8.91327677
plt.scatter(zHD,MU_SHOES_only_alpha , c='g', marker='.', label='only alpha')
y_only_alpha = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_only_alpha)
print(y_only_alpha)
plt.plot(zHD,y_only_alpha[0][0]+y_only_alpha[0][1]*np.log(zHD* y_only_alpha[0][2]), c = 'g',label='Fitted Curve only alpha')

#only_alpha_exp : 37.66489336 2.31241412 13.34412373
plt.scatter(zHD,MU_SHOES_only_alpha_form_exp , c='r', marker='.', label='only alpha exp')
y_only_alpha_exp = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_only_alpha_form_exp)
#print(y_only_alpha_exp)
plt.plot(zHD,y_only_alpha_exp[0][0]+y_only_alpha_exp[0][1]*np.log(zHD*y_only_alpha_exp[0][2]), c ='r' ,label='Fitted Curve only alpha exp')
"""
#alpha_fixé_et_gamma_à_optimiser : 36.41466768 2.31238919 22.91085229
plt.scatter(zHD,MU_SHOES_alpha_fixé_et_gamma_à_optimiser , c='green', marker='.', label='alpha_fixé_et_gamma_à_optimiser ')
"""
y_alpha_fixé_et_gamma_à_optimiser = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_alpha_fixé_et_gamma_à_optimiser)
#print(y_alpha_fixé_et_gamma_à_optimiser)
plt.plot(zHD,y_alpha_fixé_et_gamma_à_optimiser[0][0]+y_alpha_fixé_et_gamma_à_optimiser[0][1]*np.log(zHD*y_alpha_fixé_et_gamma_à_optimiser[0][2]), c ='red' ,label='Fitted Curve alpha_fixé_et_gamma_à_optimiser')

#alpha_fixé_et_gamma_à_optim_forme_exp 35.16131108 2.31234324 39.39771072
plt.scatter(zHD,MU_SHOES_alpha_fixé_et_gamma_à_optim_forme_exp , c='blue', marker='.', label='alpha_fixé_et_gamma_à_optim_forme_exp')
y_alpha_fixé_et_gamma_à_optim_forme_exp = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_alpha_fixé_et_gamma_à_optim_forme_exp)
#print(y_alpha_fixé_et_gamma_à_optim_forme_exp)
plt.plot(zHD,y_alpha_fixé_et_gamma_à_optim_forme_exp[0][0]+y_alpha_fixé_et_gamma_à_optim_forme_exp[0][1]*np.log(zHD*y_alpha_fixé_et_gamma_à_optim_forme_exp[0][2]), c ='blue' ,label='Fitted Curve alpha_fixé_et_gamma_à_optim_forme_exp')

#alpha_fixé_et_beta_à_optimiser 37.57521548 2.31237853 13.86919261
plt.scatter(zHD,MU_SHOES_alpha_fixé_et_beta_à_optimiser , c='yellow', marker='.', label='alpha_fixé_et_beta_à_optimiser')
y_alpha_fixé_et_beta_à_optimiser = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_alpha_fixé_et_beta_à_optimiser)
#print(y_alpha_fixé_et_beta_à_optimiser)
plt.plot(zHD,y_alpha_fixé_et_beta_à_optimiser[0][0]+y_alpha_fixé_et_beta_à_optimiser[0][1]*np.log(zHD*y_alpha_fixé_et_beta_à_optimiser[0][2]), c ='yellow' ,label='Fitted Curve alpha_fixé_et_beta_à_optimiser')

#alpha_fixe_et_beta_gamma_à_optim 38.57879767 2.30005315 8.89739746
plt.scatter(zHD,MU_SHOES_alpha_fixe_et_beta_gamma_à_optim , c='purple', marker='.', label='alpha_fixe_et_beta_gamma_à_optim')
y_alpha_fixe_et_beta_gamma_à_optim = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES_alpha_fixe_et_beta_gamma_à_optim)
#print(y_alpha_fixe_et_beta_gamma_à_optim)
plt.plot(zHD,y_alpha_fixe_et_beta_gamma_à_optim[0][0]+y_alpha_fixe_et_beta_gamma_à_optim[0][1]*np.log(zHD*y_alpha_fixe_et_beta_gamma_à_optim[0][2]), c ='purple' ,label='Fitted Curve alpha_fixe_et_beta_gamma_à_optim')



print("on calcule la différence pour z = ",zHD[1700])
shift = MU_SHOES[1700]-MU_SHOES_alpha_fixé_et_gamma_à_optimiser[1700]
print("différence obtenue avec l'algo = ",shift)
"""
plt.legend()

plt.figure(2)
residuals_sans_modif = []
residuals_avec_modif = []

for i in range(1701):
     residuals_sans_modif.append(MU_SHOES[i]-MU_theory_flatLCDM[i])
     residuals_avec_modif.append(MU_SHOES_alpha_fixé_et_gamma_à_optimiser[i]-MU_theory_flatLCDM[i])

plt.scatter(zHD,residuals_sans_modif,c='green',label="luminosity non-modified")
plt.scatter(zHD,residuals_avec_modif,c='red',label='luminosity modified')
plt.xscale('log')
plt.legend()

plt.show()


