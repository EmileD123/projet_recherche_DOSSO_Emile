import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
import scipy.integrate as spi
import csv
   
    
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
#C'est bon on a nos deux listes : il ne reste plus qu'à tracer le nuages de points
plt.scatter(zHD,MU_SHOES , c='b', marker='.', label='Original Data')
plt.ylabel('modulus distance')
plt.xlabel('redshift')
plt.xscale('log')
plt.title('Hubble diagram of SNe IA')


#on approxime la courbe expérimentale (intérêt?)

y = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES)
print(y)
plt.plot(zHD,39.29971437+2.31618666*np.log(zHD*6.60324733),label='Fitted Log Curve')



#on trace la courbe théorique selon le modèle wCDM
# PROBLEME : ne recoupe pas les données comme dans le cas du papier ...
MU_theory = []
for z in zHD:
    def f(x):
        return (1/(73.5*((0.309*((1+x)**3)+0.691)**(1/2))))*(3*(10**5))*(10**6)
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    #print(result_final)   
    MU_theory.append(result_final)
plt.plot(zHD,MU_theory,label='Flat wCDM')

plt.legend()
plt.show()


