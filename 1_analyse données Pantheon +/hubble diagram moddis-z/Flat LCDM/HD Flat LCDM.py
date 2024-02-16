import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
import scipy.integrate as spi
import csv
   
    
file = "..\Pantheon+Shoes data.txt"

# Initialize two empty lists to store the values from the columns
zHD = []
MU_SHOES = []
CEPH_DIST = []
# Open the text file for reading
with open(file) as file:
    # Loop through each line in the file
    for line in file:
        # Split the line into columns based on a tab separator
        columns = line.strip().split(' ')
        
        # Check if there are at least two columns
        if len(columns) >= 2:
            # Append the values from the first and second columns to their respective lists
            zHD.append(columns[2])
            MU_SHOES.append(columns[10])
            CEPH_DIST.append(columns[12])



CEPH_DIST.pop(0); zHD.pop(0); MU_SHOES.pop(0)
CEPH_DIST = np.array(CEPH_DIST); zHD = np.array(zHD); MU_SHOES = np.array(MU_SHOES) #passer de liste à array facilite les manipulations par la suite
CEPH_DIST = CEPH_DIST.astype(float); zHD = zHD.astype(float); MU_SHOES = MU_SHOES.astype(float)
plt.figure(1)

Dl_SHOES = [] #distances lumineuses déterminées à partir de MU_SHOES (en parsecs)
for i in range(len(MU_SHOES)):
    dl = 10**((MU_SHOES[i]/5)+1)
    Dl_SHOES.append(dl)

#C'est bon on a nos deux listes : il ne reste plus qu'à tracer le nuages de points
    
#plt.scatter(zHD,Dl_SHOES , c='b', marker='.', label='Original Data')
#plt.ylabel('luminous distance')
plt.scatter(zHD,MU_SHOES , c='b', marker='.', label='Original Data')
plt.ylabel('modulus distance')
plt.xlabel('redshift')
plt.xscale('log')
plt.title('Hubble diagram of SNe IA')


#on approxime la courbe expérimentale (intérêt?)
"""
y = scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(c*t),  zHD,  MU_SHOES)
print(y)
plt.plot(zHD,39.29971437+2.31618666*np.log(zHD*6.60324733),label='Fitted Log Curve')
"""


#on trace la courbe théorique selon le modèle LCDM
# PROBLEME : ne recoupe pas les données comme dans le cas du papier ...
MU_theory = []
Dl_theory = []
residuals = []

for z in zHD:
    def f(x):
        return (1/(73.6*((0.334*((1+x)**3)+0.664)**(1/2))))*(3*(10**5))*(10**6) 
    result = spi.quad(f,0,z)
    result_final = 5*np.log10(((1+z)*result[0])/10)
    MU_theory.append(result_final)
    Dl_theory.append(10**((result_final/5)+1))

for i in range(1701):
    if CEPH_DIST == -9.0 :
        residuals.append(MU_SHOES[i]-MU_theory[i])
    else : 
        residuals.append(MU_SHOES[i]-CEPH_DIST[i])


plt.figure(1)
plt.plot(zHD,MU_theory,c='orange',label='Flat wCDM')
#plt.plot(zHD,Dl_theory,c='red',label='luminous distance Flat LCDM')
plt.title('Hubble Diagram flat wCDM')
plt.xscale('log')
plt.legend()

plt.figure(2)
plt.scatter(zHD,residuals,c='g')
plt.title('residuals flat wCDM')
plt.xscale('log')
plt.legend()


plt.show()