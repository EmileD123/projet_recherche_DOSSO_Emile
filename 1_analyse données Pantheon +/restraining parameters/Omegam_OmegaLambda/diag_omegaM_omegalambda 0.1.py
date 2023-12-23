import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy
import scipy.integrate as spi

#on commence par créer les vecteurs DeltaD

DeltaD = []
zHD = []
CEPH_DIST = []      #les distances calculées à l'aide de la présence de Cepheids
IS_CALIBRATOR = []  #binaire pour désigner si la distance de la SN est calibrée à laide d'une Cépheide (SN situé dans une galaxie avec une Cépheide)
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
            IS_CALIBRATOR.append(columns[13])
            MU_SHOES.append(columns[10])
CEPH_DIST.pop(0); zHD.pop(0); MU_SHOES.pop(0); IS_CALIBRATOR.pop(0)
CEPH_DIST = np.array(CEPH_DIST); zHD = np.array(zHD); MU_SHOES = np.array(MU_SHOES); IS_CALIBRATOR = np.array(IS_CALIBRATOR)
CEPH_DIST = CEPH_DIST.astype(float); zHD = zHD.astype(float); MU_SHOES = MU_SHOES.astype(float); IS_CALIBRATOR = IS_CALIBRATOR.astype(float)

for i in range(1701):
    mu_shoes = MU_SHOES[i] ; mu_cepheid = CEPH_DIST[i]
    def f(x):
        return (1/(73.5*((0.309*((1+x)**3)+0.691)**(1/2))))*(3*(10**5))*(10**6) #calcul de la distance lumineuse avec les paramètres cosmologiques correspondant au flat wCDM (Brout et al. 2022 = Analysis on cosmological constraints)
    result = spi.quad(f,0,zHD[i])                                               #idem
    mu_theory = 5*np.log10(((1+zHD[i])*result[0])/10)                              

    if CEPH_DIST[i] == -9.0 : #on vérifie si la mesure est relié à la mesure d'une distance avec une Cepheid (CEPH_DIST[i] == -9.0 signifie que ce n'est pas le cas)
        mu = mu_shoes-mu_theory
        DeltaD.append(mu)
    else :
        mu = mu_shoes-mu_cepheid
        DeltaD.append(mu)
#print(DeltaD)
#on dispose du deltaD de la formule de la likelihood (formule (14) Brout et al. 2022) -> on peut essayer de voir si le résultat n'est pas aberrant en affichant les distance residuals en fonction du redshift et en comparant avec la fig 4 (il faudrait trouver un autre moyen de vérifier)
"""
plt.scatter(zHD,DeltaD)
plt.xscale('log')
plt.ylabel('modulus distance residuals')
plt.xlabel('redshift')
plt.legend()
plt.show()
"""
#on calcule la transposée
DeltaD_transpose = np.transpose(DeltaD)

#on s'attaque à la matrice de covariance
with open('Pantheon+SH0ES_STAT+SYS.txt') as file:
    data = [line.strip() for line in file]
data = np.array(data) ; data = data.astype(float)
matcov_SN_Cepheid = [data[i:i+1701] for i in range(0, 1701**2, 1701)] #on a les colonnes (ou les lignes) de notre matrice de covariance -> Comment faire pour savoir si un élément de l'array représente
                                                                      #une ligne ou une colonne ? -> pose problème pour le calcul matriciel par la suite ; Pour éviter ce problème, pour l'instant on ne 
                                                                      #récupère que la diagonale
matcov_SN_Cepheid_diag = np.zeros((1701,1701))
for i in range(1701):
    matcov_SN_Cepheid_diag[i][i] = matcov_SN_Cepheid[i][i]


#on calcule la likelihood ; ⚠️-> ici on le fait avec matcov_SN_Cepheid_diag et non la matrice de covariance entière (risque d'imprécision)
A = np.dot(DeltaD_transpose,np.linalg.inv(matcov_SN_Cepheid_diag))
likelihood = np.dot(A,DeltaD) ; print(likelihood)

#maintenant on va tenter de tracer le diagramme (Omegam,Omegalambda) en faisant varier ces paramètres et trouver le minimum de la likelihood




