# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 17:25:55 2026

@author: maxen
"""

from matplotlib import pyplot as plt
import numpy as np
import random as rd

#%%calcul distance focale
def loi_normale(moy,sigma,nombre_tirages) :
    return np.random.normal(moy,sigma,nombre_tirages)

# valeurs des paramètres et leurs incertitudes

def focale(Y,oa) :
    return oa/(1+1/Y)

N = 10**6
Y=0.00549
oa=1000
uoa=5
Doa = loi_normale(oa,uoa, N)


# on calcule la focale pour chaque élément des distributionS

Dfocale = [focale(Y,Doa[i]) for i in range(N)]

# valeur mesurée et incertitude

focale = np.mean(Dfocale)
ufocale = np.std(Dfocale)

print('la focale est :', focale , '+-', 2*ufocale,'mm')

# représentation graphique

plt.hist(Dfocale,bins=100)
plt.show()