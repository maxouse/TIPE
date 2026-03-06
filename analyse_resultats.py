from matplotlib import pyplot as plt
import numpy as np
import math
from sympy import symbols,Eq,solve,Matrix,solve_linear_system
from copy import deepcopy
from sympy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import time
from colorsys import rgb_to_hsv
from numba import jit
from skimage.color import rgb2hsv,hsv2rgb
from PIL import Image
import random as rd
from math import sin,exp,pi,cos
import cv2
import os

from main import Dov,Dor,Dvr,D3,D,D2,L_bruit_ov,L_bruit_ro,L_bruit_vr
import parametres_et_fonctions


from parametres_et_fonctions import Nu, max_dist, fichier_resolutions 
#%% Graphiques des performances


plt.hist(Dov,bins=100,color='r')
plt.xlabel( 'distance vert/orange en mm d\'après l\'algorithme' )
plt.ylabel('nombre d\'occurences sur 300 images')
plt.show()

plt.hist(Dor,bins=100,color='b')
plt.xlabel( 'distance rose/orange en mm d\'après l\'algorithme' )
plt.ylabel('nombre d\'occurences sur 300 images')
plt.show()


plt.hist(Dvr,bins=100,color='m')
plt.xlabel( 'distance vert/rose en mm d\'après l\'algorithme' )
plt.ylabel('nombre d\'occurences sur 300 images')
plt.show()

##
Nim=[i+1 for i in range (len(D3))]
plt.plot(Nim,D3,'m')
plt.ylabel('distance rose/vert')
plt.xlabel('nombre images')
plt.show()

Nim=[i+1 for i in range (len(D))]
plt.plot(Nim,D,'b')
plt.ylabel('distance orange/rose')
plt.xlabel('nombre images')
plt.show()

Nim=[i+1 for i in range (len(D2))]
plt.plot(Nim,D2,'r')
plt.ylabel('distance vert/orange')
plt.xlabel('nombre images')
plt.show()

#valeurs réelles

d_roseorange=0.47
d_rosevert=0.47
d_orangevert=0.55

#différences entre moyenne des distances et valeurs réelles

d=parametres_et_fonctions.distance_1D(d_roseorange,parametres_et_fonctions.Esperance(D))
d2=parametres_et_fonctions.distance_1D(d_rosevert,parametres_et_fonctions.Esperance(D3))
d3=parametres_et_fonctions.distance_1D(d_orangevert,parametres_et_fonctions.Esperance(D2))

# distance maximale aux vraies valeurs, pourcentages d'erreur

print('L_o/r'+str(Nu)+'p=',[max_dist(d_roseorange,D),d,d*100/d_roseorange])
print('L_v/r'+str(Nu)+'p=',[max_dist(d_rosevert,D3),d3,d3*100/d_rosevert])
print('L_v/o'+str(Nu)+'p=',[max_dist(d_orangevert,D2),d2,d2*100/d_orangevert])


a=max(L_bruit_ro)-1
b=max(L_bruit_ov)-1
c=max(L_bruit_vr)-1
print(max(a,b,c))

#données pour orange/rose
L_ecartsmax=[0.0275756439255100,0.0370658672344418,0.0436051014086934,0.0371272078399870,0.0412581047638436,0.109971267411584,0.104600096134721,0.125225847508093,0.128796163013388]

L_ecartsmaxrel=[5.87,7.89,9.28,7.90,8.78,23.40,22.26,26.64,27.43]

L_ecartsmoyrel=[0.793993867540580,0.890223792241604,1.24255106073279,0.930403795882113,0.866489423084205,0.882052016477336,1.12617482345757,1.40388588722289,2.24692478009537]

L_ips=[1.5073751865798632,6.424389456193411,13.40594451663873,23.374431219222274,32.83816554049817,42.30890019971321,50.429181753708995,91.04832449061044,106.87074323825387]


n=len(L_ips)

L_resolutions=[str(fichier_resolutions[i][0])+'p' for i in range(n)]

#figure ecart max tol =0.5
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(L_resolutions,L_ips,'r')
ax2.plot(L_resolutions,L_ecartsmaxrel,'b')


ax1.set_xlabel('résolution')
ax1.set_ylabel('ips', color='r')
ax2.set_ylabel('ecart maximal à la valeur réelle pour orange-rose en %', color='b')
plt.title('compromis précision/vitesse execution')

plt.show()

#figure ecart moyen tol =0.5
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(L_resolutions,L_ips,'r')
ax2.plot(L_resolutions,L_ecartsmoyrel,'y')


ax1.set_xlabel('résolution')
ax1.set_ylabel('ips', color='r')
ax2.set_ylabel('ecart moyen à la valeur réelle pour orange-rose en %', color='y')
plt.title('compromis précision/vitesse execution')

plt.show()

#figure ecart max tol =0.005

L_ips=[1.54,3.514318700283023,8.47004546680799,15.260208258451863,27.928300702172535,46.236976272122455,65.77244593110401,84.6450003488585,100.83468860790934,122.43986157156715,135.2453174241777,154.496851007356,163.31435819992606,177.52470381879547,187.3099564891049,206.6430648151914,216.4905702283941,219.63159435930294,223.60717049777992,234.29385744796286]

n=len(L_ips)

L_resolutions=[str(parametres_et_fonctions.fichier_resolutions[i][0])+'p' for i in range(n)]

L_ecartsmaxrel=[6.09131035762678,6.04929489432385,6.90551730576629,6.94072122808613,7.15754983810691,7.72346363324589,7.29689075308297,7.72777014920023,7.73837240091691,7.54209886393189,8.16335863700819,8.58077912848281,9.21737723328189,16.494127533761,21.7038773467608,15.1970476681910,8.87159895313290,7.79661743207307, 6.65896698480176,40.9133433638093]

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(L_resolutions,L_ips,'r')
ax2.plot(L_resolutions,L_ecartsmaxrel,'b')


ax1.set_xlabel('résolution')
ax1.set_ylabel('ips', color='r')
ax2.set_ylabel('ecart maximal à la valeur réelle pour orange-rose en %', color='b')

plt.title('compromis précision/vitesse execution')

plt.show()

#figure bruitmax

bruitmax=[0.01834237000036,0.0155401385218237,0.0154907937764626,0.0208052907994616,0.0195070026299098,0.0247092877403305,0.0253002619876896,0.0286730208790820,0.0289504558684599,0.0354982318417281,0.0279569464609604,0.0314541514238356]

bruitmaxnul=[0.0226083961399395,0.0266925507964959,0.0275472183637204,0.0340578775898850,0.0306190442731984,0.0540662557546989,0.0427985037803640,0.0585019139838563,0.0490930569139973,0.0638293087344957,0.0575748711010644,0.0622140082284093]

n=len(bruitmax)

L_resolutions=[str(parametres_et_fonctions.fichier_resolutions[i][0])+'p' for i in range(n)]


plt.plot(L_resolutions,bruitmax,label='bruit maximal en %')
plt.xlabel('résolution')
plt.legend()
plt.show()

plt.plot(L_resolutions,bruitmaxnul ,label='bruit maximal en %')
plt.xlabel('résolution')
plt.legend()
plt.show()

