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

import parametres_et_fonctions

from parametres_et_fonctions import f, ku , cu , cv , kv, tax , tay, taz ,tbx , tby ,tbz 

#%% Parcours des images et détection des points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Then use it like this:
IMAGES_DIR = os.path.join(BASE_DIR, "images3")
IM1 = plt.imread(os.path.join(IMAGES_DIR, "immax_000.png"))

m=np.max(IM1,axis=1)


# Affiche(1,IM1)
#
# IM2=plt.imread('C:\\Users\\Maxence\\Desktop\\TIPE\\images3\\imcha_000.png')
#
# Affiche(1,IM2)

##execution cM2,cL2

#2eme caméra
listfiles = [ "imcha2_"+str(parametres_et_fonctions.Nu)+'p'+str(n).zfill(3)+".png" for n in range(300)]
h_refvert = 0.3662173202614379
h_reforange = 0.04730617608409987
h_refrose = 0.9230457675172135
L_hue=[h_refvert,h_reforange,h_refrose]
Lpix1=[[] for i in range(len(L_hue))]
tol = 0.005

ti=time.perf_counter()

for filename in listfiles :

    fn  = os.path.join(IMAGES_DIR, filename)
    pixels = plt.imread(fn)[:,:,:3]
    pixvert,pixorange,pixrose= parametres_et_fonctions.detection(pixels,L_hue)

    Lpix1[0].append(pixvert)
    Lpix1[1].append(pixorange)
    Lpix1[2].append(pixrose)


#1ere caméra
listfiles = [ "immax"+str(parametres_et_fonctions.Nu)+'p'+str(n).zfill(3)+".png" for n in range(300) ]
Lpix2=[[] for i in range(len(L_hue))]
for filename in listfiles :

    fn = os.path.join(IMAGES_DIR, filename)

    pixels = plt.imread(fn)[:,:,:3]
    pixvert,pixorange,pixrose= parametres_et_fonctions.detection(pixels,L_hue)

    Lpix2[0].append(pixvert)
    Lpix2[1].append(pixorange)
    Lpix2[2].append(pixrose)


tf=time.perf_counter()
print(tf-ti,'temps total')
ips=len(listfiles)/(tf-ti)
print(ips,'ips')
#print(Lpix1,Lpix2)

##résolution système

n=len(Lpix1[0]) #=len(Lpix2[0])
k=len(L_hue) #nombre de couleurs
Lpos=[[]for i in range(k)]
for i in range(n):
    for j in range(k):
        u_1=Lpix1[j][i][0]
        v_1=Lpix1[j][i][1]
        u_2=Lpix2[j][i][0]
        v_2=Lpix2[j][i][1]

        #le système peut aussi s'écrire AX=B

        A=np.array([[f*ku,cu-u_1,0],[0,cv-v_1,-f*kv],[cu-u_2,-f*ku,0],[cv-v_2,0,-f*kv]])

        B=np.array([[f*ku*tax+(cu-u_1)*tay],[-f*kv*taz+(cv-v_1)*tay],[-f*ku*tby+tbx*(cu-u_2)],[tbx*(cv-v_2)-f*kv*tbz]])


        #la meilleure solution est la solution de At*A*X=At*B

        At=A.transpose()

        S=np.dot(At,A)
        R=np.dot(At,B)

        # solution de Sx=R

        res=np.linalg.solve(S,R).tolist()

        '''pour transformer tuples en listes'''
        pos=[]
        for l in range(3):
            pos.append(res[l][0])

        Lpos[j].append (pos)

# Lpos=Lissage(Lpos)

##Calcul distance entre points
vert=Lpos[0]
orange=Lpos[1]
rose=Lpos[2]

D=parametres_et_fonctions.distance_3D(orange,rose)


D2=parametres_et_fonctions.distance_3D(orange,vert)

Dov=[]
for d in D2:
    d=round(d,3)
    d*=1000
    d=int(d)
    Dov.append(d)


Dor=[]
for d in D:
    d=round(d,3)
    d*=1000
    d=int(d)
    Dor.append(d)


D3=parametres_et_fonctions.distance_3D(vert,rose)


Dvr=[]
for d in D3:
    d=round(d,3)
    d*=1000
    d=int(d)
    Dvr.append(d)

n=len(D)

#listes des valeurs du bruit

L_bruit_ro=[D[i+1]/D[i]for i in range(n-1)]

L_bruit_ov=[D2[i+1]/D2[i]for i in range(n-1)]

L_bruit_vr=[D3[i+1]/D3[i]for i in range(n-1)]




#%% affichage 3D du déplacement des points cibles 
import matplotlib.animation as animation

def AffichePathsV2(l1, l2, l3, tail_length, colors):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    data = [np.array(l1), np.array(l2), np.array(l3)]

  
    all_pts = np.vstack(data)
    ax.set_xlim3d([all_pts[:,0].min(), all_pts[:,0].max()])
    ax.set_ylim3d([all_pts[:,1].min(), all_pts[:,1].max()])
    ax.set_zlim3d([all_pts[:,2].min(), all_pts[:,2].max()])

    scatters = [ax.scatter([], [], [], c=np.array([c])/255, s=20) for c in colors]

    def update(num):
        for idx, sc in enumerate(scatters):
            start = max(0, num - tail_length)
            pts = data[idx][start:num+1]
            
            if len(pts) > 0:
                sc._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
                

        return scatters

    ani = animation.FuncAnimation(fig, update, frames=len(l1), interval=30, blit=False)
    plt.show()
    return ani 

colors_rgb = [(0, 255, 0), (255, 128, 0), (255, 102, 178)]
ani = AffichePathsV2(Lpos[0], Lpos[1], Lpos[2], 5, colors_rgb)

#%% Autre méthode : tracking des points en regardant les points proches du précédent 
# approche moins performante mais intéressante.

# startpoint = (1700,210)  # c,l
# Lpix=[]
# hue_ref = 0.423 # correspond à peu près au vert sur la première image point 1700,210
# tol = 0.005 #tolerance autour de hue_ref
#
#
# listfiles = [ "vert"+str(n).zfill(5)+".png" for n in range(1,300,20) ]
# # listfiles = [ "vertlou"+str(n).zfill(5)+".png" for n in range(1,1150,20) ] # il faut redefinir le point de départ
#
#
#
# for filename in listfiles :
#     t0=time.perf_counter()
#
#     fn = "C:\\Users\\Maxence\\Desktop\\TIPE COMPLETEMENT FUMAX\\images\\"+filename
#     im1 = Image.open(fn)
#     pixels = np.array(im1.getdata(), np.uint8).reshape(im1.size[1], im1.size[0], 3)
#
#     t1=time.perf_counter()
#     print(t1-t0,'pixels')
#
#     # idee : on ne garde que la teinte (hue) donc on normalise la saturation/vivacité et valeur/luminosité
#
#     # passage de rgb à hsv
#
#     t2=time.perf_counter()
#
#
#     m1 = np.max(pixels,axis=2)
#     m2 = np.min(pixels,axis=2)
#
#     hue = ( m1==pixels[:,:,0] ) * (pixels[:,:,1]-pixels[:,:,2])/(m1-m2)
#     + ( m1==pixels[:,:,1] ) * (2+(pixels[:,:,2]-pixels[:,:,0])/(m1-m2))
#     ( m1==pixels[:,:,2] ) * (4+(pixels[:,:,0]-pixels[:,:,1])/(m1-m2))
#
#     hue = hue / 6 % 1
#
#     pixels2 = 255* np.ones(pixels[:,:,0].shape)  *  (hue_ref-tol< hue ) * (  hue <hue_ref+tol )
#
#     t3=time.perf_counter()
#     print(t3-t2,'pixels1 + hsv')
#
#
#     hsv = rgb2hsv ( pixels )
#     hsv[:,:,1] = 1
#     hsv[:,:,2] = 1
#     # retour à une image rgb (0-255)
#     pixels1 = 255 * hsv2rgb(hsv)
#
#     t3=time.perf_counter()
#     print(t3-t2,'pixels1 + hsv')
#
#     # detection des points verts
#     t4=time.perf_counter()
#
#     npixdetected = 0
#     tolpix = tol
#
#     while npixdetected == 0 :
#         pixels2 = 255* np.ones(pixels[:,:,0].shape)  *  ((hue_ref-tolpix)< hsv[:,:,0] ) * (  hsv[:,:,0] <(hue_ref+tolpix) )
#         npixdetected = int( sum(sum(pixels2))/255 )
#
#         print(npixdetected,"pixels verts détectés avec la tolérance",tolpix)
#
#         tolpix=tolpix*2
#
#     t5=time.perf_counter()
#     print(t5-t4,'detection')
#
#     t3=time.perf_counter()
#     print(t3-t2,'pixels1 + hsv')
#
#     '''' j'ai besoin d'une fonction décroissante et positive de la distance euclidienne à startpoint, 1/(1+x^2+y^2) est parfait '''
#
#
#     t6=time.perf_counter()
#
#     y = startpoint[0]
#     x = startpoint[1]
#     sx = pixels2.shape[0]
#     sy = pixels2.shape[1]
#     xx = np.linspace(0,sx-1,sx).reshape(-1,1)
#     yy = np.linspace(0,sy-1,sy).reshape(1,-1)
#     cone = 1/(1 + (x-xx)**2 + (y-yy)**2)
#
#
#     t7=time.perf_counter()
#     print(t7-t6,'cone')
#
#     Affiche(1,10*cone)
#
#     t8=time.perf_counter()
#     distribution = cone * pixels2 # le max de cette fonction est au point vert le plus proche
#     t9=time.perf_counter()
#     print(t9-t8,'distribution')
#
#
#
#     newpoint = indexmax(distribution)
#     Lpix.append(newpoint)
#     print ("point vert le plus proche :", newpoint)
#     startpoint = newpoint
#
#
#
#     '''création images'''
#
#     pixels3= pixels2.reshape(( pixels2.shape[0],pixels2.shape[1],1))*np.ones(( pixels2.shape[0],pixels2.shape[1],3))
#
#     leftb = max(0,newpoint[1]-5)
#     rightb = min(pixels2.shape[0]-1 ,newpoint[1]+5)
#     topb = min(pixels2.shape[1]-1 ,newpoint[0]+5)
#     botb = max(0,newpoint[0]-5)
#     pixels3[ leftb:rightb,botb:topb,: ] = (255,0,0) #rouge
#
#
#
#
#     '''sauvegarde des images'''
#
#     f_out = "C:\\Users\\Maxence\\Desktop\\TIPE COMPLETEMENT FUMAX\\images-hue\\" + filename
#     im = Image.fromarray(np.array(pixels1, dtype='uint8'))
#     im.save(f_out, "png")
#     print ("- ",f_out, "saved.")
#
#     f_out2 = "C:\\Users\\Maxence\\Desktop\\TIPE COMPLETEMENT FUMAX\\images-detect\\" + filename
#     im2 = Image.fromarray(np.array(pixels3, dtype='uint8'))
#     im2.save(f_out2, "png")
#     print ("- ",f_out2, "saved.")



