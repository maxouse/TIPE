# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 17:21:32 2026

@author: maxen
"""

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



#%%paramètres
'''données '''

fichier_resolutions=[(int(1920/i),int(1080/i)) for i in range (1,21) ]

Nu=fichier_resolutions[15][0]
Nv=fichier_resolutions[15][1]

#rapport Nl originel sur Nl modifié
Ru=Nu/1920
Rv=Nv/1080

'''distance focale'''

f=5.4*(10**-3)

'''centre image'''

cu=960*Ru
cv=540*Rv

#nombre de pixel/m du capteur
'''capteur 6.3x3.5 mm^2'''
'''1920/6.3,1080/3.5'''

ku=305*(10**3)*Ru
kv=309*(10**3)*Rv


'''première caméra,cha'''

tax=0
tay=0
taz=0.85

'''deuxieme caméra'''

tbx=-1.34
tby=1.39
tbz=0.85

tol = 0.005

h_refvert = 0.3662173202614379
h_reforange = 0.04730617608409987
h_refrose = 0.9230457675172135
#%%fonctions
def Esperance(X):           #loi uniforme
    n=len(X)
    E=0
    for x in X:
        E+=x/n
    return E

def Ecart_type(X):
    n=len(X)
    X_2=[x**2 for x in X]
    std=sqrt(Esperance(X_2)-Esperance(X)**2)
    return std


def Affiche(fig,im):
    plt.figure(fig)
    plt.imshow(im)
    plt.axis('off')
    plt.show()


'''ouvre images '''
def ouverture(image,nmin,nmax,pas=1):
    for i in range (nmin,int(nmin+(nmax-nmin)/pas)):
        num=str(i+(i-nmin)*pas).zfill(3)
        scene=plt.imread('C:\\Users\\Maxence\\Desktop\\TIPE\\images3\\'+str(image)+num+'.png')
        Affiche(1,scene)

@jit
def moyenne_distance(D):#distance moyenne d'une liste de distances
    m=0
    for d in D:
        m+=d
    m=m/len(D)
    return m


'''affiche l'image avec la teinte plutot que en RGB''' 

def Affiche_hue(im):
    pixels=plt.imread(im)

    m1 = np.max(pixels,axis=2)
    m2 = np.min(pixels,axis=2)

    eps=10**(-10) #pour eviter l'erreur m1=m2

    hue = ( m1==pixels[:,:,0] ) * (0.0+pixels[:,:,1]-pixels[:,:,2])/(m1-m2+eps)
    hue += ( m1==pixels[:,:,1] ) * (2.0+(pixels[:,:,2]-pixels[:,:,0])/(m1-m2+eps))
    hue += ( m1==pixels[:,:,2] ) * (4.0+(pixels[:,:,0]-pixels[:,:,1])/(m1-m2+eps))

    hue = ((hue / 6))% 1

    Affiche(1,hue)

''' à partir de deux listes de meme longueur, renvoie une liste des distances de chaque element de L1 a chaque element de l2, en 3D'''

def distance_3D(L1,L2):
    n=len(L1)
    L_d=[]
    for i in range(n):
        x1,y1,z1=L1[i][0],L1[i][1],L1[i][2]
        x2,y2,z2=L2[i][0],L2[i][1],L2[i][2]
        d=(x1-x2)**2+(y1-y2)**2+(z1-z2)**2
        d=sqrt(d)
        L_d.append(d)
    return L_d

"""en 2D"""
def distance_2D(L1,L2):
    n=len(L1)
    L_d=[]
    for i in range(n):
        x1,y1=L1[i][0],L1[i][1]
        x2,y2=L2[i][0],L2[i][1]
        d=float((x1-x2)**2+(y1-y2)**2)
        d=sqrt(d)
        L_d.append(d)
    return L_d

def distance_1D(d1,d2):
    return abs(d1-d2)

#renvoie la valeur maximale et le pourcentage de la difference entre la valeur de reference et chaque valeur de L

def max_dist(ref,L):
    M=distance_1D(ref,L[0])
    for x in L:
        d=distance_1D(ref,x)
        if d>M:
            M=d
    return M,M*100/ref



@jit
def carre_noir(pixels,pix,taille):
    x,y=pix[1],pix[0]

    t=int(taille*Ru/2)

    pixels[x-t:x+t,y-t:y+t]=np.array(0.6)#remplace la section de pixels par un carré noir

    return pixels

@jit
def carre_noir_arrayRGB(pixels,pix,taille):#hue[64,89,128] donne 0.6 , valeur loin du rose orange et vert
    x,y=pix[1],pix[0]

    t=int(taille*Ru/2)

    pixels[x-t:x+t,y-t:y+t,:]=np.array([64/255,89/255,128/255]) #remplace la section de pixels par un carré noir

    return pixels

def centre_couleur_vectoriel(pixhue) :
    nl,nc = pixhue.shape
    c=[0,0]

    # en vectoriel : xs = pixhue ou je remplace les 1 par la valeur de x ; meme chose pour ys avec y
    xs = pixhue * np.arange(nl).reshape((nl,1))*np.ones((nl,nc))
    ys = pixhue * np.arange(nc).reshape((1,nc))*np.ones((nl,nc))

    # je n'ai plus qu'a prendre la moyenne, en ne tenant pas compte des 0 : je fais un masque
    c[1] =  int( np.ma.masked_equal(xs, 0).mean() )
    c[0] =  int( np.ma.masked_equal(ys, 0).mean() )

    return tuple(c)



''' renvoie les points de la bonne couleur dans pixels ''' #carré noir si pix orange

def detection(pixels,L_hue):

        # array ne contenant que la plus haute valeur de r,g,b (entre 0 et 1 ici)

        m1 = np.max(pixels,axis=2)
        m2 = np.min(pixels,axis=2)

        eps=10**(-10) #pour eviter l'erreur m1=m2

        hue = ( m1==pixels[:,:,0] ) * (0+pixels[:,:,1]-pixels[:,:,2])/(m1-m2+eps)

        hue += ( m1==pixels[:,:,1] ) * (2+(pixels[:,:,2]-pixels[:,:,0])/(m1-m2+eps))

        hue += ( m1==pixels[:,:,2] ) * (4+(pixels[:,:,0]-pixels[:,:,1])/(m1-m2+eps))

        hue = ((hue / 6))% 1
        Lpix=[]

        for h in L_hue:
            hue_ref=h
            npixdetected = 0
            tolpix = tol
            while npixdetected == 0 :

                pixels2 =np.ones(pixels[:,:,0].shape)  *  ((hue_ref-tolpix)< hue ) * (  hue <(hue_ref+tolpix) )

                npixdetected = sum(sum(pixels2))
                tolpix=tolpix*2

            pix= centre_couleur_vectoriel(pixels2)



            if h==h_reforange or h==h_refvert:
                hue=carre_noir(hue,pix,200)
            # # #
            # Affiche(1,pixels2)
            # Affiche(1,hue)

            Lpix.append(pix)


        return Lpix


def Lissage(Lpos):
    k=len(Lpos)
    n=len(Lpos[0])
    liss=[[] for i in range(k)]
    for j in range(k):
        for i in range(n-1):
            x=(Lpos[j][i][0]+Lpos[j][i+1][0])/2
            y=(Lpos[j][i][1]+Lpos[j][i+1][1])/2
            z=(Lpos[j][i][2]+Lpos[j][i+1][2])/2
            liss[j].append([x,y,z])
        liss[j].append(Lpos[j][n-1])
    return liss



def h(L_RGB):
    R,G,B=L_RGB
    hue=rgb_to_hsv(R,G,B)[0]
    return hue

##anciennes fonctions
#renvoie le pixel moyenne des positions des pixels d'un array a 2 dimensions dont les valeurs sont 0 ou 1

def centre_couleur(pixhue):
    nl,nc=pixhue.shape
    c=[0,0]
    compteur=0
    for i in range (nl):
        for j in range(nc):
            if pixhue[i,j]==1:
                compteur+=1
                c[0]+=j
                c[1]+=i
    c[0]=int(c[0]/compteur)
    c[1]=int(c[1]/compteur)
    return (c[0],c[1])

def Couleur_dif(image):
    T=[]
    im=np.copy(image)
    Nl,Nc=im.shape[:2]
    for i in range (Nl):
        for j in range (Nc):
            R,G,B=im[i,j]
            if G>2*B and G>2*R:
                im[i,j]=1,1,1
    return im

'''une fois puis s'arrete'''
def Couleur_dif_inst(image):
    T=[]
    im=np.copy(image)
    Nl,Nc=im.shape[:2]
    for i in range (Nl):
        for j in range (Nc):
            R,G,B=im[i,j]
            if G>2*B and G>2*R:
                im[i,j]=1,1,1
                return im,i,j

def recuperation_indices(image,n):
    resultat=[]
    for i in range (n):
        numero=1+10*i
        num='0000'+ str(2*numero-1)
        while len(num)>5:
            num=num[1:]
        scene=plt.imread('C:\\Users\\Maxence\\Desktop\\TIPE\\images\\'+str(image)+num+'.png')
        a=Couleur_dif_inst(scene)
        if a!=None:
            i,j,t=a
        resultat.append((i,j))
    return resultat


def Test_point(im,l,c) :
    (R,G,B)=im[l,c]
    if  (G>=2*B and G>=2*R) :
        return (True, l, c)
    else :
        return (False, 0,0)


#animation test 
#
# from matplotlib import animation
#
# N=len(Lpos)-1
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# def update(num, data, line):
#     line.set_data(data[:2, :num])
#     line.set_3d_properties(data[2, :num])
#
# data=np.array(Lpos).T
# line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
# ax.set_xlabel('X')
# ax.set_xlim3d([-1,1.5])
# ax.set_ylabel('Y')
# ax.set_ylim3d([-1, 1.0])
# ax.set_zlabel('Z')
# ax.set_zlim3d([-0.5, 0.6])
#
# ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False)
#
# # plt.show()
#
# # test ligne droite:
# Lpixlou=[(0,0) for i in range (100)]
# Lpix=[(15*i,500)for i in range (100)]
#

#trouve le point de la bonne couleur en faisaint une spirale autour du point (i,j)
def Couleur_dif_inst_suite(im,i,j,Nl,Nc):
    pas=1
    check = (False,0,0)

    while check[0]==False and pas<2000:
        for l in range(max(0,i-pas),min(i+pas+1,Nl)):
            c=j-pas
            if c > 0:
                check = Test_point( im,l,c )

                if check[0] :

                    return pas,check[1],check[2],t1-t0

            c=j+pas
            if c < Nc:
                check = Test_point( im,l,c )

                if check[0] :

                    return pas,check[1],check[2],t1-t0

        for c in range (max(j-pas,0),min(j+pas+1,Nc)):
            l = i-pas
            if l > 0:
                check = Test_point( im,l,c )

                if check[0] :

                    return pas,check[1],check[2],t1-t0

            l=i+pas
            if l < Nl:
                check = Test_point( im,l,c )

                if check[0] :

                    return pas,check[1],check[2],t1-t0

        pas+=1


'''return l'indice c,l de la valeur maximum d'un array'''
@jit
def indexmax(array) :

    indexmax = np.argmax(array)

    width = array.shape[1]

    l=int(indexmax/width)
    c=indexmax-l*width   #division euclidienne

    return  (c,l)