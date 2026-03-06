# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 17:23:48 2026

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
from resizeimage import resizeimage

import parametres_et_fonctions

#%%traitement video

def decoupe_video(chemin,chemin_arrivee, basename, ext='png'):
    cap = cv2.VideoCapture(chemin)

    if not cap.isOpened():
        return "Impossible d'ouvrir la video"

    os.makedirs(chemin_arrivee, exist_ok=True)

    ch = os.path.join(chemin_arrivee, basename)

    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n = 0

    while True:
        ret, Image = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(ch, str(n).zfill(N), ext), image)
            n += 1
        else:
            return

# =============================================================================
# decoupe_video('C:\\Users\\Maxence\\Desktop\\TIPE\\videos3\\vidcha.mp4', 'C:\\Users\\Maxence\\Desktop\\TIPE\\images3', 'imcha')
# decoupe_video('C:\\Users\\Maxence\\Desktop\\TIPE\\videos3\\vidmax.mp4', 'C:\\Users\\Maxence\\Desktop\\TIPE\\images3', 'immax')
# =============================================================================


#modifier taille images cam1

fichier_resolutions=[[int(1920/i),int(1080/i)] for i in range (1,21) ]
for reso in fichier_resolutions:
    for filename in range(346):

        f_in = os.path.join("images3", f"immax_{str(filename).zfill(3)}.png")

        img=Image.open(f_in)

        img = resizeimage.resize_thumbnail(img,reso)

        save_path = os.path.join("images3", f"immax{reso[0]}p{str(filename).zfill(3)}.png")
        img.save(save_path, 'png')

#modifier taille images cam2
fichier_resolutions=[[int(1920/i),int(1080/i)] for i in range (1,21) ]
for reso in fichier_resolutions:
    for filename in range(336):

        f_in = 'C:\\Users\\Maxence\\Desktop\\TIPE\\images3\\imcha2_'+str(filename).zfill(3)+".png"

        img=Image.open(f_in)

        img = resizeimage.resize_thumbnail(img,reso)

        img.save('C:\\Users\\Maxence\\Desktop\\TIPE\\images3\\imcha2_'+str(reso[0])+'p'+ str(filename).zfill(3)+".png",'png')
    print('imcha'+str(reso[0])+ ': processed')