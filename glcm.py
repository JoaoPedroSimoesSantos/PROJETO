# -*- coding: utf-8 -*-
"""
Created on Sun May 06 19:28:14 2018

@author: Michael
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

from skimage.feature import greycomatrix, greycoprops
from skimage import data
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(8, 8))
PATCH_SIZE = 8
print "PATCH_SIZE", PATCH_SIZE

# img = cv.imread("images/seagull_database_vis001_small.png")
img = cv.imread("images/Frame120.jpg")

res = cv.resize(img,None,fx=0.25, fy=0.25, interpolation = cv.INTER_CUBIC)
print "Original", img.shape
print "Resized", res.shape
print 
# res = cv.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
# print "SHAPE", res.shape
gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
lab = cv.cvtColor(res,cv.COLOR_BGR2LAB)
#
##cv.imshow("gray",gray)
##cv.waitKey(0)
##cv.destroyAllWindows()
#
# select some patches from water areas of the image

# ###Imagem 1
# # water_locations = [(10, 10),(200,200),(50,200),(110,220),(240,350),(110,200)]
# water_locations = [(5, 5),(100,100),(25,100),(50,110),(120,175),(50,95)]
# print "Imagem 1"

# ### Imagem vis001_01
# water_locations = [(110,225)]
# print "Imagem 1_01"
# print gray.shape

# ##Imagem 2
# water_locations = [(10, 10),(240,350),(50,200),(150,230),(150,280)]
# print "Imagem 2"
# print gray.shape

# ###Frame 61
# water_locations = [(10, 10),(510,200),(200,200),(520,715),(110,220)]
# print "Frame 61"

# ###Frame 61 SMALL
# water_locations = [(10, 10),(200,230),(50,200),(255,355),(70,100)]
# print "Frame 61"
# print gray.shape

# ### Image vis005
# water_locations = [(1,174)]
# print "Imagem 5"
# print gray.shape

### Image vis006
# water_locations = [(120,170),(50,20),(50,250)]
# print "Imagem 6"
# print gray.shape

# ###Frame 600
# # water_locations = [(105,130),(25,10),(25,125)]
# water_locations = [(220,260),(40,20),(50,250)]
# # water_locations = [(440,540),(100,40),(100,500)]
# print "Frame 600"
# # print gray.shape

# ###Frame 1200
# water_locations = [(290,610),(335,300),(50,250)]
# print "Frame 1200"
# print gray.shape

# ### Frame 4518 salvamento
# # water_locations = [(115,220),(35,150),(100,275)]
# water_locations = [(230,440),(70,300),(200,550)]
# print "Frame 4518"
# print gray.shape

# ## Frame 3731 salvamento
# # water_locations = [(90,150),(50,380),(200,550)]
# # water_locations = [(45,75),(25,170),(100,275)]
# water_locations = [(20,32),(10,80),(50,130)]
# # water_locations = [(8,15),(5,40),(25,60)]
# print "Frame 3731"
# print gray.shape

# ## Frame 3580 salvamento
# # water_locations = [(85,220),(250,300),(100,275)]
# # water_locations = [(40,110),(125,150),(50,275)]
# water_locations = [(20,55),(60,70),(20,130)]
# # water_locations = [(8,25),(30,35),(10,60)]
# print "Frame 3580"
# print gray.shape

# ## Frame 3602 salvamento
# # water_locations = [(85,220),(250,300),(100,275)]
# # water_locations = [(40,110),(125,150),(50,275)]
# water_locations = [(10,50),(60,70),(20,130)]
# # water_locations = [(8,25),(30,35),(10,60)]
# print "Frame 3602"
# print gray.shape

# ## Frame 4729 salvamento
# # water_locations = [(180,300),(35,150),(100,275)]
# # water_locations = [(80,90),(145,340),(200,550)]
# water_locations = [(35,40),(65,165),(100,275)]
# print "Frame 4729"
# print gray.shape

# ## Frame 4514 salvamento
# # water_locations = [(65,305),(145,340),(180,560)]
# # water_locations = [(30,150),(70,170),(90,280)]
# water_locations = [(10,50),(27,152),(70,170),(90,280)]
# # water_locations = [(4,24),(16,75),(35,80),(45,140)]
# print "Frame 4514"
# print gray.shape

# ## Frame 1 salvamento
# # water_locations = [(90,150),(50,380),(200,550)]
# # water_locations = [(45,75),(25,170),(100,275)]
# water_locations = [(0,35),(10,80),(114,184)]
# # water_locations = [(8,15),(5,40),(55,90)]
# print "Frame 1"
# print gray.shape

# ## Frame 121 salvamento
# # water_locations = [(0,35),(81,248),(140,60)]
# water_locations = [(20,70),(36,120),(68,35)]
# print "Frame 121"
# print gray.shape

## Frame 120 salvamento
# water_locations = [(0,35),(81,248),(140,60)]
water_locations = [(64,135),(71,135),(71,127)]
print "Frame 120"
print gray.shape

# ## Frame 675 salvamento
# # water_locations = [(0,35),(81,248),(140,60)]
# water_locations = [(162,70),(36,120),(135,32)]
# print "Frame 675"
# print gray.shape


water_patches = []
for loc in water_locations:
    water_patches.append(gray[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

xs = []
ys = []
zs = []
for patch in (water_patches):
    glcm = greycomatrix(patch, [1], [3*np.pi/4], symmetric=True, normed=True)
    # print greycoprops(glcm, 'contrast')
    zs.append(greycoprops(glcm, 'contrast')[0, 0])
    # zs.append(greycoprops(glcm, 'correlation')[0, 0])
    # zs.append(greycoprops(glcm, 'dissimilarity')[0, 0])

    media = np.mean(patch)
    desvio_padrao = np.std(patch)
    desvio_padrao_a = np.std(patch)
    desvio_padrao_b = np.std(patch)

    xs.append(media)
    ys.append(desvio_padrao)


print "Media",xs
print "Desvio",ys
print "Contraste",zs

# display original image with locations of patches
fig.subplots_adjust(bottom=0.125, left=0.065, top = 0.875, right=0.975)
ax = fig.add_subplot(221)
ax.imshow(gray, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
for (y, x) in water_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, "bs")
ax.set_xlabel('Original Image')
ax.axis('image')


# for each patch, plot (dissimilarity, correlation)
# ax = fig.add_subplot(222)
# ax.plot(xs[:len(water_patches)], ys[:len(water_patches)], "bs", label='Water')
# ax.set_xlabel('GLCM Contrast')
# ax.set_ylabel('GLCM Energy')
# ax.legend()

###3D AXIS
ax = fig.add_subplot(222,projection="3d")
ax.scatter3D(xs,ys,zs, 'gray')
ax.set_xlabel("Media")
ax.set_ylabel("Desvio")
ax.set_zlabel("Contraste")
ax.legend()

# display the image patches
for i, patch in enumerate(water_patches):
    ax = fig.add_subplot(2,len(water_patches),len(water_patches)+i+1)
    ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    ax.set_xlabel('Water %d' % (i + 1))

#display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
plt.show()