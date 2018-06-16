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
PATCH_SIZE = 16

video = cv.VideoCapture('images/video2.mp4')
ret, frame = video.read()
cv.imwrite("frame1.png",frame)
img = cv.imread("images/seagull_database_vis001_small.png")
print frame.shape
# res = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
# res = cv.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)

gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
lab = cv.cvtColor(res,cv.COLOR_BGR2LAB)
#
##cv.imshow("gray",gray)
##cv.waitKey(0)
##cv.destroyAllWindows()
#
# select some patches from water areas of the image

###Imagem 1
water_locations = [(10, 10),(200,200),(50,200),(110,220),(240,350),(110,200)]
print "Imagem 1"

###Imagem 2
# water_locations = [(10, 10),(240,350),(50,200),(150,230),(150,280)]
# print "Imagem 2"

water_patches = []
for loc in water_locations:
    water_patches.append(lab[:,:,2][loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE])

xs = []
ys = []
zs = []
for patch in (water_patches):
    glcm = greycomatrix(patch, [1], [0, np.pi/2, np.pi, 3*np.pi/2], symmetric=True, normed=True)
    zs.append(greycoprops(glcm, 'contrast')[0, 0])
    
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
ax.imshow(res, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
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