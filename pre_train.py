# -*- coding: utf-8 -*-
"""
Created on Sun May 06 19:47:12 2018

@author: Michael
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

from skimage.feature import greycomatrix, greycoprops
from skimage import data
from sklearn.metrics.cluster import entropy

from mpl_toolkits.mplot3d import Axes3D

from sklearn.svm import SVC

import cPickle as pickle
import os

from numpy import linalg as LA

def blocos_16_16(image,windowsize_r,windowsize_c):

	original_height = image.shape[0]
	original_width = image.shape[1]

	window = np.zeros((original_width/windowsize_c*original_height/windowsize_r,windowsize_r,windowsize_c))

	k = 0
	for r in range(0,image.shape[0] , windowsize_r):
	    for c in range(0,image.shape[1], windowsize_c):
	    	# print r,c
	        window[k] = image[r:r+windowsize_r,c:c+windowsize_c]
	        k+=1
	return np.uint8(window)

def features_extraction(window,window2,window3, n_features):

	features = np.zeros((window.shape[0], n_features))

	for bloco in range(len(window)):

		glcm = greycomatrix(window[bloco], [1], [0, np.pi/2, np.pi, 3*np.pi/2], symmetric=True, normed=True)
		contrast = greycoprops(glcm, 'contrast')[0, 0]
		desvio_padrao_a = np.std(window2[bloco])
		desvio_padrao_b = np.std(window3[bloco])
		media = np.mean(window2[bloco])

		if(n_features==1):
			features[bloco][0] = contrast
		elif (n_features==2):
			features[bloco][0] = contrast
			features[bloco][1] = desvio_padrao_a
		elif (n_features==3):
			features[bloco][0] = contrast
			features[bloco][1] = desvio_padrao_a
			features[bloco][2] = desvio_padrao_b

	return features

def features_filter(features,window,n_features):

	zeros = np.zeros(window.shape)
	ground_truth = np.zeros(window.shape[0])

	for i in range(features.shape[0]):

		if(n_features==1):
			if(features[i] < 400 ):
				
				zeros[i] = window[i]
				ground_truth[i] = 1

		if(n_features==2):

			if(features[i][0] > 400 and features[i][1] > 1.2):
				
				zeros[i] = window[i]
				ground_truth[i] = 1

		if(n_features==3):

			if(features[i][0] > 400 and features[i][1] > 3 and features[i][2] > 4):
				
				zeros[i] = window[i]
				ground_truth[i] = 1


	return zeros,ground_truth

def reconstruct_GT_aux(ground_truth,window):

	zeros = np.zeros(window.shape)

	for i in range(len(ground_truth)):

		if(ground_truth[i] == 1):

			zeros[i] = window[i]
			# zeros[i] = np.array([[255]*window[i].shape[1]]*window[i].shape[0])
			# print i

	return zeros

def invers_blocos_16x16(blocos,window,windowsize_r,windowsize_c):

	original_height = window.shape[0]
	original_width = window.shape[1]

	image_reconstructed = np.zeros((original_height,original_width))
	k = 0
	for i in range(0,image_reconstructed.shape[0],windowsize_r):
	    for j in range(0,image_reconstructed.shape[1],windowsize_c):
	        image_reconstructed[i:i+windowsize_r,j:j+windowsize_c] = blocos[k]
	        k+=1
	return np.uint8(image_reconstructed)

def show_features_1d(features):

	plt.plot(features,len(features)*[1])
	plt.show()

def show_features_1d(features,ground_truth):

	plt.plot(features[ground_truth==1],len(features[ground_truth==1])*[1],'xg')
	plt.plot(features[ground_truth==0],len(features[ground_truth==0])*[-1],'or')
	plt.axis([0,6000,-2,2])
	plt.show()

def show_features_2d(features,ground_truth):

	plt.scatter(features[ground_truth==1,0],features[ground_truth==1,1],color = 'blue',marker = 's')
	plt.scatter(features[ground_truth==0,0],features[ground_truth==0,1],color = 'red', marker = 'o')

	plt.xlabel('Contraste'),plt.ylabel('Desvio Padrao')
	plt.show()

def show_features_3d(features):

	ax = plt.axes(projection='3d')
	ax.scatter3D(features[:,0],features[:,1],features[:,2], 'gray')
	ax.set_xlabel("Contraste")
	ax.set_ylabel("Desvio")
	ax.set_zlabel("Media")
	ax.legend()
	plt.show()

def show_features_3d_2(features,ground_truth):

	ax = plt.axes(projection='3d')
	ax.scatter3D(features[ground_truth==1,0],features[ground_truth==1,1],features[ground_truth==1,2], color = 'blue',marker = 's')
	ax.scatter3D(features[ground_truth==0,0],features[ground_truth==0,1],features[ground_truth==0,2], color = 'red', marker = 'o')
	ax.set_xlabel("Contraste")
	ax.set_ylabel("Desvio_A")
	ax.set_zlabel("Desvio_B")
	ax.legend()
	plt.show()

def size_block(img):

	img_height = img.shape[0]
	img_width = img.shape[1]


	r = 0
	c = 0
	dimensoes = [14,15,16]

	r = [dimensao for dimensao in dimensoes if img_height%dimensao == 0]	
	c = [dimensao for dimensao in dimensoes if img_width%dimensao == 0]	


	return r[-1],c[-1]

def read_file(path):

	file_read = open(path,"rb")
	example_dict = pickle.load(file_read)
	file_read.close()

	return example_dict

def write_file(path,features,ground_truth):

	file = open(path,"wb")
	dic = {"features":features,"ground_truth":ground_truth}
	pickle.dump(dic,file)
	file.close()

	return "Criou"

def read_or_write_pickle(path,new_features,new_ground_truth,default):

	if os.path.isfile(path):
		
		dic = read_file(path)
		feat = dic["features"]
		gt = dic["ground_truth"]

		feat = np.vstack((feat,new_features))
		gt = np.hstack((gt,new_ground_truth))
		# gt = np.reshape(gt,(gt.shape[1],gt.shape[0]))
		write_file(path,feat,gt)

		return "Juntou"
	else:
		return write_file(path,new_features,new_ground_truth)

	return default

def classificator_train(classificator,features,ground_truth):
	
	classificator.fit(features,ground_truth)
	
	return classificator

def classificator_test(classificator,features):

	return classificator.predict(features)

def showimg(title, img):
	cv.imshow(title,img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def resize(img,vx,vy):
	return cv.resize(img,None,fx=vx, fy=vy, interpolation = cv.INTER_CUBIC)

def groundtruth(window):

	groundtruth = np.zeros(len(window))

	#Imagem vis003
	# for i in range(len(window)):
	# 	if(0 <= i < 106 or 107 <= i < 110 or 112 <= i < 136 or 137 <= i < 139 or 141 <= i < 165 or 169 <= i < 193 or 198 <= i < 221 or 226 <= i < 249 or 253 <= i < 278 or i >= 281):
	# 		groundtruth[i] = 1
	# 	elif(i == 106 or 110 <= i < 112 or i == 136 or 139 <= i < 141 or 165 <= i < 169 or 193 <= i < 198 or 221 <= i < 226 or 249 <= i < 253 or 280 <= i < 281):
	# 		groundtruth[i] = 3
	# 	elif(278 <= i < 280):
	# 		groundtruth[i] = 2

	#Imagem vis004
	# for i in range(len(window)):
	# 	if(0 <= i < 64 or 66 <= i < 75 or 76 <= i < 84 or 85 <= i < 102 or 103 <= i < 105 or 106 <= i < 134 or 135 <= i < 166 or 167 <= i < 176 or 177 <= i < 184 or 185 <= i < 221 or 222 <= i < 235 or 236 <= i < 283 or 285 <= i < 301 or 302 <= i < 313 or 315 <= i < 326 or 327 <= i < 343 or 344 <= i < 361 or 362 <= i < 391 or 392 <= i < 413 or 414 <= i < 424 or 425 <= i < 431 or 432 <= i < 453 or 454 <= i < 492 or i >= 493):
	# 		groundtruth[i] = 1
	# 	elif(64 <= i < 66 or i == 75 or i == 84 or i == 102 or i == 105 or i == 134 or i == 166 or i == 176 or i == 184 or i == 221 or i == 235 or i == 301 or i == 326 or i == 343 or i == 361 or i == 391 or i == 413 or i == 424 or i == 431 or i == 453 or i == 492):
	# 		groundtruth[i] = 3
	# 	elif(283 <= i < 285 or 313 <= i < 315):
	# 		groundtruth[i] = 2

	#Imagem vis005
	for i in range(len(window)):
		if(0 <= i < 12 or 18 <= i < 38 or 50 <= i < 67 or 81 <= i < 96 or 112 <= i < 125 or 141 <= i < 155 or 173 <= i < 185 or 203 <= i < 214 or 234 <= i < 244 or 264 <= i < 266 or 267 <= i < 273 or 293 <= i < 303 or 324 <= i < 333 or 355 <= i < 364 or 385 <= i < 394 or 415 <= i < 424 or 444 <= i < 453 or 473 <= i < 485 or i >= 503):
			groundtruth[i] = 1
		elif(12 <= i < 18 or 38 <= i < 50 or 67 <= i < 81 or 96 <= i < 112 or 125 <= i < 141 or 155 <= i < 173 or 185 <= i < 203 or 214 <= i < 234 or 244 <= i < 264 or 273 <= i < 293 or 303 <= i < 324 or 333 <= i < 355 or 364 <= i < 385 or 394 <= i < 415 or 424 <= i < 444 or 453 <= i < 473 or 485 <= i < 503):
			groundtruth[i] = 4
		elif(i == 266):
			groundtruth[i] = 2

	return groundtruth

if __name__=="__main__":

	plt.clf()

	img = cv.imread("images/frame600.jpg")


	res = cv.resize(img,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
	

	gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
	lab = cv.cvtColor(res,cv.COLOR_BGR2LAB)
	

	windowsize_r = size_block(res)[0]
	windowsize_c = size_block(res)[1]


	window = blocos_16_16(gray,windowsize_r,windowsize_c)

	window_2 = blocos_16_16(lab[:,:,1],windowsize_r,windowsize_c)

	window_3 = blocos_16_16(lab[:,:,2],windowsize_r,windowsize_c)

	features = features_extraction(window,window_2,window_3,3)
	# print features[:,0]

	# showimg("Original",res)

	#Mostrar os blocks:
	print "Resolução Imagem",img.shape
	for i in range(len(window)):
		print i
		showimg("Blocks",resize(window[i],10,10))
		glcm = greycomatrix(window[i], [1], [0, np.pi/2, np.pi/4, 3*np.pi/4], symmetric=True, normed=True)
		print np.array([np.mean(window[i]),np.std(window[i]),greycoprops(glcm, 'contrast')[0, 0]])

	# show_features_3d(features)

	# zeros,ground_truth = features_filter(features,window,3)

	# print features
	# print ground_truth
	# show_features_3d_2(features,ground_truth)

	# show_features_1d(features,ground_truth)
	
	# print "ANTIGO",features
	# feat = read_or_write_pickle("features_train.pickle",features,ground_truth,"Erro")
	# print "NOVO",feat


	# dic = read_file("features_train.pickle")
	# old_feat = dic["features"]
	# old_gt = dic["ground_truth"]

	# print dic
	# print old_gt.shape
	# classifier = SVC()
	# classi = classificator_train(classifier,old_feat,old_gt)
	# predi = classificator_test(classi,features)

	# print predi
	
	ground_truth = groundtruth(window)
	print ground_truth.shape
	# zeros = reconstruct_GT_aux(predi,window)
	zeros = reconstruct_GT_aux(ground_truth,window)

	image_reconstructed = invers_blocos_16x16(zeros,gray,windowsize_r,windowsize_c)
	# # contours, hierarchy = cv.findContours(image_reconstructed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	# # print "Antes",contours
	# # cv.drawContours(img, np.multiply(contours,2), -1, (0,0,255), 2)
	# # print "Depois",contours*2

	cv.imshow("Imagem Reconstruida",image_reconstructed)
	cv.waitKey(0)
	cv.destroyAllWindows()