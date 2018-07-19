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

		if(ground_truth[i] == 0):

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


if __name__=="__main__":

	plt.clf()

	img = cv.imread("images/seagull_database_vis011_small.png")


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
	
	
	# zeros = reconstruct_GT_aux(predi,window)
	# zeros = reconstruct_GT_aux(ground_truth,window)

	# image_reconstructed = invers_blocos_16x16(zeros,gray,windowsize_r,windowsize_c)
	# # contours, hierarchy = cv.findContours(image_reconstructed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	# # print "Antes",contours
	# # cv.drawContours(img, np.multiply(contours,2), -1, (0,0,255), 2)
	# # print "Depois",contours*2

	# cv.imshow("Imagem Reconstruida",image_reconstructed)
	# cv.waitKey(0)
	# cv.destroyAllWindows()