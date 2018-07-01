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
		desvio = np.std(window[bloco])
		desvio_padrao_a = np.std(window2[bloco])
		desvio_padrao_b = np.std(window3[bloco])
		media = np.mean(window[bloco])

		if(n_features==1):
			features[bloco][0] = contrast
		elif (n_features==2):
			features[bloco][0] = contrast
			features[bloco][1] = desvio_padrao_a
		elif (n_features==3):
			features[bloco][0] = contrast
			features[bloco][1] = desvio
			features[bloco][2] = media

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

def ground_truth_filter(features,window):

	ground_truth = np.zeros(len(window))

	for i in range(len(features)):

		if(features[i][0] < 12 and features[i][1] < 8 and 120 < features[i][2] < 145):
				
				ground_truth[i] = 1
		elif(features[i][0] > 150 and features[i][1] > 20 and features[i][2] < 140):
				
				ground_truth[i] = 2
		elif(features[i][0] > 15 and features[i][1] > 15 and features[i][2] > 140):
		
			ground_truth[i] = 3

	return ground_truth		

def reconstruct_GT_aux(ground_truth,window):

	zeros = np.zeros(window.shape)

	for i in range(len(ground_truth)):

		if(ground_truth[i] == 3):

			# zeros[i] = window[i]
			zeros[i] = np.array([[255]*window[i].shape[1]]*window[i].shape[0])
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
	ax.scatter3D(features[ground_truth==3,0],features[ground_truth==3,1],features[ground_truth==3,2], color = 'yellow',marker = 's')
	ax.scatter3D(features[ground_truth==2,0],features[ground_truth==2,1],features[ground_truth==2,2], color = 'green',marker = 's')
	ax.scatter3D(features[ground_truth==1,0],features[ground_truth==1,1],features[ground_truth==1,2], color = 'blue',marker = 's')
	ax.scatter3D(features[ground_truth==0,0],features[ground_truth==0,1],features[ground_truth==0,2], color = 'red', marker = 'o')
	ax.set_xlabel("Contraste")
	ax.set_ylabel("Desvio")
	ax.set_zlabel("Media")
	ax.legend()
	plt.show()

def show_features_3d_3(features,ground_truth, new_features):

	ax = plt.axes(projection='3d')
	ax.scatter3D(features[ground_truth==3,0],features[ground_truth==3,1],features[ground_truth==3,2], color = 'yellow',marker = 's')
	ax.scatter3D(features[ground_truth==2,0],features[ground_truth==2,1],features[ground_truth==2,2], color = 'green',marker = 's')
	ax.scatter3D(features[ground_truth==1,0],features[ground_truth==1,1],features[ground_truth==1,2], color = 'blue',marker = 's')
	ax.scatter3D(features[ground_truth==0,0],features[ground_truth==0,1],features[ground_truth==0,2], color = 'red', marker = 'o')
	ax.scatter3D(new_features[ground_truth==2,0],new_features[ground_truth==2,1],new_features[ground_truth==2,2], color = 'black', marker = 'o')
	ax.set_xlabel("Contraste")
	ax.set_ylabel("Desvio")
	ax.set_zlabel("Media")
	ax.legend()
	plt.show()

def trans_class(groundtruth):

	for i in range(len(groundtruth)):

		if(groundtruth[i] == 0 or groundtruth[i] == 1):

			groundtruth[i] = 0

		else:
			groundtruth[i] = 1

	return groundtruth

def size_block_video(video):

	ret, frame = video.read()

	frame_height = frame.shape[0]
	frame_width = frame.shape[1]

	r = 0
	c = 0
	dimensoes = [7,8,9]

	r = [dimensao for dimensao in dimensoes if frame_height%dimensao == 0]	
	c = [dimensao for dimensao in dimensoes if frame_width%dimensao == 0]	


	return r[-1],c[-1]

def size_block(img):

	img_height = img.shape[0]
	img_width = img.shape[1]


	r = 0
	c = 0
	dimensoes = [7,8,9]

	r = [dimensao for dimensao in dimensoes if img_height%dimensao == 0]	
	c = [dimensao for dimensao in dimensoes if img_width%dimensao == 0]	


	return r[-1],c[-1]

def read_file(path):

	file_read = open(path,"rb")
	example_dict = pickle.load(file_read)
	file_read.close()

	return example_dict

def write_ground(path,name,ground_truth):

	file = open(path,"wb")
	dic = {name:ground_truth}
	pickle.dump(dic,file)
	file.close()

	return "Criou"

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

	# ##Imagem vis003
	# for i in range(len(window)):
	# 	if(0 <= i < 106 or 107 <= i < 110 or 112 <= i < 136 or 137 <= i < 139 or 141 <= i < 165 or 169 <= i < 193 or 198 <= i < 221 or 226 <= i < 249 or 253 <= i < 278 or i >= 281):
	# 		groundtruth[i] = 1
	# 	elif(i == 106 or 110 <= i < 112 or i == 136 or 139 <= i < 141 or 165 <= i < 169 or 193 <= i < 198 or 221 <= i < 226 or 249 <= i < 253 or 280 <= i < 281):
	# 		groundtruth[i] = 3
	# 	elif(278 <= i < 280):
	# 		groundtruth[i] = 2

	# ##Imagem vis004
	# for i in range(len(window)):
	# 	if(0 <= i < 64 or 66 <= i < 75 or 76 <= i < 84 or 85 <= i < 102 or 103 <= i < 105 or 106 <= i < 134 or 135 <= i < 166 or 167 <= i < 176 or 177 <= i < 184 or 185 <= i < 221 or 222 <= i < 235 or 236 <= i < 283 or 285 <= i < 301 or 302 <= i < 313 or 315 <= i < 326 or 327 <= i < 343 or 344 <= i < 361 or 362 <= i < 391 or 392 <= i < 413 or 414 <= i < 424 or 425 <= i < 431 or 432 <= i < 453 or 454 <= i < 492 or i >= 493):
	# 		groundtruth[i] = 1
	# 	elif(64 <= i < 66 or i == 75 or i == 84 or i == 102 or i == 105 or i == 134 or i == 166 or i == 176 or i == 184 or i == 221 or i == 235 or i == 301 or i == 326 or i == 343 or i == 361 or i == 391 or i == 413 or i == 424 or i == 431 or i == 453 or i == 492):
	# 		groundtruth[i] = 3
	# 	elif(283 <= i < 285 or 313 <= i < 315):
	# 		groundtruth[i] = 2

	# ##Imagem vis005
	# for i in range(len(window)):
	# 	if(0 <= i < 12 or 18 <= i < 38 or 50 <= i < 67 or 81 <= i < 96 or 112 <= i < 125 or 141 <= i < 155 or 173 <= i < 185 or 203 <= i < 214 or 234 <= i < 244 or 264 <= i < 266 or 267 <= i < 273 or 293 <= i < 303 or 324 <= i < 333 or 355 <= i < 364 or 385 <= i < 394 or 415 <= i < 424 or 444 <= i < 453 or 473 <= i < 485 or i >= 503):
	# 		groundtruth[i] = 1
	# 	elif(12 <= i < 18 or 38 <= i < 50 or 67 <= i < 81 or 96 <= i < 112 or 125 <= i < 141 or 155 <= i < 173 or 185 <= i < 203 or 214 <= i < 234 or 244 <= i < 264 or 273 <= i < 293 or 303 <= i < 324 or 333 <= i < 355 or 364 <= i < 385 or 394 <= i < 415 or 424 <= i < 444 or 453 <= i < 473 or 485 <= i < 503):
	# 		groundtruth[i] = 4
	# 	elif(i == 266):
	# 		groundtruth[i] = 2

	# # #Imagem 
	# for i in range(len(window)):
	# 	if(1 <= i < 5 or i == 6 or 12 <= i < 14 or 16 <= i < 18 or i == 20 or 23 <= i < 25 or 30 <= i < 32 or i == 33 or 41 <= i < 45 or i == 46 or i == 59 or i == 73 or 81 <= i < 84 or i == 86 or 100 <= i < 103 or 120 <= i < 124 or i == 133 or i == 152 or i == 160 or 162 <= i < 164 or i == 173 or i == 185 or i ==190 or i == 193 or i == 200 or i == 203 or i == 219 or i == 228 or i == 230 or 240 <= i < 244 or i == 270 or 280 <= i < 284 or i == 311 or i == 321 or 323 <= i < 325 or i == 393 or i == 395 or i == 398 or i == 417 or i == 426 or i == 433 or i == 435 or i == 437 or i == 450 or i == 457 or i == 466 or i == 473 or i == 492 or 509 <= i < 511 or i == 665 or i == 673 or i == 699 or 743 <= i < 745 or i == 816 or i == 820 or i == 855 or i == 895 or i == 905 or i == 913 or 935 <= i < 937 or i == 940 or i == 943):
	# 		groundtruth[i] = 1
	# 	elif(i == 0 or i == 5 or 7 <= i < 12 or 14 <= i < 16 or 18 <= i < 20 or 21 <= i < 23 or 25 <= i < 30 or i == 32 or 34 <= i < 41 or i == 45 or 47 <= i < 59 or 60 <= i < 73 or 74 <= i < 81 or 84 <= i < 86 or 87 <= i < 100 or 103 <= i < 120 or 124 <= i < 133 or 134 <= i < 152 or 153 <= i < 160 or i == 161 or 164 <= i < 173 or 174 <= i < 185 or 186 <= i < 190 or 191 <= i < 193  or 194 <= i < 200 or 201 <= i < 203 or 204 <= i < 219 or 220 <= i < 228 or i == 229 or 231 <= i < 240 or 244 <= i < 270 or 271 <= i < 280 or 284 <= i < 311 or 312 <= i < 321 or i == 322 or 325 <= i < 393 or i == 394 or 396 <= i < 398 or 399 <= i < 417 or 418 <= i < 426 or 427 <= i < 433 or i == 434 or i == 436 or 438 <= i < 450 or 451 <= i < 457 or 458 <= i < 466 or 467 <= i < 473 or 474 <= i < 492 or 493 <= i < 509 or 511 <= i < 537 or 538 <= i < 576 or 578 <= i < 616 or 618 <= i < 656 or 658 <= i < 665 or 666 <= i < 673 or 674 <= i < 699 or 700 <= i < 743 or 745 <= i < 816 or 817 <= i < 820 or 821 <= i < 855 or 856 <= i < 895 or 896 <= i < 905 or 906 <= i < 913 or 914 <= i < 935 or 937 <= i < 940 or 941 <= i < 943 or i >= 944):
	# 		groundtruth[i] = 3
	# 	elif(i == 537 or 576 <= i < 578 or 616 <= i < 618 or 656 <= i < 658):
	# 		groundtruth[i] = 2

	# #Frame 4514
	for i in range(len(window)):
		if(i == 0 or 2 <= i < 6 or 7 <= i < 13 or 14 <= i < 46 or 47 <= i < 54 or 55 <= i < 139 or 140 <= i < 179 or 180 <= i < 361 or 362 <= i < 389 or 390 <= i < 395 or 398 <= i < 430 or i == 433 or 438 <= i < 487 or 488 <= i < 548 or 549 <= i < 582 or 583 <= i < 595 or 596 <= i < 635 or 636 <= i < 672 or 674 <= i < 693 or 697 <= i < 708 or 709 <= i < 720 or 723 <= i < 726 or 728 <= i < 730 or 736 <= i < 739 or 740 <= i < 756 or 757 <= i < 760 or i == 764 or 783 <= i < 788 or i >= 797):
			groundtruth[i] = 1
		elif(i == 1 or i == 6 or i == 13 or i == 54 or i == 361 or i == 389 or 395 <= i < 398 or 430 <= i < 433 or 434 <= i < 438 or i == 487 or i == 582 or i == 595 or i == 635 or 672 <= i < 674 or 693 <= i < 696 or i == 708 or 720 <= i < 723 or 726 <= i < 728 or 730 <= i < 736 or i == 739 or i == 756 or 760 <= i < 764 or 765 <= i < 783 or 788 <= i < 797):
			groundtruth[i] = 3
		elif(i == 46 or i == 139 or i == 179 or i == 548):
			groundtruth[i] = 2

	return groundtruth

def process_video(video,classi, windowsize_r, windowsize_c, fator):

	while(video.isOpened()):

		ret, frame = video.read()
		print ret
		res = cv.resize(frame,None,fx=1./fator, fy=1./fator, interpolation = cv.INTER_CUBIC)

		gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
		lab = cv.cvtColor(res,cv.COLOR_BGR2LAB)
		
		windowsize_r = size_block(res)[0]
		windowsize_c = size_block(res)[1]


		window = blocos_16_16(gray,windowsize_r,windowsize_c)

		window_2 = blocos_16_16(lab[:,:,1],windowsize_r,windowsize_c)

		window_3 = blocos_16_16(lab[:,:,2],windowsize_r,windowsize_c)

		features = features_extraction(window,window_2,window_3,3)

		
		predi = classificator_test(classi,features)
		
		
		zeros = reconstruct_GT_aux(predi,window)
		# zeros = reconstruct_GT_aux(ground_truth,window)

		image_reconstructed = invers_blocos_16x16(zeros,gray,windowsize_r,windowsize_c)

		contours, hierarchy = cv.findContours(image_reconstructed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		cv.drawContours(frame, np.multiply(contours,fator), -1, (0,0,255), 2)

		cv.imshow("Imagem Reconstuida",frame)
		

		if cv.waitKey(30) & 0xff == ord('q'):
			break    

	video.release()
	cv.destroyAllWindows()


def input_label(window, res):

	groundtruth = np.ones(len(window))

	
	#Mostrar os blocks:
	print "Resolução Imagem",res.shape
	print "Dimensao dos blocos", window[0].shape
	for i in range(len(window)):
		print i
		showimg("Block Original",resize(window[i],window[i].shape[0],window[i].shape[1]))
		
		glcm = greycomatrix(window[i], [1], [0, np.pi/2, np.pi/4, 3*np.pi/4], symmetric=True, normed=True)
		print "Media     Desvio      Contraste"
		print np.array([np.mean(window[i]),np.std(window[i]),greycoprops(glcm, 'contrast')[0, 0]])
  		
		print "Label: "
		label = input()
		
		groundtruth[i] = label

	return groundtruth


if __name__=="__main__":

	plt.clf()

	# cap = cv.VideoCapture('images/video1.mp4')

	# img = cv.imread("images/seagull_database_vis002_small.png")
	img = cv.imread("images/Frame4514.jpg")

	res = cv.resize(img,None,fx=0.25, fy=0.25, interpolation = cv.INTER_CUBIC)
	

	gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
	lab = cv.cvtColor(res,cv.COLOR_BGR2LAB)
	

	windowsize_r = size_block(res)[0]
	windowsize_c = size_block(res)[1]


	window = blocos_16_16(gray,windowsize_r,windowsize_c)

	window_2 = blocos_16_16(lab[:,:,1],windowsize_r,windowsize_c)

	window_3 = blocos_16_16(lab[:,:,2],windowsize_r,windowsize_c)

	features = features_extraction(window,window_2,window_3,3)
	print "Nfeatures",features.shape
	# print features[:,0]

	# #Mostrar os blocks:
	# print "Resolução Imagem",res.shape
	# print "Dimensao dos blocos", window[0].shape
	# for i in range(len(window)):
	# 	print i
	# 	showimg("Block Original",resize(window[i],window[i].shape[0],window[i].shape[1]))
		
	# 	glcm = greycomatrix(window[i], [1], [0, np.pi/2, np.pi/4, 3*np.pi/4], symmetric=True, normed=True)
	# 	print "Media       Desvio        Contraste"
	# 	print np.array([np.mean(window[i]),np.std(window[i]),greycoprops(glcm, 'contrast')[0, 0]])
	# showimg("Original",res)

	# ground_truth = ground_truth_filter(features,window)
	# write_ground("ground_truth_images.pickle","images/Frame4514.jpg",ground_truth)
	# show_features_3d(features)

	# zeros,ground_truth = features_filter(features,window,3)

	# print features
	# print ground_truth
	# show_features_3d_2(features,ground_truth)
	
	ground_truth = groundtruth(window)
	# ground_truth = trans_class(ground_truth)
	# # print ground_truth.shape

	show_features_3d_2(features,ground_truth)

	# feat = read_or_write_pickle("3features_train_ship_3Classes.pickle",features,ground_truth,"Erro")
	# zeros = reconstruct_GT_aux(predi,window)
	# dic = read_file("3features_train_ship_3Classes.pickle")
	# old_feat = dic["features"]
	# old_gt = dic["ground_truth"]

	# show_features_3d_2(old_feat,old_gt)
	# show_features_3d_3(old_feat,old_gt, features)
	# print old_feat.shape
	# # print old_gt

	# classifier = SVC(kernel = 'rbf', C = 1.0)
	# classi = classificator_train(classifier,old_feat,old_gt)
	# predi = classificator_test(classi,features)

	# process_video(cap,classi, size_block_video(cap)[0], size_block_video(cap)[1],2)
	

	# print "Coef1", classi.coef_
	# print "Number of Support Vectors", classi.n_support_

	zeros = reconstruct_GT_aux(ground_truth,window)
	# zeros = reconstruct_GT_aux(predi,window)

	image_reconstructed = invers_blocos_16x16(zeros,gray,windowsize_r,windowsize_c)
	contours, hierarchy = cv.findContours(image_reconstructed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	# # print "Antes",contours
	cv.drawContours(img, np.multiply(contours,4), -1, (0,0,255), 2)
	# # print "Depois",contours*2

	cv.imshow("Imagem Reconstruida",img)
	cv.waitKey(0)
	cv.destroyAllWindows()

