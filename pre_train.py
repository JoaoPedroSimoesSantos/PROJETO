import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math

from skimage.feature import greycomatrix, greycoprops
from skimage import data
from sklearn.metrics import confusion_matrix
import itertools

from mpl_toolkits.mplot3d import Axes3D

from sklearn.svm import SVC

import cPickle as pickle
import os

from numpy import linalg as LA

import time 

import sys

def divisao_de_blocos(image,windowsize_r,windowsize_c):

	original_height = image.shape[0]
	original_width = image.shape[1]

	window = np.zeros((original_width/windowsize_c*original_height/windowsize_r,windowsize_r,windowsize_c))

	loc_blocos = np.zeros((window.shape[0],2))

	k = 0
	for r in range(0,image.shape[0] , windowsize_r):
	    for c in range(0,image.shape[1], windowsize_c):
	        window[k] = image[r:r+windowsize_r,c:c+windowsize_c]
	        loc_blocos[k] = [r,c]
	        k+=1


	return np.uint8(window), loc_blocos

def features_extraction(window,n_features):

	features = np.zeros((window.shape[0], n_features))

	for bloco in range(len(window)):

		glcm = greycomatrix(window[bloco], [1], [3*np.pi/4], symmetric=True, normed=True)
		contrast = greycoprops(glcm, 'contrast')[0, 0]
		correlation = greycoprops(glcm, 'correlation')[0, 0]
		desvio = np.std(window[bloco])
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
		elif (n_features==4):
			features[bloco][0] = contrast
			features[bloco][1] = desvio
			features[bloco][2] = media
			features[bloco][3] = correlation


	return features


def ajuste_ground_truth(bloco,window):

	booleana = False
	for k in range(window[bloco].shape[0]):
	    for l in range(window[bloco].shape[1]):
	        if(window[bloco][k][l] < 40):
	            booleana = True	
	            break

	return booleana

def reconstruct_GT_aux(ground_truth,window,features):

	zeros = np.zeros(window.shape)
	idx_true = []
	for i in range(len(ground_truth)):

		if(ground_truth[i] == 1):
			# booleana = ajuste_ground_truth(i,window)

			# if(booleana):
				zeros[i] = np.array([[255]*window[i].shape[1]]*window[i].shape[0])
				idx_true.append(i)

	return zeros, idx_true

def invers_blocos_16x16(blocos,window,windowsize_r,windowsize_c):

	original_height = window.shape[0]
	original_width = window.shape[1]

	image_reconstructed = np.zeros((original_height,original_width))
	k = 0
	for i in range(0,image_reconstructed.shape[0],windowsize_r):
	    for j in range(0,image_reconstructed.shape[1],windowsize_c):

	        image_reconstructed[i:i+windowsize_r,j:j+windowsize_c] = blocos[0][k]
	        k+=1
	return np.uint8(image_reconstructed)

def show_features_1d_1(features):

	plt.plot(features,len(features)*[1])
	plt.show()

def show_features_1d_2(features,ground_truth):

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
	ax.scatter3D(new_features[:,0],new_features[:,1],new_features[:,2], color = 'black', marker = 'o')
	ax.set_xlabel("Contraste")
	ax.set_ylabel("Desvio")
	ax.set_zlabel("Media")
	ax.legend()
	plt.show()

def size_block_video(video):

	ret, frame = video.read()
	frame_height = frame.shape[0]
	frame_width = frame.shape[1]

	r = 0
	c = 0

	# dimensoes = [4,5,6]
	dimensoes = [7,8,9]

	r = [dimensao for dimensao in dimensoes if frame_height%dimensao == 0]	
	c = [dimensao for dimensao in dimensoes if frame_width%dimensao == 0]	


	return r[-1],c[-1]

def size_block(img):

	img_height = img.shape[0]
	img_width = img.shape[1]


	r = 0
	c = 0
	# dimensoes = [4,5,6]
	dimensoes = [7,8,9]

	r = [dimensao for dimensao in dimensoes if img_height%dimensao == 0]	
	c = [dimensao for dimensao in dimensoes if img_width%dimensao == 0]	


	return r[-1],c[-1]


def train_pickle(path_img,features_p,ground_truth_p, pickle):

	if os.path.isfile(pickle):
		
		dic1 = read_file(features_p)
		dic2 = read_file(ground_truth_p)
		feat = dic1[path_img]
		gt = dic2[path_img]

		train_dic = read_file(pickle)
		if(len(train_dic["ground_truth"])==0):
			train_dic["ground_truth"] = gt
			train_dic["features"] = feat
		else:
			train_dic["ground_truth"] = np.hstack((train_dic["ground_truth"],gt))
			train_dic["features"] = np.vstack((train_dic["features"],feat))

		write_file(pickle,train_dic)

		return "Juntou"

	return "Nao Juntou"

def read_file(path):
	file = open(path,"r")
	dic = pickle.load(file)
	file.close()

	return dic

def write_file(path,dic):
	file = open(path,"w")
	pickle.dump(dic,file)
	file.close()

def update_pickle(path,namefig,features):

	if os.path.isfile(path):
		
		dic = read_file(path)

		dic[namefig] = features

		write_file(path,dic)

		return "Juntou"
	else:
		return "Nao Juntou"

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

def groundtruth(window,path):

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

	# # #Frame 4514
	# for i in range(len(window)):
	# 	if(i == 0 or 2 <= i < 6 or 7 <= i < 13 or 14 <= i < 46 or 47 <= i < 54 or 55 <= i < 139 or 140 <= i < 361 or 362 <= i < 389 or 390 <= i < 395 or 398 <= i < 430 or i == 433 or 438 <= i < 487 or 488 <= i < 548 or 549 <= i < 582 or 583 <= i < 595 or 596 <= i < 635 or 636 <= i < 672 or 674 <= i < 693 or 697 <= i < 708 or 709 <= i < 720 or 723 <= i < 726 or 728 <= i < 730 or 736 <= i < 739 or 740 <= i < 756 or 757 <= i < 760 or i == 764 or 783 <= i < 788 or i >= 797):
	# 		groundtruth[i] = 1
	# 	elif(i == 1 or i == 6 or i == 13 or i == 54 or i == 361 or i == 389 or 395 <= i < 398 or 430 <= i < 433 or 434 <= i < 438 or i == 487 or i == 582 or i == 595 or i == 635 or 672 <= i < 674 or 693 <= i < 696 or i == 708 or 720 <= i < 723 or 726 <= i < 728 or 730 <= i < 736 or i == 739 or i == 756 or 760 <= i < 764 or 765 <= i < 783 or 788 <= i < 797):
	# 		groundtruth[i] = 3
	# 	elif(i == 46 or i == 139 or i == 548):
	# 		groundtruth[i] = 2

	# #Frame 3597
	# for i in range(len(window)):
	# 	if(22 <= i <= 23 or 40 <= i <= 50 or i == 53 or 60 <= i <= 74 or 80 <= i <= 89 or 92 <= i <= 93 or 99 <= i <= 105 or i == 113 or i == 116 or 120 <= i <= 122 or i == 127 or i == 138 or 140 <= i <= 142 or 161 <= i <= 164 or i == 180 or 184 <= i <= 186 or 195 <= i <= 196):
	# 		groundtruth[i] = 1
	# 	elif(i == 26 or i == 90):
	# 		groundtruth[i] = 2
	# 	else:
	# 		groundtruth[i] = 3

	#Frame 117
	if(path == "images/Frame117.jpg"):
		groundtruth[350] = 2
		groundtruth[351] = 2
		groundtruth[390] = 2
		groundtruth[391] = 2
		groundtruth[607] = 2
		groundtruth[646] = 2
		groundtruth[647] = 2


	#Frame 120
	if(path == "images/Frame120.jpg"):
		groundtruth[351] = 2
		groundtruth[390] = 2
		groundtruth[391] = 2
		groundtruth[608] = 2
		groundtruth[647] = 2
		groundtruth[648] = 2

	#Frame 123
	if(path == "images/Frame123.jpg"):
		groundtruth[390] = 2
		groundtruth[391] = 2
		groundtruth[608] = 2
		groundtruth[609] = 2

	#Frame 124
	elif(path == "images/Frame124.jpg"):
		groundtruth[350] = 2
		groundtruth[351] = 2
		groundtruth[390] = 2
		groundtruth[391] = 2
		groundtruth[608] = 2
		groundtruth[609] = 2

	#Frame 125
	elif(path == "images/Frame125.jpg"):
		groundtruth[350] = 2
		groundtruth[351] = 2
		groundtruth[390] = 2
		groundtruth[391] = 2
		groundtruth[608] = 2
		groundtruth[609] = 2

	#Frame 126
	elif(path == "images/Frame126.jpg"):
		groundtruth[390] = 2
		groundtruth[391] = 2
		groundtruth[608] = 2
		groundtruth[609] = 2

	#Frame 500
	elif(path == "images/Frame500.jpg"):
		groundtruth[68] = 2
		groundtruth[90] = 2

	#Frame 501
	elif(path == "images/Frame501.jpg"):
		groundtruth[68] = 2
		groundtruth[90] = 2

	#Frame 502
	elif(path == "images/Frame502.jpg"):
		groundtruth[68] = 2
		groundtruth[90] = 2

	#Frame 503
	elif(path == "images/Frame503.jpg"):
		groundtruth[68] = 2
		groundtruth[90] = 2

	#Frame 679
	elif(path == "images/Frame679.jpg"):
		groundtruth[689] = 2
		groundtruth[729] = 2

	#Frame 680
	elif(path == "images/Frame680.jpg"):
		groundtruth[689] = 2
		groundtruth[729] = 2

	#Frame 681
	elif(path == "images/Frame681.jpg"):
		groundtruth[689] = 2

	#Frame 682
	elif(path == "images/Frame682.jpg"):
		groundtruth[689] = 2
		groundtruth[777] = 2

	#Frame 1207
	elif(path == "images/Frame_salvamento1207.jpg"):
		groundtruth[90] = 2

	#Frame 1208
	elif(path == "images/Frame_salvamento1208.jpg"):
		groundtruth[90] = 2

	#Frame 1209
	elif(path == "images/Frame_salvamento1209.jpg"):
		groundtruth[90] = 2

	#Frame 1210
	elif(path == "images/Frame_salvamento1210.jpg"):
		groundtruth[90] = 2

	#Frame 1890
	elif(path == "images/Frame_salvamento1890.jpg"):
		groundtruth[34] = 2
		groundtruth[125] = 2

	#Frame 1891
	elif(path == "images/Frame_salvamento1891.jpg"):
		groundtruth[34] = 2
		groundtruth[125] = 2

	#Frame 1892
	elif(path == "images/Frame_salvamento1892.jpg"):
		groundtruth[14] = 2
		# groundtruth[34] = 2
		groundtruth[125] = 2

	#Frame 1893
	elif(path == "images/Frame_salvamento1893.jpg"):
		groundtruth[34] = 2
		groundtruth[125] = 2

	#Frame 1895
	elif(path == "images/Frame_salvamento1895.jpg"):
		groundtruth[14] = 2
		groundtruth[104] = 2
		groundtruth[105] = 2
		groundtruth[125] = 2

	#Frame 1898
	elif(path == "images/Frame_salvamento1898.jpg"):
		groundtruth[105] = 2
		groundtruth[125] = 2

	#Frame 1901
	elif(path == "images/Frame_salvamento1901.jpg"):
		groundtruth[105] = 2
		groundtruth[125] = 2

	#Frame 2500
	elif(path == "images/Frame_salvamento2500.jpg"):
		groundtruth[109] = 2
		groundtruth[110] = 2

	#Frame 3600
	elif(path == "images/Frame3600.jpg"):
		groundtruth[26] = 2
		groundtruth[46] = 2
		groundtruth[47] = 2
		groundtruth[90] = 2
		groundtruth[91] = 2

	#Frame 3734
	elif(path == "images/Frame3734.jpg"):
		groundtruth[85] = 2

	#Frame 4527
	elif(path == "images/Frame4527.jpg"):
		groundtruth[86] = 2
		groundtruth[178] = 2
		groundtruth[179] = 2
		groundtruth[547] = 2

	return groundtruth

def trans_class(groundtruth):

	for i in range(len(groundtruth)):

		if(groundtruth[i] == 0 or groundtruth[i] == 1 or groundtruth[i] == 3 or groundtruth[i] == 4):

			groundtruth[i] = 0

		else:
			groundtruth[i] = 1

	return groundtruth

def process_video(video,out,classi, windowsize_r, windowsize_c):

	idx = 0
	old_blocos_true = []
	old_locations = []
	blocos_desaparecidos = []
	while(video.isOpened()):

		ret, frame = video.read()
		idx+=1

		if(idx >= 1870):
			print "Frame -------------------------------------------------------------", idx
			t0 = time.time()
			if(381 <= idx <= 665 or 1141 <= idx <= 3070 or 3291 <= idx <= 3880 or 4771 <= idx <= 5752):
				res = cv.resize(frame,None,fx=1./8, fy=1./8, interpolation = cv.INTER_CUBIC)
			
			elif(0 <= idx <= 380 or 666 <= idx <= 1140 or 3071 <= idx <= 3290 or 3881 <= idx <= 4770):
				res = cv.resize(frame,None,fx=1./4, fy=1./4, interpolation = cv.INTER_CUBIC)

			if(idx == 381 or idx == 666 or idx == 1141 or idx == 3071 or idx == 3291 or idx == 3881 or idx == 4771):
				old_blocos_true = []
				old_locations = []
				blocos_desaparecidos = []
			if(idx%10 == 0): ### 10 EM 10 FRAMES DESAPARECEM
				blocos_desaparecidos = []

			t1 = time.time() -t0
			t2 = time.time()
			gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
			#lab = cv.cvtColor(res,cv.COLOR_BGR2LAB)
			t3 = time.time() - t2
			# windowsize_r = size_block(res)[0]
			# windowsize_c = size_block(res)[1]

			t4 = time.time()
			window,loc_blocos = divisao_de_blocos(gray,windowsize_r,windowsize_c)
			t5 = time.time() - t4
			#window_2 = blocos_16_16(lab[:,:,1],windowsize_r,windowsize_c)

			#window_3 = blocos_16_16(lab[:,:,2],windowsize_r,windowsize_c)
			t6 = time.time()
			features = features_extraction(window,3)
			t7 = time.time() - t6
			
			t8 = time.time()
			predi = classificator_test(classi,features)
			# print predi[predi==1]
			t9 = time.time() - t8

			t10 = time.time()
			# zeros, idx_true = reconstruct_GT_aux(predi,window,features)
			zeros, blocos_true = reconstruct_GT_aux(predi,window,features)

			print "NEW Blocos ----> ", blocos_true

			if(len(old_blocos_true)!=0 or len(blocos_true)!=0):
				old_blocos_true,old_locations, blocos_desaparecidos, idx_remover = tracking(old_blocos_true,blocos_true,old_locations,blocos_desaparecidos,gray,loc_blocos,windowsize_r, windowsize_c)
				blocos_desaparecidos = remover_desaparecidos(blocos_desaparecidos,idx_remover)
			if(len(old_locations)!=0):
				old_locations = ajuste_locations(old_locations,blocos_desaparecidos,gray,windowsize_r,windowsize_c)
			
			print "--------------"
			print "OLD Blocos SAIDA ----> ",old_blocos_true
			print "OLD LOCATIONS SAIDA ---->",old_locations
			print "BLOCOS DESAPARECIDOS ---->", blocos_desaparecidos
			print "--------------"

			# zeros = reconstruct_GT_aux(ground_truth,window)
			t11 = time.time() - t10

			t12 = time.time()
			# image_reconstructed = invers_blocos_16x16(zeros,gray,windowsize_r,windowsize_c)
			# locations = ajuste_bloco(loc_blocos, idx_true, gray, windowsize_r, windowsize_c)
			locations = ajuste_bloco(loc_blocos, old_blocos_true, gray, windowsize_r, windowsize_c)
			# print "LOCATIONS PARA A MASCARA 0 --> ", locations
			if(len(old_locations)!= 0):
				print "LOCATIONS PARA ADICIONAR---> ", old_locations
				locations = add_locations(old_locations,locations)
			# 	print "LOCATIONS PARA A MASCARA 1 --> ", locations 
			print "LOCATIONS PARA A MASCARA 2 --> ", locations
			image_reconstructed = nova_mascara(locations,gray,windowsize_r,windowsize_c)
			t13 = time.time() - t12

			t14 = time.time()
			contours, hierarchy = cv.findContours(image_reconstructed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
			t15 = time.time() - t14

			t16 = time.time() 
			if(381 <= idx <= 665 or 1141 <= idx <= 3070 or 3291 <= idx <= 3880 or 4771 <= idx <= 5752):
				cv.drawContours(frame, np.multiply(contours,8), -1, (0,0,255), 2)

			elif(0 <= idx <= 380 or 666 <= idx <= 1140 or 3071 <= idx <= 3290 or 3881 <= idx <= 4770):
				cv.drawContours(frame, np.multiply(contours,4), -1, (0,0,255), 2)
			t17 = time.time() - t16

			# print "Resize", t1
			# print "Gray", t3
			# print "Window", t5
			# print "Features", t7
			# print "Predict", t9
			# print "Reconstruct", t11
			# print "Invert", t13
			# print "Contours", t15
			# print "Draw", t17
			cv.imshow("Imagem Reconstuida",frame)
			# out.write(frame)
		

		if cv.waitKey(30) & 0xff == ord('q') or idx == 3650:
			break    

	video.release()
	cv.destroyAllWindows()

def mostrar_blocos(res,window):

	#Mostrar os blocks:
	print "Resolucao Imagem",res.shape
	print "Dimensao dos blocos", window[0].shape
	for i in range(len(window)):
		print i
		showimg("Block Original",resize(window[i],window[i].shape[0],window[i].shape[1]))
		
		glcm = greycomatrix(window[i], [1], [3*np.pi/4], symmetric=True, normed=True)
		print "Media       Desvio        Contraste"
		print np.array([np.mean(window[i]),np.std(window[i]),greycoprops(glcm, 'contrast')[0, 0]])


def ajuste_bloco(object_locations,blocos,gray, windowsize_r, windowsize_c):

	locations = []

	y_final = 0
	x_final = 0

	window_size = 4
	print "AJUSTE BLOCOS ---> ", blocos
	for bloco in blocos:
		location = object_locations[bloco] 

		y_inicial = np.int(location[0])
		x_inicial = np.int(location[1])

		patch_inicial = gray[y_inicial:y_inicial+windowsize_r,x_inicial:x_inicial+windowsize_c]
		glcm = greycomatrix(patch_inicial, [1], [3*np.pi/4], symmetric=True, normed=True)
		contrast_inicial = greycoprops(glcm, 'contrast')[0, 0]

		maior = False

		patch_atual = np.zeros((windowsize_r,windowsize_c))

		for col in range(-window_size,window_size,1):
			x_atual = x_inicial + col
			for row in range(-window_size,window_size,1):
				y_atual = y_inicial + row
				patch_atual = gray[y_atual:y_atual+windowsize_r,x_atual:x_atual+windowsize_c]
				if(patch_atual.shape == (windowsize_r,windowsize_c)):
					glcm = greycomatrix(patch_atual, [1], [3*np.pi/4], symmetric=True, normed=True)
					contrast = greycoprops(glcm, 'contrast')[0, 0]
					
					if(contrast > contrast_inicial):
						maior = True
						contrast_inicial = contrast
						y_final = y_atual
						x_final = x_atual
		
		if(maior==False):
			y_final = y_inicial
			x_final = x_inicial
		locations.append([y_final,x_final,bloco])


	return locations

def ajuste_locations(locations,blocos_desaparecidos,gray, windowsize_r, windowsize_c):
	
	y_final = 0
	x_final = 0

	window_size = 4

	new_locations = []
	for location in locations: 
		continua = False

		y_inicial = location[0]
		x_inicial = location[1]
		bloco = location[2]

		if(len(blocos_desaparecidos)!=0):
			for i in range(len(blocos_desaparecidos)):
				if(blocos_desaparecidos[i] == bloco):
					continua = True
		if(continua == True):
			patch_inicial = gray[y_inicial:y_inicial+windowsize_r,x_inicial:x_inicial+windowsize_c]
			glcm = greycomatrix(patch_inicial, [1], [3*np.pi/4], symmetric=True, normed=True)
			contrast_inicial = greycoprops(glcm, 'contrast')[0, 0]

			maior = False

			patch_atual = np.zeros((windowsize_r,windowsize_c))

			for col in range(-window_size,window_size,1):
				x_atual = x_inicial + col
				for row in range(-window_size,window_size,1):
					y_atual = y_inicial + row
					patch_atual = gray[y_atual:y_atual+windowsize_r,x_atual:x_atual+windowsize_c]
					if(patch_atual.shape == (windowsize_r,windowsize_c)):
						glcm = greycomatrix(patch_atual, [1], [3*np.pi/4], symmetric=True, normed=True)
						contrast = greycoprops(glcm, 'contrast')[0, 0]

						booleana = False
						for k in range(windowsize_r):
						    for l in range(windowsize_c):
						        if(patch_atual[k][l] < 40):
						            booleana = True	
						            break
						
						if(contrast >= contrast_inicial and booleana == True):
							maior = True
							contrast_inicial = contrast
							y_final = y_atual
							x_final = x_atual

			if(maior == True):
				new_locations.append([y_final,x_final,bloco])
			else:
				new_locations.append([y_inicial,x_inicial,bloco])

	return new_locations

def nova_mascara(locations,window,windowsize_r,windowsize_c):

	original_height = window.shape[0]
	original_width = window.shape[1]

	image_reconstructed = np.zeros((original_height,original_width))

	for loc in locations:

		image_reconstructed[loc[0]:loc[0]+windowsize_r,loc[1]:loc[1]+windowsize_c] = np.array([[255]*windowsize_c]*windowsize_r)

	return np.uint8(image_reconstructed)

def proximidade_blocos(bloco1,bloco2):

	proximidade = False
	vizinhos = [bloco1,bloco1-1,bloco1+1,bloco1-40,bloco1+40,bloco1-1-40,bloco1+1-40,bloco1-1+40,bloco1+1+40]

	for vizinho in vizinhos:

		if(bloco2 == vizinho):
			proximidade = True
			break
		
	return proximidade

def proximidade_locations(location1,location2):

	proximidade = False
	
	dist = np.sqrt((location2[1]-location1[1])**2 + (location2[0]-location1[0])**2)
	
	if(dist<10):
		proximidade = True
		
	return proximidade

def add_locations(locations1,locations2):

	final_locations = np.zeros((len(locations1)+len(locations2),2))

	k = 0
	for i in range(len(locations2)):
		final_locations[k] = [locations2[i][0],locations2[i][1]]
		k += 1
	for j in range(len(locations1)):
		final_locations[k] = [locations1[j][0],locations1[j][1]]
		k += 1

	locations2.append(locations1)
	return final_locations

def remover_desaparecidos(blocos_desaparecidos,idx_remover):


	for bloco1 in idx_remover:
		k = 0
		for bloco2 in idx_remover:
			if(bloco1==bloco2):
				k+=1
			if(k==2):
				idx_remover.remove(bloco1)

	for idx in idx_remover:
		blocos_desaparecidos.remove(idx)

	return blocos_desaparecidos
def tracking(old_blocos,new_blocos,old_locations,blocos_desaparecidos,gray,object_locations,windowsize_r, windowsize_c):

	blocos = []
	idx_remover = []
	if(len(new_blocos)>=len(old_blocos)):
		print "NEWBLOCOS >= OLDBLOCOS"
		print " "
		if(len(old_blocos)==0):
			print "OLDBLOCOS == 0"
			print " "
			blocos = new_blocos
			if(len(blocos_desaparecidos)!=0):
				print "0 - BLOCOS DESAPARECIDOS != 0"
				print " "
				for i in range(len(new_blocos)):
					for j in range(len(blocos_desaparecidos)):

						proximidade = proximidade_blocos(new_blocos[i],blocos_desaparecidos[j])

						if(proximidade == True):
							print "NEWBLOCO PROXIMO DESAPARECIDO"
							print " "
							idx_remover.append(blocos_desaparecidos[j])
		else:
			if(len(blocos_desaparecidos)!=0):
				print "1 - BLOCOS DESAPARECIDOS != 0"
				print " "
				for i in range(len(new_blocos)):
					for j in range(len(blocos_desaparecidos)):

						proximidade = proximidade_blocos(new_blocos[i],blocos_desaparecidos[j])

						if(proximidade == True):
							print "NEWBLOCO PROXIMO DESAPARECIDO"
							print " "
							idx_remover.append(blocos_desaparecidos[j])
				
			print "OLDBLOCOS != 0"
			print " "		
					
			blocos = new_blocos
			for i in range(len(old_blocos)):
				k = len(new_blocos)
				for j in range(len(new_blocos)):

					proximidade = proximidade_blocos(old_blocos[i],new_blocos[j])

					if(proximidade == True):
						print "OLDBLOCO PROXIMO NEWBLOCO"
						print " "
						k +=1
					k -=1
					if(k==0):
						print "AJUSTE OLDBLOCO E APPEND NOVA LOCATION E BLOCO DESAPARECIDO"
						print " "
						location = ajuste_bloco(object_locations,[old_blocos[i]],gray, windowsize_r, windowsize_c)
						if(len(old_locations)!=0):
							print "OLD LOCATIONS != 0"
							print " "
							for loc in old_locations:
								proximidade_loc = proximidade_locations(loc,location[0]) ###FALTA VER LOCATION COM DESAPARECIDOS
								if(proximidade_loc == True):
									print "NOVA LOCATION == OLD LOCATION"
									print " "
									old_locations.remove(loc)
									old_locations.append(location[0])
						old_locations.append(location[0])
						blocos_desaparecidos.append(old_blocos[i])
	else:
		print "NEWBLOCOS < OLDBLOCOS"
		print " "
		if(len(blocos_desaparecidos)!=0 & len(new_blocos)!=0):
			print "3 - BLOCOS DESAPARECIDOS !=0"
			print " "
			for i in range(len(new_blocos)):
				for j in range(len(blocos_desaparecidos)):

						proximidade = proximidade_blocos(new_blocos[i],blocos_desaparecidos[j])
						if(proximidade == True):
							print "NEWBLOCO PROXIMO DESAPARECIDO"
							print " "
							idx_remover.append(blocos_desaparecidos[j])					
		if(len(new_blocos)!=0):
			print "NEWBLOCO != 0"
			print " "

			blocos = new_blocos
			for i in range(len(old_blocos)):
				k = len(new_blocos)
				for j in range(len(new_blocos)):

					proximidade = proximidade_blocos(old_blocos[i],new_blocos[j])

					if(proximidade == True):
						print "NEWBLOCO PROXIMO OLDBLOCO"
						print " "
						k +=1
					k -=1
					if(k==0):
						print "BLOCO -->", old_blocos[i]
						print "AJUSTE OLDBLOCO E APPEND NOVA LOCATION E BLOCO DESAPARECIDO"
						print " "
						location = ajuste_bloco(object_locations,[old_blocos[i]],gray, windowsize_r, windowsize_c)
						if(len(old_locations)!=0):
							print "OLD LOCATIONS != 0"
							print " "
							for loc in old_locations:
								proximidade_loc = proximidade_locations(loc,location[0])
								if(proximidade_loc == True):
									print "NOVA LOCATION == OLD LOCATION"
									print " "
									old_locations.remove(loc)
									old_locations.append(location[0])
						old_locations.append(location[0])
						blocos_desaparecidos.append(old_blocos[i])
		else:
			print "NEWBLOCO == 0"
			print " "
			print "AJUSTE OLDBLOCO E APPEND DO BLOCO DESAPARECIDO E APPEND NOVA LOCATION"
			print " "

			for i in range(len(old_blocos)):
				blocos_desaparecidos.append(old_blocos[i])
				location = ajuste_bloco(object_locations,[old_blocos[i]],gray, windowsize_r, windowsize_c)
				if(len(old_locations)!=0):
							print "OLD LOCATIONS != 0"
							print " "
							for loc in old_locations:
								print " "
								print "LOC NA OLD LOCATION -->    ", loc
								print "LOCATION PARA ADICIONAR --> ", location
								print " "
								proximidade_loc = proximidade_locations(loc,location[0]) ###FALTA VER LOCATION COM DESAPARECIDOS
								if(proximidade_loc == True):
									print "NOVA LOCATION == OLD LOCATION"
									print " "
									old_locations.remove(loc)
									old_locations.append(location[0])
				old_locations.append(location[0])
			blocos = new_blocos

	return blocos, old_locations, blocos_desaparecidos, idx_remover

def metrica(window,groundthruthTrue,groundthruthEst):
	conf = confusion_matrix(groundthruthTrue, groundthruthEst)
	print conf

	tn, fp, fn, tp = conf.ravel()

	print tn
	print fp
	print fn
	print tp

	accuracy = (float)(tp + tn)/(groundthruthTrue.shape[0])
	precision = tp / (float)(tp + fp) 
	recall = tp / (float)(tp + fn)
	f_score = 2*((precision*recall)/(precision+recall))

	print "accuracy:", accuracy
	print "precision:", precision	
	print "recall:", recall
	print "f_score:", f_score

def read_gt_predi(path):
	gt_class = np.array([])
	predic_class = np.array([])

	gt = read_file("ground_truth.p")
	predi = read_file("predict.p")

	for i in range(len(path)):
		gt_class = np.append(gt_class,gt[path[i]])
		predic_class = np.append(predic_class,predi[path[i]])

	return gt_class,predic_class


if __name__=="__main__":

	# plt.clf()

	# cap = cv.VideoCapture('images/video_salvamento_aquatico.mp4')


	# out = cv.VideoWriter('images/belele.avi', cv.cv.CV_FOURCC('X','V','I','D'), 20, (1280,720))
	# # out = 0
	# dic = read_file("train_pickle.p")
	# old_feat = dic["features"]
	# old_gt = dic["ground_truth"]

	# show_features_3d_2(old_feat,old_gt)

	# classifier = SVC(kernel = 'linear', C = 1.0)
	# classi = classificator_train(classifier,old_feat,old_gt)

	# process_video(cap,out,classi, size_block_video(cap)[0], size_block_video(cap)[1])


	# img = cv.imread("images/seagull_database_vis002_small.png")
	# path_img = "images/Frame_salvamento1901.jpg"
	path_img = "images/Frame126.jpg"
	img = cv.imread(path_img)

	fator = 4

	res = cv.resize(img,None,fx=1./fator, fy=1./fator, interpolation = cv.INTER_CUBIC)

	gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)

	windowsize_r = size_block(res)[0]
	windowsize_c = size_block(res)[1]

	window, loc_blocos = divisao_de_blocos(gray,windowsize_r,windowsize_c)

	features = features_extraction(window,3)


	# mostrar_blocos(res,window)
	
	ground_truth = groundtruth(window,path_img)
	ground_truth = trans_class(ground_truth)

	update_pickle("ground_truth.p",path_img,ground_truth)
	dic = read_file("ground_truth.p")
	print dic

	# print ground_truth
	# # print ground_truth.shape

	# show_features_3d_2(features,ground_truth)
	# train_pickle(path_img,"features_4.p","ground_truth.p","train_pickle_4.p")

	# write_file("train_pickle_4.p",dict(ground_truth=[],features=[]))
	# write_file("features_4.p",dict())
	# dic = read_file("train_pickle_4.p")
	# print dic["features"].shape
	# print dic

	# update_pickle("ground_truth.p",path_img,ground_truth)
	# dic = read_file("features.p")
	# print dic.keys()
	# print dic
	# update_pickle("features_4.p",path_img,features)
	# dic = read_file("features_4.p")
	# print dic.keys()
	# print dic[path_img].shape

	# show_features_3d_2(features,ground_truth)

	# feat = read_or_write_pickle("3features_train_ship_3Classes.pickle",features,ground_truth,"Erro")
	# zeros = reconstruct_GT_aux(predi,window)
	dic = read_file("train_pickle.p")
	old_feat = dic["features"]
	old_gt = dic["ground_truth"]

	# show_features_3d_2(old_feat,old_gt)
	# show_features_3d_3(old_feat,old_gt, features)
	# print old_feat.shape
	# # print old_gt

	classifier = SVC(kernel = 'linear', C = 1.0)
	classi = classificator_train(classifier,old_feat,old_gt)
	predi = classificator_test(classi,features)

	# write_file("predict.p",dict())
	# update_pickle("predict.p",path_img,predi)
	# dic = read_file("predict.p")
	# print dic[path_img].shape

	# metrica(window,ground_truth,predi)

	# for i in range(len(predi)):
	# 	if(predi[i] == 1):
	# 		print i


	# print "Acerto ", ((np.sum(predi[predi==1]))/np.sum(ground_truth[ground_truth==1]))*100 
	# print "%"
	

	# print "Coef1", classi.coef_
	# print "Number of Support Vectors", classi.support_vectors_

	zeros = reconstruct_GT_aux(ground_truth,window,features)
	# zeros, idx_true = reconstruct_GT_aux(predi,window, features)

	image_reconstructed = invers_blocos_16x16(zeros,gray,windowsize_r,windowsize_c)

	# # locations = ajuste_bloco(loc_blocos, idx_true, gray, windowsize_r, windowsize_c)

	# # print pontos_medios(locations[0],locations[1])
	# # image_reconstructed = nova_mascara(locations,gray,windowsize_r,windowsize_c)

	contours, hierarchy = cv.findContours(image_reconstructed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	
	# # # # # print "Antes",contours
	cv.drawContours(img, np.multiply(contours,fator), -1, (0,0,255), 2)
	# # # # # print "Depois",contours*2

	cv.imshow("Imagem Reconstruida",img)
	cv.waitKey(0)
	cv.destroyAllWindows()


	# #Metricas
	# path_array_img = ["images/Frame123.jpg","images/Frame500.jpg","images/Frame679.jpg","images/Frame_salvamento1207.jpg","images/Frame_salvamento1890.jpg","images/Frame_salvamento1898.jpg","images/Frame_salvamento2500.jpg","images/Frame3600.jpg","images/Frame3734.jpg","images/Frame4527.jpg"]

	# gt,predi = read_gt_predi(path_array_img)

	# metrica(window,gt,predi)

