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

def blocos_16_16(image,window,windowsize_r,windowsize_c):

	k = 0
	for r in range(0,image.shape[0] , windowsize_r):
	    for c in range(0,image.shape[1], windowsize_c):
	    	# print r,c
	        window[k] = image[r:r+windowsize_r,c:c+windowsize_c]
	        k+=1
	return np.uint8(window)

def features_extraction(window,window2, n_features):

	features = np.zeros((window.shape[0], n_features))

	for bloco in range(len(window)):

		glcm = greycomatrix(window[bloco], [1], [0, np.pi/2, np.pi, 3*np.pi/2], symmetric=True, normed=True)
		contrast = greycoprops(glcm, 'contrast')[0, 0]
		ent = entropy(window[bloco])

		features[bloco][0] = contrast
		features[bloco][1] = ent
		#features[bloco][1] = mean_extraction(window2[bloco])

	return features

def features_extraction(window, n_features):

	features = np.zeros((window.shape[0], n_features))

	for bloco in range(len(window)):

		glcm = greycomatrix(window[bloco], [1], [0, np.pi/2, np.pi, 3*np.pi/2], symmetric=True, normed=True)
		contrast = greycoprops(glcm, 'contrast')[0, 0]
		ent = entropy(window[bloco])

		if(n_features == 1):

			features[bloco][0] = contrast

		if(n_features ==2):
			features[bloco][0] = contrast
			features[bloco][1] = ent
			

	return features

def mean_extraction(bloco):

	return np.mean(bloco)

def norm_matrix_one_feature(matrix,num_feature):

	matrix[:,num_feature] = matrix[:,num_feature]/LA.norm(matrix[:,num_feature])

	return matrix

def norm_matrix(matrix):

	for i in range(matrix.shape[1]):
		matrix[:,i] = matrix[:,i]/LA.norm(matrix[:,i])

	return matrix

def features_filter_norm(features,window):

	zeros = np.zeros(window.shape)
	ground_truth = np.zeros(window.shape[0])

	for i in range(features.shape[0]):

		if(features[i][0] > 0.03 and features[i][1] > 0.055):
			
			zeros[i] = window[i]
			ground_truth[i] = 1


	return zeros,ground_truth

def features_filter(features,window,n_features):

	zeros = np.zeros(window.shape)
	ground_truth = np.zeros(window.shape[0])

	for i in range(features.shape[0]):

		if(n_features==1):
			if(features[i] < 400 ):
				
				zeros[i] = window[i]
				ground_truth[i] = 1

		if(n_features==2):

			if(features[i][0] > 400 and features[i][1] > 4):
				
				zeros[i] = window[i]
				ground_truth[i] = 1


	return zeros,ground_truth

def invers_blocos_16x16(blocos,original_height,original_width,windowsize_r,windowsize_c):
    image_reconstructed = np.zeros((original_height,original_width))
    k = 0
    for i in range(0,image_reconstructed.shape[0],windowsize_r):
        for j in range(0,image_reconstructed.shape[1],windowsize_c):
            image_reconstructed[i:i+windowsize_r,j:j+windowsize_c] = blocos[k]
            k+=1
    return np.uint8(image_reconstructed)

def reconstruct_GT_aux(ground_truth,window):

	zeros = np.zeros(window.shape)

	for i in range(len(ground_truth)):

		if(ground_truth[i] == 0):

			zeros[i] = np.array([[255]*window[i].shape[1]]*window[i].shape[0])

	return zeros

def classificator_train(classificator,features,ground_truth):
	
	classificator.fit(features,ground_truth)
	
	return classificator

def classificator_test(classificator,features):

	return classificator.predict(features)

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

def show_features_1d(features):

	plt.plot(features,len(features)*[1])
	plt.show()

def show_features_1d(features,ground_truth):

	plt.plot(features[ground_truth==1],len(features[ground_truth==1])*[1],'xg')
	plt.plot(features[ground_truth==0],len(features[ground_truth==0])*[1],'or')
	plt.show()

def show_features_2d(features):

	plt.scatter(features[:,0],features[:,1],color = 'blue',marker = 's')
	plt.scatter(features[:,0],features[:,1],color = 'red', marker = 'o')

	plt.xlabel('Contraste'),plt.ylabel('Entropia')
	plt.show()

def show_features_2d(features,ground_truth):

	plt.scatter(features[ground_truth==1,0],features[ground_truth==1,1],color = 'blue',marker = 's')
	plt.scatter(features[ground_truth==0,0],features[ground_truth==0,1],color = 'red', marker = 'o')

	plt.xlabel('Contraste'),plt.ylabel('Entropia')
	plt.show()

def show_features_3d(features):

	plt.plot(features,len(features)*[1],'xg')
	plt.show()

def show_features_3d(features,ground_truth):

	plt.plot(features[ground_truth==1,:],len(features[ground_truth==1,:])*[1],'xg')
	plt.plot(features[ground_truth==0,:],len(features[ground_truth==0,:])*[1],'or')
	plt.show()

def size_block(video):

	ret, frame = video.read()

	frame_height = frame.shape[0]
	frame_width = frame.shape[1]

	r = 0
	c = 0
	dimensoes = [14,15,16]

	r = [dimensao for dimensao in dimensoes if frame_height%dimensao == 0]	
	c = [dimensao for dimensao in dimensoes if frame_width%dimensao == 0]	


	return r[-1],c[-1]

def process_video(video,classi, windowsize_r, windowsize_c, fator):

	while(video.isOpened()):

		ret, frame = video.read()
		print ret
		res = cv.resize(frame,None,fx=1./fator, fy=1./fator, interpolation = cv.INTER_CUBIC)

		gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
		lab = cv.cvtColor(res,cv.COLOR_BGR2LAB)
		

		original_height = gray.shape[0]
		original_width = gray.shape[1]


		window = np.zeros((original_width/windowsize_c*original_height/windowsize_r,windowsize_r,windowsize_c))
		window2 = np.zeros((original_width/windowsize_c*original_height/windowsize_r,windowsize_r,windowsize_c))

		window = blocos_16_16(gray,window,windowsize_r,windowsize_c)

		features = features_extraction(window,1)

		# zeros,ground_truth = features_filter(features,window,1)

		
		predi = classificator_test(classi,features)
		
		
		zeros = reconstruct_GT_aux(predi,window)
		# zeros = reconstruct_GT_aux(ground_truth,window)

		image_reconstructed = invers_blocos_16x16(zeros,original_height,original_width,windowsize_r,windowsize_c)

		contours, hierarchy = cv.findContours(image_reconstructed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		cv.drawContours(frame, np.multiply(contours,fator), -1, (0,0,255), 2)

		cv.imshow("Imagem Reconstuida",frame)
		

		if cv.waitKey(30) & 0xff == ord('q'):
			break    

	video.release()
	cv.destroyAllWindows()


# def recordvideo(cap,frame):

# 	# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# 	# We convert the resolutions from float to integer.
# 	frame_width = int(cap.get(3))
# 	frame_height = int(cap.get(4))
	 
# 	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# 	out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# 	# Write the frame into the file 'output.avi'
#     out.write(frame)
 
	

if __name__=="__main__":

	plt.clf()

	cap = cv.VideoCapture('images/video2.mp4')

	dic = read_file("features_train.pickle")
	old_feat = dic["features"]
	old_gt = dic["ground_truth"]

	# show_features_1d(old_feat,old_gt)
	# # # # print "RESULTADO -->", np.sum(features == old_feat[510:]) 

	# print old_feat.shape
	# # # # print old_gt.shape
	classifier = SVC()
	classi = classificator_train(classifier,old_feat,old_gt)

	process_video(cap,classi, size_block(cap)[0], size_block(cap)[1],4)



	
