import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters.rank import entropy
from sklearn.svm import SVC
from skimage.morphology import disk
from skimage.feature import greycomatrix, greycoprops
from skimage import data

from mpl_toolkits.mplot3d import Axes3D

import os


PATCH_SIZE = 8

def resize(img,vx,vy):
	return cv.resize(img,None,fx=vx, fy=vy, interpolation = cv.INTER_CUBIC)

def gray_scale(img):
	return cv.cvtColor(img,cv.COLOR_BGR2GRAY)

def lab_plan(img):
	return cv.cvtColor(img,cv.COLOR_BGR2LAB)

def dezaseis_dezaseis(img, rows, cols, size):
	bloco = np.zeros((size,size)).astype(np.uint8)
	blocos = np.array([bloco]*(int(np.ceil(rows)*np.ceil(cols)))).astype(np.uint8)
	aux = 0

	for r in range(rows):
		for c in range(cols):
			blocos[aux] = img[(r*size) : ((r+1)*size), (c*size) : ((c+1)*size)]
			aux += 1  

	return blocos

def extract_features(img_block_l,img_block_b):
	features = []
	for i in range(len(img_block_l)):
	    glcm = greycomatrix(img_block_l[i], [1], [0, np.pi/2, np.pi/4, 3*np.pi/4], symmetric=True, normed=True)
	    features.append([greycoprops(glcm, 'contrast')[0, 0],np.mean(img_block_b[i]),np.std(img_block_b[i])])

	return features

def escolha(img,features,rows,cols,size):
	bloco = np.zeros((size,size)).astype(np.uint8)
	blocos = np.array([bloco]*(int(np.ceil(rows)*np.ceil(cols)))).astype(np.uint8)
	ground_truth = np.zeros(img.shape[0])

	for i in range(len(features)):
		if(features[i] < 400):
			blocos[i] = np.array([[255]*size]*size)
			# blocos[i] = img[i]
			ground_truth[i] = 1

	return blocos.astype(np.uint8),ground_truth

def reconst_img(img,ground_truth,rows,cols,size):
	bloco = np.zeros((size,size)).astype(np.uint8)
	blocos = np.array([bloco]*(int(np.ceil(rows)*np.ceil(cols)))).astype(np.uint8)

	for i in range(len(ground_truth)):
		if(ground_truth[i] == 0):
			# blocos[i] = np.array([[255]*size]*size)
			blocos[i] = img[i]

	return blocos.astype(np.uint8)


def train(classificador,features,ground_truth):
	classificador.fit(features,ground_truth)

	return classificador

def test(classificador,features):
	return classificador.predict(features)

def juntar_blocos(blocos, rows, cols , size):

	matriz = np.array([0]*(rows*size)*(cols*size))
	matriz = np.reshape(matriz, (rows*size, cols*size))
	r = 0
	c = 0

	for bloco in blocos:
	    matriz[r : r+size, c : c+size] = bloco
	    c = c + size
	    
	    if c == np.shape(matriz)[1]:
	        c = 0
	        r = r + size  
	                                        
	return matriz.astype(np.uint8)

def showimg(title, img):
	cv.imshow(title,img)
	cv.waitKey(0)
	cv.destroyAllWindows()

def writeimg(title, img):
	cv.imwrite(title,img)

def read_file(path):
	file = open(path,"rb")
	dic = pickle.load(file)
	file.close()

	return dic

def write_file(path,dic):
	file = open(path,"wb")
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
	

def show_features_3d_2(features,ground_truth):

	ax = plt.axes(projection='3d')
	ax.scatter3D(features[ground_truth==3,0],features[ground_truth==3,1],features[ground_truth==3,2], color = 'yellow',marker = 's')
	ax.scatter3D(features[ground_truth==2,0],features[ground_truth==2,1],features[ground_truth==2,2], color = 'green',marker = 's')
	ax.scatter3D(features[ground_truth==1,0],features[ground_truth==1,1],features[ground_truth==1,2], color = 'blue',marker = 's')
	ax.scatter3D(features[ground_truth==0,0],features[ground_truth==0,1],features[ground_truth==0,2], color = 'red', marker = 'o')
	ax.set_xlabel("Contraste")
	ax.set_ylabel("Media de B")
	ax.set_zlabel("Desvio de B")
	ax.legend()
	plt.show()

# # Imagem 1
# def ground_truth(img):
# 	ground_truth = [0]*img.shape[0]

# 	for i in range(len(img)):
# 		if 60 <= i <= 86 or 90 <= i <= 117 or 120 <= i <= 218 or 226 <= i <= 324 or 327 <= i <= 378 or 384 <= i <= 408 or 418 <= i <= 437 or 448 <= i <= 465 or 478 <= i <= 495 or i == 509:
# 			ground_truth[i] = 1
# 		elif 223 <= i <= 225:
# 			ground_truth[i] = 2
# 		elif 219 <= i <= 222:
# 			ground_truth[i] = 3
# 		elif 325 <= i <= 326 or 379 <= i <= 383 or 409 <= i <= 417 or 438 <= i <= 447 or 466 <= i <= 477 or 496 <= i <= 508:
# 			ground_truth[i] = 4

# 	return ground_truth

# # Imagem 2
# def ground_truth(img):
# 	ground_truth = [0]*img.shape[0]

# 	for i in range(len(img)):
# 		if 30 <= i <= 55  or 60 <= i <= 282  or 287 <= i <= 313  or 321 <= i <= 509:
# 			ground_truth[i] = 1
# 		elif 283 <= i <= 286:
# 			ground_truth[i] = 2
# 		elif 314 <= i <= 320:
# 			ground_truth[i] = 3

# 	return ground_truth

# # Imagem 3
# def ground_truth(img):
# 	ground_truth = [0]*img.shape[0]

# 	for i in range(len(img)):
# 		if 18 <= i <= 113:
# 			ground_truth[i] = 1
# 		elif 118 <= i <= 119:
# 			ground_truth[i] = 2
# 		elif 114 <= i <= 117:
# 			ground_truth[i] = 3

# 	return ground_truth

# # Imagem 4
# def ground_truth(img):
# 	ground_truth = [0]*img.shape[0]
 
# 	for i in range(len(img)):
# 		if 15 <= i <= 22 or 30 <= i <= 42 or 45 <= i <= 104 or 106 <= i <= 119:
# 			ground_truth[i] = 1
# 		elif i == 105:
# 			ground_truth[i] = 2

# 	return ground_truth

# # Imagem 5
# def ground_truth(img):
# 	ground_truth = [0]*img.shape[0]

# 	for i in range(len(img)):
# 		if 0 <= i <= 119:
# 			ground_truth[i] = 1

# 	return ground_truth

# # Imagem 6
# def ground_truth(img):
# 	ground_truth = [0]*img.shape[0]

# 	for i in range(len(img)):
# 		if 0 <= i <= 22 or 26 <= i <= 39 or 43 <= i <= 49 or 58 <= i <= 62 or 74 <= i <= 78 or 87 <= i <= 91 or i == 104 or 118 <= i <= 119:
# 			ground_truth[i] = 1
# 		elif 23 <= i <= 25 or 40 <= i <= 42 or 50 <= i <= 57 or 63 <= i <= 73 or 79 <= i <= 86 or 92 <= i <= 103 or 105 <= i <= 117:
# 			ground_truth[i] = 4

# 	return ground_truth


if __name__=="__main__":

	classific = SVC()

	nameImg = "images/Frame3580.jpg"

	fig = plt.figure(figsize=(5,5))

	img = cv.imread(nameImg)

	res = resize(img,0.125,0.125)

	lab = lab_plan(res)

	rows = lab.shape[0]/PATCH_SIZE
	cols = lab.shape[1]/PATCH_SIZE

	img_dezaseis_l = dezaseis_dezaseis(lab[:,:,0],rows,cols,PATCH_SIZE)
	img_dezaseis_b = dezaseis_dezaseis(lab[:,:,2],rows,cols,PATCH_SIZE)

	# # # #Mostrar os blocks:
	# for i in range(len(img_dezaseis_l)):
	# 	print i
	# 	showimg("Blocks",resize(img_dezaseis_l[:][i],10,10))
	# 	glcm = greycomatrix(img_dezaseis_l[i], [1], [0, np.pi/2, np.pi/4, 3*np.pi/4], symmetric=True, normed=True)
	# 	print np.array([greycoprops(glcm, 'contrast')[0, 0],np.mean(img_dezaseis_b[i]),np.std(img_dezaseis_b[i])])

	# ground_truth = ground_truth(img_dezaseis_l)
	# # print ground_truth

	features = extract_features(img_dezaseis_l,img_dezaseis_b)

	# img_reconst = reconst_img(img_dezaseis_l,ground_truth,rows,cols,PATCH_SIZE)

	# final = juntar_blocos(img_reconst,rows,cols,PATCH_SIZE)

	# showimg("Final", resize(final,2,2))

	# # 
	# showimg("Original",lab[:,:,0])
	# showimg("Block",resize(img_dezaseis[:][0],10,10))
	# print res.shape

	# blocos_naomar,ground_truth = escolha(img_dezaseis_1,features_1,rows,cols,PATCH_SIZE)

	# final = juntar_blocos(blocos_naomar,rows,cols,PATCH_SIZE)

	# showimg("Final",resize(final,2,2))

	# # Feutures
	# write_file('features.p',dict())
	# fea = read_file('features.p')
	# print dic
	# print update_pickle('features.p',nameImg,features)

	# # Ground Truth
	# write_file('ground-truth.p',dict())
	# update_pickle('ground-truth.p',nameImg,ground_truth)
	# ground = read_file('ground-truth.p')
	# print ground

	# show_features_3d_2(np.array([read['features']]),np.array([read['ground_truth']]))

	# treino = train(classific,read['features'],read['ground_truth'])
	# teste = test(treinos,features)
	# # print teste

	# img_reconst = reconst_img(img_dezaseis_l,teste,rows,cols,PATCH_SIZE)

	# final = juntar_blocos(img_reconst,rows,cols,PATCH_SIZE)

	# showimg("Final", resize(final,2,2))