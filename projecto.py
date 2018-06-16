import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters.rank import entropy
from sklearn.svm import SVC
from skimage.morphology import disk
from skimage.feature import greycomatrix, greycoprops
from skimage import data


fig = plt.figure(figsize=(5,5))

img = cv.imread("images/seagull_database_vis001_small.png")

PATCH_SIZE = 16

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
		if(ground_truth[i] == 1):
			# blocos[i] = np.array([[255]*size]*size)
			blocos[i] = img[i]

	return blocos.astype(np.uint8)


def train(classificador,features,ground_truth):
	return classificador.fit(features,ground_truth)

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

# Imagem 1
def ground_truth(img):
	ground_truth = np.zeros(img.shape[0])
	for i in range(len(img)):
		if 2 <= i <= 10 or 15 <= i <= 50 or 53 <= i <= 85 or 89 <= i <= 97 or 105 <= i <= 111 or i == 119:
			ground_truth[i] = 1
		elif 51 <= i <= 52:
			ground_truth[i] = 2
		elif 86 <= i <= 88 or 98 <= i <= 104 or 112 <= i <= 118:
			ground_truth[i] = 3

	return ground_truth

# # Imagem 2
# def ground_truth(img):
# 	ground_truth = np.zeros(img.shape[0])
# 	for i in range(len(img)):
# 		if 1 <= i <= 13 or 15 <= i <= 66 or 69 <= i <= 81 or 85 <= i <= 119:
# 			ground_truth[i] = 1
# 		elif 67 <= i <= 68 or 82 <= i <= 84:
# 			ground_truth[i] = 2

# 	return ground_truth

def write_or_read_pickle(bool,path,features,ground_truth):
	mydict = dict([('features',[]),('ground_truth',[])])
	if bool == 'write':
		file = open(path,'wb')
		pickle.dump(mydict,file)
		file.close()
	elif bool == 'read':
		file = open(path,'rb')
		pickle.load(file)
		file.close()
	elif bool == 'update':
		file = open(path,'wb')
		mydict['ground_truth'].append(ground_truth)
		mydict['features'].append(features)
		pickle.dump(mydict,file)
		file.close()
	return mydict

classific = SVC()

res = resize(img,0.25,0.25)
lab = lab_plan(res)

rows = lab.shape[0]/PATCH_SIZE
cols = lab.shape[1]/PATCH_SIZE

img_dezaseis_l = dezaseis_dezaseis(lab[:,:,0],rows,cols,PATCH_SIZE)
img_dezaseis_b = dezaseis_dezaseis(lab[:,:,2],rows,cols,PATCH_SIZE)

# showimg("Original",res)

##Mostrar os blocks:
# for i in range(len(img_dezaseis_l)):
# 	print i
# 	showimg("Blocks",resize(img_dezaseis_l[:][i],10,10))

ground_truth = ground_truth(img_dezaseis_l)

features = extract_features(img_dezaseis_l,img_dezaseis_b)

# img_reconst = reconst_img(img_dezaseis_l,ground_truth,rows,cols,PATCH_SIZE)

# final = juntar_blocos(img_reconst,rows,cols,PATCH_SIZE)

# showimg("Final", resize(final,2,2))

# showimg("Original",lab[:,:,0])
# showimg("Block",resize(img_dezaseis[:][0],10,10))
# print res.shape

# blocos_naomar,ground_truth = escolha(img_dezaseis_1,features_1,rows,cols,PATCH_SIZE)

# final = juntar_blocos(blocos_naomar,rows,cols,PATCH_SIZE)

# showimg("Final",resize(final,2,2))

# write = write_or_read_pickle('write','classificado.p',features,ground_truth)

# read = write_or_read_pickle('read','classificado.p',features,ground_truth)

# update = write_or_read_pickle('update','classificado.p',features,ground_truth)

# print write
# print read
# print update

# treino = train(classific,read['features'],read['ground_truth'])
# teste = test(classific,features)
# print teste