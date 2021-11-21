import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from matplotlib import pyplot as plt
from getImageFeatures import get_image_features
import pickle
from getVisualWords import get_visual_words
from skimage.color import label2rgb
from getImageDistance import get_image_distance

img = cv.imread('D:/Desktop/MM_811/Assignment 1/Project/data/landscape/sun_bwmhmdvisxfogksv.jpg',1)

with open('../data/landscape/sun_bwmhmdvisxfogksv_Harris.pkl', 'rb') as f: w_map = pickle.load(f)

labeled= label2rgb(w_map,img)
cv.imshow('image',labeled)

plt.imshow(labeled,interpolation='nearest')
plt.show()


with open('visionRandom.pkl','rb') as f: random_dict = pickle.load(f)
hists=random_dict["trainFeatures"]
dist=get_image_distance(hists[0], hists[1:], method="Euclidean")
ddist=dist.reshape(1,dist.shape[0])
min=np.argmin(dist)

A = np.array([9,3,2,10,1,1,7,8,9,2,4])
z=np.argsort(A)[:4]


a=np.array([[2,4,6]])
b=np.array([[3,4,6],[4,4,6]])
chi2dist(a, b)

w_map=get_visual_words(img, harris_dict, create_filterbank())
K=500

H=get_image_features(w_map, K)




post=extract_filter_responses(img, create_filterbank())
cv.imshow('image',img)
# cv.imshow('filter1',post[:,:,35])
plt.imshow(post[:,:,20],interpolation='nearest')
plt.show()

alpha=500
k=0.04


r_points= get_random_points(img, alpha)

h_points= get_harris_points(img, alpha, k)

# cv.imshow('image',img)
for i in range(len(h_points)) :
    image = cv.circle(img, h_points[i], radius=1, color=(0, 0, 255), thickness=-1)
cv.imshow('image',image)


meta = pickle.load(open('../data/traintest.pkl', 'rb'))
test_imagenames=meta["test_imagenames"]
test_labels=meta["test_labels"]


with open('visionHarris.pkl','rb') as f: harris_dict = pickle.load(f)
train_Features=harris_dict["trainFeatures"]
trainLabels=harris_dict["trainLabels"]

t_labels=np.zeros(160)
for i, path in enumerate(test_imagenames[0:1]):
        
        with open(f'../data/{path[:-4]}_Harris.pkl', 'rb') as f: w_map = pickle.load(f)
        print(path)
        H = get_image_features(w_map, 500)
        dist=get_image_distance(H, train_Features, "chi2")
        dist=dist.reshape(dist.shape[0],)
        neighbours = np.argsort(dist)[:20]
        print(neighbours)
        
        neighbours_labels=[]
        for neighbour in neighbours:
            neighbours_labels.append(trainLabels[neighbour])
        nL=np.array(neighbours_labels)    
        print(nL)
        unique,counts=np.unique(nL,return_counts=True)
        neighbourhood=dict(zip(unique,counts))
        print(neighbourhood)
        print(max(neighbourhood, key=neighbourhood.get))





    









