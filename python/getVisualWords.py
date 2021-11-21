import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
import cv2 as cv
import pickle
from createFilterBank import create_filterbank

def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------
    
    filtered_image = extract_filter_responses(I, filterBank)
    reshaped_filtered_image=filtered_image.reshape((filtered_image.shape[0]*filtered_image.shape[1],filtered_image.shape[2]))

    dist = cdist(reshaped_filtered_image,dictionary,metric='euclidean')

    sorted_array=np.argmin(dist, axis=1)
    wordMap=sorted_array.reshape((filtered_image.shape[0],filtered_image.shape[1]))

    # ----------------------------------------------

    return wordMap



# img = cv.imread('D:/Desktop/MM_811/Assignment 1/Project/data/airport/sun_aerinlrdodkqnypz.jpg',1)
# cv.imshow('image',img)
# img = img.astype(np.float64)

# with open('dictionaryRandom.pkl','rb') as f: random_dict = pickle.load(f)

# filtered_image = extract_filter_responses(img, create_filterbank())

# reshaped_filtered_image=filtered_image.reshape((filtered_image.shape[0]*filtered_image.shape[1],filtered_image.shape[2]))

# dist = cdist(reshaped_filtered_image,random_dict,metric='euclidean')
# sorted_array=np.argmin(dist, axis=1)
# final=sorted_array.reshape((filtered_image.shape[0],filtered_image.shape[1]))


# first=np.ones((3,2,2))
# second=np.ones((10,2))

# first=first.reshape(6,2)
# dist = cdist(first,second,metric='euclidean')

