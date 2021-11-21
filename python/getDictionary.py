import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))
    

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv.imread('../data/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        
        # -----fill in your implementation here --------
        
        filtered_image = extract_filter_responses(image, filterBank)
        
        if(method=="Random"):
            r_points= get_random_points(image, alpha)
            filtered_pixels = get_random_filtered_pixels(r_points, filtered_image)
        
        elif(method=="Harris"):
            h_points= get_harris_points(image,alpha, 0.04)
            filtered_pixels = get_harris_filtered_pixels(h_points, filtered_image)

        pixelResponses[alpha*i : alpha*(i+1)] = filtered_pixels
        

        # ----------------------------------------------

    dictionary = KMeans(n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_
    return dictionary

def get_random_filtered_pixels(r_points,filtered_image):
    lst=[]
    for i,j in r_points:
        lst.append(filtered_image[i,j,:])

    a=np.array(lst)
    return a

def get_harris_filtered_pixels(h_points,filtered_image):
    lst=[]
    for i,j in h_points:
        lst.append(filtered_image[i,j,:])

    a=np.array(lst)
    return a    

# from computeDictionary import *

# meta = pickle.load(open('../data/traintest.pkl', 'rb'))
# train_imagenames = meta['train_imagenames']
# imgPaths=train_imagenames
# filterBank = create_filterbank()
# alpha=200
# K=500
# pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))
# # method="Random"
# method="Harris"


# for i, path in enumerate(imgPaths[0:1]):
#         print('-- processing %d/%d' % (i, len(imgPaths)))
#         image = cv.imread('../data/%s' % path)
#         image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        
#         filtered_image = extract_filter_responses(image, filterBank)
        
#         if(method=="Random"):
#             r_points= get_random_points(image, alpha)
#             filtered_pixels = get_random_filtered_pixels(r_points, filtered_image)
        
#         elif(method=="Harris"):
#             h_points= get_harris_points(image,alpha, 0.04)
#             filtered_pixels = get_harris_filtered_pixels(h_points, filtered_image)

#         pixelResponses[alpha*i : alpha*(i+1)] = filtered_pixels
        
            
    
        
        



        
        
        
        
        
        
        
        
        
























