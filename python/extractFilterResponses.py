import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *


def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))        

    # -----fill in your implementation here --------
    
    I= rgb2lab(I)
    
    filterResponse_lst=[]   
    
    for filter in filterBank:
        dimension_lst=[]
        for i in range(3):
            dimension=imfilter(I[:,:,i], filter)
            dimension_lst.append(dimension)
        fR=np.stack(tuple(dimension_lst), axis=-1)
        filterResponse_lst.append(fR)
    
    filterResponses = np.concatenate(tuple(filterResponse_lst),axis=2)


    # ----------------------------------------------
    
    return filterResponses


# img = cv.imread('D:/Desktop/MM_811/Assignment 1/Project/data/airport/sun_aerinlrdodkqnypz.jpg',1)
# # cv.imshow('image',img)

# img = img.astype(np.float64)

# img= rgb2lab(img)

# gaussianScales = [1, 2, 4, 8, np.sqrt(2)*8]
# logScales      = [1, 2, 4, 8, np.sqrt(2)*8]
# dxScales       = [1, 2, 4, 8, np.sqrt(2)*8]
# dyScales       = [1, 2, 4, 8, np.sqrt(2)*8]

# filterBank = []

# for scale in gaussianScales:
#     filter = fspecial_gaussian(2*np.ceil(scale*2.5)+1, scale)
#     filterBank.append(filter)

# for scale in logScales:
#     filter = fspecial_log(2*np.ceil(scale*2.5)+1, scale)
#     filterBank.append(filter)

# for scale in dxScales:
#     filter0 = fspecial_gaussian(2 * np.ceil(scale * 2.5) + 1, scale)
#     filter = imfilter(filter0, np.array([[-1, 0, 1]]))
#     filterBank.append(filter)

# for scale in dyScales:
#     filter0 = fspecial_gaussian(2 * np.ceil(scale * 2.5) + 1, scale)
#     filter = imfilter(filter0, np.array([[-1], [0], [1]]))
#     filterBank.append(filter)




# P_lst=[]
# for filter in filterBank:
#     I=imfilter(img[:,:,0], filter)
#     J=imfilter(img[:,:,1], filter)
#     K=imfilter(img[:,:,2], filter)
    
#     P=np.stack((I,J,K), axis=-1)
#     P_lst.append(P)

# T=np.concatenate(tuple(P_lst),axis=2) 



# filterResponse_lst=[]   
# for filter in filterBank:
#     dimension_lst=[]
#     for i in range(3):
#         I=imfilter(img[:,:,i], filter)
#         dimension_lst.append(I)
#     fR=np.stack(tuple(dimension_lst), axis=-1)
#     filterResponse_lst.append(fR)

# finalImage = np.concatenate(tuple(filterResponse_lst),axis=2) 
    
# demo=finalImage[:,:,20]    
# print(demo)    

# from matplotlib import pyplot as plt
# plt.imshow(demo, interpolation='nearest')
# plt.show()