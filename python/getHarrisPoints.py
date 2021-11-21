import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter


def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------
    #Finding the gradients
    dy, dx = np.gradient(I)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    
    
    sum_filter=np.ones((3,3))
    
    #Generating covariance matrix
    Sxx=ndimage.convolve(Ixx,sum_filter, mode="constant")
    Sxy=ndimage.convolve(Ixy,sum_filter, mode="constant")
    Syy=ndimage.convolve(Iyy,sum_filter, mode="constant")
    
    m1=np.stack((Sxx,Sxy),axis=-1)
    m2=np.stack((Sxy,Syy),axis=-1)
    
    M= np.stack((m1,m2),axis=-1)
    
    #Finding determinant and trace
    D=np.linalg.det(M)
    t=np.trace(M,axis1=2, axis2=3)

    R=D - k*(t**2)
    RD= R.flatten()
    idx = np.argpartition(RD, -alpha)[-alpha:]
    indices = idx[np.argsort((-RD)[idx])]
    
    x,y=np.unravel_index(indices,R.shape)
    points = np.vstack((x, y)).T



    # ----------------------------------------------
    
    return points

# img = cv.imread('D:/Desktop/MM_811/Assignment 1/Project/data/airport/sun_aerinlrdodkqnypz.jpg',1)
# img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# img.max()
# img=img/255.0

# dy, dx = np.gradient(img)
# Ixx = dx**2
# Ixy = dy*dx
# Iyy = dy**2

# sum_filter=np.ones((3,3))

# Sxx=ndimage.convolve(Ixx,sum_filter, mode="constant")
# Sxy=ndimage.convolve(Ixy,sum_filter, mode="constant")
# Syy=ndimage.convolve(Iyy,sum_filter, mode="constant")

# m1=np.stack((Sxx,Sxy),axis=-1)
# m2=np.stack((Sxy,Syy),axis=-1)

# M= np.stack((m1,m2),axis=-1)

# D=np.linalg.det(M)
# t=np.trace(M,axis1=2, axis2=3)
# k=0.04
# R=D - k*(t**2)
# a=20

# RD= R.flatten()
# print(RD)
# RD.shape

# idx = np.argpartition(RD, -a)[-a:]
# indices = idx[np.argsort((-RD)[idx])]


# print(indices)
# RD[14018]
# max(RD)
# np.amax(R)
# R[95,99]
# x,y=np.unravel_index(indices,R.shape)
# points = np.vstack((x, y)).T
