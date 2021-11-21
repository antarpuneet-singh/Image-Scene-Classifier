
import pickle
from scipy.spatial.distance import cdist
import numpy as np
from utils import *


def get_image_distance(hist1,histSet,method):
    
    if (method=="Euclidean"):
        hist1=hist1.reshape(1,hist1.shape[0])
        dist= cdist(hist1,histSet,metric='euclidean')
        dist=dist.reshape(histSet.shape[0],1)
        return dist
    
    elif (method=="chi2"):
        dist= np.array(chi2dist(hist1, histSet))
        dist=dist.reshape(histSet.shape[0],1)
        return dist






