import numpy as np

def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    
    h=np.zeros(dictionarySize)
    unique,counts=np.unique(wordMap,return_counts=True)
    k=dict(zip(unique,counts))
    
    for key,value in k.items():
        h[key]=value
        
    h = h/h.sum()    
    # ----------------------------------------------
    
    return h




       