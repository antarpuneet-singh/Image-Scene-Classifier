import pickle
from createFilterBank import create_filterbank
import numpy as np
from getImageFeatures import get_image_features



def create_vision_dictionary(method,K):
    
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    train_imagenames = meta['train_imagenames']


    with open(f'dictionary{method}.pkl','rb') as f: d = pickle.load(f)
    
    
    filterBank= create_filterbank()
    
    train_Features=np.zeros((len(train_imagenames),K))
    
    for i, path in enumerate(train_imagenames):
        print(f'processing {path}')
        with open(f'../data/{path[:-4]}_{method}.pkl', 'rb') as f: w_map = pickle.load(f)
        H = get_image_features(w_map, K)
        train_Features[i,:]=H
        
        
    train_labels= meta['train_labels']
        
    
    vision_dict={"dictionary":d,
                  "filterBank":filterBank ,
                  "trainFeatures":train_Features ,
                  "trainLabels":train_labels
                  }
    
    with open(f'vision{method}.pkl','wb') as f: pickle.dump(vision_dict, f)


if __name__ == "__main__":
    __spec__ = None
    create_vision_dictionary(method="Harris", K=500)
    
    
