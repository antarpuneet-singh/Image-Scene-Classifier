from getImageFeatures import get_image_features
import pickle
from getImageDistance import get_image_distance
import numpy as np
import sklearn.metrics


def load_files():
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    test_imagenames=meta["test_imagenames"]
    test_labels=meta["test_labels"]
    
    
    with open('visionHarris.pkl','rb') as f: vision_harris = pickle.load(f)
    train_Features_Harris=vision_harris["trainFeatures"]
    trainLabels_Harris=vision_harris["trainLabels"]
    
    with open('visionRandom.pkl','rb') as f: vision_random = pickle.load(f)
    train_Features_Random=vision_random["trainFeatures"]
    trainLabels_Random=vision_random["trainLabels"]


    return test_imagenames,test_labels,train_Features_Harris,trainLabels_Harris,train_Features_Random,trainLabels_Random




def find_accuracy_and_confusion_matrix(point_method,dist_method,K) :
    
    test_imagenames,test_labels,train_Features_Harris,trainLabels_Harris,train_Features_Random,trainLabels_Random=load_files()
    
    t_labels=np.zeros(160)
    for i, path in enumerate(test_imagenames):
            
            with open(f'../data/{path[:-4]}_{point_method}.pkl', 'rb') as f: w_map = pickle.load(f)
            H = get_image_features(w_map, K)
            
            if(point_method=="Harris"):
                dist=get_image_distance(H, train_Features_Harris, dist_method)
                index=np.argmin(dist)
                t_labels[i]=trainLabels_Harris[index]
                
            elif(point_method=="Random"):
                dist=get_image_distance(H, train_Features_Random, dist_method)
                index=np.argmin(dist)
                t_labels[i]=trainLabels_Random[index]
            
           
    correct = (test_labels == t_labels)
    acc = correct.sum() / correct.size
    confusion_matrix=sklearn.metrics.confusion_matrix(test_labels,t_labels)
    print(f"Accuracy of point method {point_method} with distance method {dist_method} is {acc} ")
    print(f"Confusion matrix of point method {point_method} with distance method {dist_method} is {confusion_matrix}")
    
        

        
if __name__ == "__main__":
    __spec__ = None
    find_accuracy_and_confusion_matrix(point_method="Random", dist_method="chi2", K=500)
    
         
        
        

