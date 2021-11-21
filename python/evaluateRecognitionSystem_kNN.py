from getImageFeatures import get_image_features
import pickle
from getImageDistance import get_image_distance
import numpy as np
import sklearn
from matplotlib import pyplot as plt

def load_parameters():
    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    test_imagenames=meta["test_imagenames"]
    test_labels=meta["test_labels"]
    
    
    with open('visionRandom.pkl','rb') as f: random_dict = pickle.load(f)
    train_Features=random_dict["trainFeatures"]
    trainLabels=random_dict["trainLabels"]
    
    return test_imagenames,test_labels,train_Features,trainLabels

def find_labels_knn(k) :
    
    test_imagenames,test_labels,train_Features,trainLabels=load_parameters()
    t_labels=np.zeros(160)
    for i, path in enumerate(test_imagenames):
            
            with open(f'../data/{path[:-4]}_Random.pkl', 'rb') as f: w_map = pickle.load(f)
            H = get_image_features(w_map, 500)
            dist=get_image_distance(H, train_Features, "chi2")
            dist=dist.reshape(dist.shape[0],)
            neighbours = np.argsort(dist)[:k]
            
            neighbours_labels=[]
            for neighbour in neighbours:
                neighbours_labels.append(trainLabels[neighbour])
            nL=np.array(neighbours_labels)    
            unique,counts=np.unique(nL,return_counts=True)
            neighbourhood=dict(zip(unique,counts))
            
            t_labels[i] = max(neighbourhood, key=neighbourhood.get)

    correct = (test_labels == t_labels)
    acc = correct.sum() / correct.size   
    confusion_matrix=sklearn.metrics.confusion_matrix(test_labels,t_labels)
    print(f'Accuracy for {k} neighbours : {acc}')       
           
        
    return acc,confusion_matrix   
               
if __name__ == "__main__":
    __spec__ = None
    
    accuracy=[]
    confusion_mat=[]
    total_k=40
    for k in range(1,total_k+1):
        acc,con=find_labels_knn(k)
        accuracy.append(acc)
        confusion_mat.append(con)
    max_acc=max(accuracy)
    max_index=accuracy.index(max_acc)
    print(max_index,confusion_mat[max_index])
    plt.plot(accuracy)   
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(len(accuracy)), np.arange(1, len(accuracy)+1))


    

    
