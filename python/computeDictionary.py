import pickle
from getDictionary import get_dictionary
import numpy as np


meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']


# -----fill in your implementation here --------

alpha=200
K=500
imgPaths=train_imagenames
# method="Random"
method="Harris"

dict = get_dictionary(imgPaths, alpha, K, method)

with open('dictionaryHarris.pkl','wb') as f: pickle.dump(dict, f)


# ----------------------------------------------





