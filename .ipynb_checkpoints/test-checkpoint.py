import pickle
import pandas as pd

with open('./source/data.pickle', 'rb') as handle:
    data = pickle.load(handle)
    
print (data[0])