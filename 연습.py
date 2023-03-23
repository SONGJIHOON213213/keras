from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense,Input,Conv2D,Flatten
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
#1. 데이터
path = './keras/_data/houseprice/'

train_csv = pd.read_csv(path +'train.csv', index_col = 0)
test_csv  = pd.read_csv(path + 'test.csv', index_col = 0)

print(train_csv.shape, test)