import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import datetime


date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
def RMSE(x,y):
    return np.sqrt(mean_squared_error(x,y))

def split_x(dt, st):
    a = []
    for i in range(len(dt)-st-1):
        b = dt[i:(i+st)]
        a.append(b)
    return np.array(a) 

# 1. 데이터
# 1.1 경로, 가져오기
path = './_data/seoul/'
path_save = './_save/seoul/'

train  = pd.read_csv(path + 'train.csv', index_col=0, encoding='cp949')
data = pd.read_csv(path + 'test.csv', index_col=0, encoding='cp949')


