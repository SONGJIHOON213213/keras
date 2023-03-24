from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, SimpleRNN, LSTM, Conv1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터
path = 'keras/_data/jena_climate/'
#70:20:10  트레인테스트

zf = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col=0)
print(zf.shape) #(420551, 14)