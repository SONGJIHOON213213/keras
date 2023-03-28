import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model


'''
7/26 삼성전자 시가 예측 
'''

# data 
# samsung, sk 필요 데이터 추출
path = './keras/_data/시험/'

samsung1 = pd.read_csv(path + '삼성전자 주가2.csv', encoding = 'cp949',index_col=0)
hyun = pd.read_csv(path + '현대자동차.csv',encoding = 'cp949',index_col=0)

samsung1 = samsung1[['시가','고가','저가','종가', '거래량','금액(백만)','신용비']]
hyun = hyun[['시가','고가','저가','종가', '거래량','금액(백만)','신용비']]

samsung1 = samsung1.to_numpy()
hyun = hyun.to_numpy()

def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset)-size+1): # range(10-4=6) -> 6번동안 반복. 10개의 데이터를 5개씩 분리하기 위한 방법 
        subset = dataset[i : (i+size)] # dataset[0:5] -> dataset 0부터 4번째 값까지 
        aaa.append(subset)
    return np.array(aaa)





