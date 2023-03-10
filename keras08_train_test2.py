import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1])

#[실습] 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라
x_train = x[:7] #0을명시하지않아도 동일한결과
print(x_train) # [1 2 3 4 5 6 7] # 7-1
x_test = x[7:] #7부터 끝까지 의미
print(x_test)  #[ 8  9 10]
y_train = y[:7] 
y_test = y[7:]


"""
print(x_train.shape, x_test.shape) #(7,) (3)
print(y_train.shape, y_test.shape) #(7,) (3) 
"""

# 7:3 으로 잘라라 의 의미
# [1 2 3 4 5 6 7]
#  [ 8  9 10]