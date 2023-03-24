import numpy as np
datasets = np.array(range(1,41)).reshape(10, 4)#1~40까지 백터형태 1차원 42개열 10개행
print(datasets)
print(datasets.shape) #(10,4)

x_data = datasets[:,:-1] #   : 모든행 이란 의미 2열이므로 3이라고 써야된다.
y_data = datasets[:, -1]

timesteps = 9

def split_x(dataset, timesteps):
    ans=list()
    for i in range (len(dataset)-timesteps): #타임스텝을 빼주는이유 타임스텝을 상쇄시킬라고
        forlistgen = dataset[i : i + timesteps]
        ans.append(forlistgen)
    return np.array(ans)

y=y_data[timesteps:]
x=split_x(x_data,timesteps)
print(x)
print(y)
# def time_splitx(x,ts,scaler):
#     x=scaler.transform(x)
#     gen=(x[i:i+ts] for i in range(len(x)-ts+1))
#     return np.array(list(gen))[:,:-1] #: 마지막데이터를안쓰겟다

# def time_splity(y,ts):
#     gen=(y[i:i+ts] for i in range(len(y)-ts+1))
#     return np.array(list(gen))[:,-1]



# bbb = split_x(x_data , timesteps)
# print(bbb)
# print(bbb.shape) #6,5 

# y_data = y_data[timesteps:]
# print(y_data)
# [[[ 1  2  3]
#   [ 5  6  7]
#   [ 9 10 11]
#   [13 14 15]
#   [17 18 19]
#   [21 22 23]
#   [25 26 27]
#   [29 30 31]
print(x.shape)

#타임스텝 9 대가로순대로 계산
# [[[ 1  2  3]
#   [ 5  6  7]
#   [ 9 10 11]
#   [13 14 15]
#   [17 18 19]
#   [21 22 23]
#   [25 26 27]
#   [29 30 31]
#   [33 34 35]]]
# [40]
# (1, 9, 3) 
#타임스텝 예를들어 10개 데이터중 9개만 쓴다 왜냐하면 과거의데이터만 사용해야 미래예측이가능하기때문\
#월요일날 시가 공가 종가, 값을 맞춰라 월요일 데이터를 받으면 수요일 데이터를 예측해야된다 5일치,20일 치를 짤라서 맞춰라,
# #건너뛰어서 바로 다음 
# 데이터 불러와서 정제
# 가중치 정제
