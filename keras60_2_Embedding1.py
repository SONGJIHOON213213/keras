from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input,Reshape,Embedding

#1.데이터
docs = ['너무 재밌어요','참 최고에요','참 잘 만든 영화에요','추천하고 싶은 영화입니다','한 번 더 보고 싶네요',
        '글세요','별로에요','생각보다 지루해요','연기가 어색해요','재미없어요','너무 재미없다','참 재밌네요','환희가 잘 생기긴 했어요','환희가 안해요']

#긍정 1 , 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

token = Tokenizer()
token.fit_on_texts(docs)#14개
print(token.word_index)
# {'참': 1, '너무': 2, '잘': 3, '환희가': 4, '재밌어요': 5, '최고에요': 6, '만든': 7, '영화에요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 
# 12, '번': 13, '더': 14, '보고': 15, '싶네요': 16, '글세요': 17, '별로에요': 18, '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없 
# 어요': 23, '재미없다': 24, '재밌네요': 25, '생기긴': 26, '했어요': 27, '안해요': 28}

x = token.texts_to_sequences(docs)
print(x)



from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x,padding='pre', maxlen =5 ) #앞에서 부터 0을 채운다 pre하면 pre 반대는post
#[[2, 5], [1, 6], [1, 3, 7, 8], [9, 10, 11], [12, 13, 14, 15, 16], [17], [18], [19, 20], [21, 22], [23], [2, 24], [1, 25], [4, 3, 26, 27], [4, 28]]  
# 0 0 25  0016 
print(pad_x)
print(pad_x.shape) #14,5

pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)
word_szie = len(token.word_index)
print("단어사전의 갯수:", word_szie)


 #단어사전의 갯수: 28

#2.모델
model = Sequential()
# model.add(Embedding(10,33))
model.add(Embedding(input_dim=28,output_dim=10))#좌표계의연산되는량이28개#좌표계의연산되는량이28개
model.add(LSTM(32))
model.add(Dense(20))#(20640, 8) (20640,)
model.add(Dense(20, activation ='relu'))
model.add(Dense(20, activation ='relu'))
model.add(Dense(1, activation='sigmoid')) 

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(pad_x,labels, epochs=100,batch_size=30)

acc =model.evaluate(pad_x,labels)[1]
print('acc:',acc)