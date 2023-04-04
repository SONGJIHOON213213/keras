from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Conv2D, SimpleRNN, Concatenate, concatenate, Dropout, Bidirectional, Flatten, MaxPooling2D, Input,Reshape,Embedding

docs = ['너무 재밌어요','참 최고에요','참 잘 만든 영화에요','추천하고 싶은 영화입니다','한 번 더 보고 싶네요',
        '글세요','별로에요','생각보다 지루해요','연기가 어색해요','재미없어요','너무 재미없다','참 재밌네요','환희가 잘 생기긴 했어요','환희가 안해요','나는 성호가 정말 재미없다 너무 정말']



labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])

token = Tokenizer()
token.fit_on_texts(docs)#14개
print(token.word_index)


x = token.texts_to_sequences(docs)
print(x)


from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x,padding='pre', maxlen =5 ) #앞에서 부터 0을 채운다 pre하면 pre 반대는post

print(pad_x)
print(pad_x.shape) #14,5

pad_x = pad_x.reshape(pad_x.shape[0],pad_x.shape[1],1)
word_szie = len(token.word_index)
print("단어사전의 갯수:", word_szie)

model = Sequential()
model.add(Embedding(28,10,input_length=5))
model.add(LSTM(32))
model.add(Dense(20))
model.add(Dense(20, activation ='relu'))
model.add(Dense(20, activation ='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(pad_x,labels, epochs=100,batch_size=30)

new_doc = ["나는 성호가 정말 재미없다 너무 정말"]
new_x = token.texts_to_sequences(new_doc)
new_pad_x = pad_sequences(new_x, padding='pre', maxlen=5)
print("단어사전의 갯수:", word_szie)
# acc =model.evaluate(pad_x,labels)[1]
# print('acc:',acc) 

# new_pad_x = new_pad_x.reshape(new_pad_x.shape[0], new_pad_x.shape[1], 1)

# pred = model.predict(new_pad_x)
# if pred > 0.5:
#     print("긍정이면",0)
# else:
#     print("부정",1)


