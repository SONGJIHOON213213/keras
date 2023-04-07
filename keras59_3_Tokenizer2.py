from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
text1 = '나는 진짜 매우 매우 맜있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '나는 지구용사 배환희다. 멋있다. 또 또 애기해부아'

token = Tokenizer()
x = token.fit_on_texts([text1,text2])

x = token.texts_to_sequences([text1,text2])
# y = token.texts_to_sequences([text2])

# token2 = Tokenizer()
# token2.fit_on_texts([text2])

# print(token.word_index) 
#  #가장 많은놈이 앞으로 갔다.  {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맜있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# print(token.word_counts)
# #OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맜있는', 1), ('밥을', 1), ('음청', 1), ('마구', 3), ('쳐묵었다', 1)]) 단어횟수를 표시
# print(token.word_index) 
# #{'또': 1, '나는': 2, '지구용사': 3, '배환희다': 4, '멋있다': 5, '애기해부아': 6}
# print(token.word_counts)
# #OrderedDict([('나는', 1), ('지구용사', 1), ('배환희다', 1), ('멋있다', 1), ('또', 2), ('애기해부아', 1)])
# #######1.카테고리컬##################
# x = x[0] + x[1] #2개짜리 못넣으므로 펴줘야된다.
# x = to_categorical(x)
# # y = to_categorical(y)
# print(x) #18,14 카테고리컬이있기때문에 14이다.
# # print(y)



####2.get_dummmies 1차원으로 받아야됨 
####2차원으로 바꿔야댐 라벨은 1차원만 가능
x = pd.get_dummies(np.array(x).reshape(11,))
x = pd.get_dummies(np.array(x).ravel())
print(x.shape)




######3.원핫########### 2차원으로 받아야됨
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
x = token.texts_to_sequences([text1])
x = np.array(x) # convert the list to a numpy array 
x_encoded = ohe.fit_transform(np.array(x).reshape(-1,1)).toarray()
print(x.shape)
print(x_encoded)

y = token.texts_to_sequences([text2])
y = np.array(x) # convert the list to a numpy array 
y_encoded = ohe.fit_transform(np.array(y).reshape(-1,1)).toarray()
print(y.shape)
print(y_encoded)