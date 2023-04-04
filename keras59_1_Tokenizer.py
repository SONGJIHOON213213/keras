from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

text = '나는 진짜 매우 매우 맜있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index) 
 #가장 많은놈이 앞으로 갔다.  {'마구': 1, '매우': 2, '나는': 3, '진짜': 4, '맜있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
print(token.word_counts)
#OrderedDict([('나는', 1), ('진짜', 1), ('매우', 2), ('맜있는', 1), ('밥을', 1), ('음청', 1), ('마구', 3), ('쳐묵었다', 1)]) 단어횟수를 표시



#####2.get_dummmies 1차원으로 받아야됨 
#####2차원으로 바꿔야댐 라벨은 1차원만 가능
# x = pd.get_dummies(np.array(x).reshape(11,))
# x = pd.get_dummies(np.array(x).reavel())
# x_numpy = x.values
######3.원핫########### 2차원으로 받아야됨
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
x = token.texts_to_sequences([text])
x = np.array(x) # convert the list to a numpy array 
x_encoded = ohe.fit_transform(np.array(x).reshape(-1,1)).toarray()
print(x.shape)
print(x_encoded)