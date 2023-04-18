import numpy as np
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
x = [1,2,3]
y = [4,5,6]

x_train,x_test,y_train,y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2, stratify=y)

for i in a: 
    for j in b:
        model = SVC(a=i,b=j)
        model.fit(x_train,y_train)
        score = model.score(x_test,y_test)
        
        if max_score < score:
            max_score = score
            best_parameters = ('x': i,'y',j) 
