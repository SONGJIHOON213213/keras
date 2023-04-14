import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from sklearn.model_selection import StratifiedShuffleSplit

path = 'd:/study_data/_data/train/'
save_path = 'd:/study_data/_save/train/'
datagen = ImageDataGenerator(rescale=1.)

start = time.time()
train = datagen.flow_from_directory(path,
            target_size=(128,128),
            batch_size=11220,
            class_mode='categorical',
            color_mode= 'grayscale',
            shuffle= True)

face_x = train[0][0]
face_y = train[0][1]

print(f'runtime : {time.time()-start}')

# StratifiedShuffleSplit을 사용하여 데이터를 라벨별로 분리합니다.
split = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
for train_index, test_index in split.split(face_x, face_y):
    train_index = train_index.flatten()
    test_index = test_index.flatten()

    x_train = face_x[train_index]
    y_train = face_y[train_index]
    x_test = face_x[test_index]
    y_test = face_y[test_index]

x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape) #(7853, 128, 128, 1)
print(x_test.shape)  #(3367, 128, 128, 1)
print(y_train.shape) #(7853, 2)
print(y_test.shape)  #(3367, 2)

np.save(save_path + 'x_train.npy', arr = x_train)
np.save(save_path + 'x_test.npy', arr = x_test)
np.save(save_path + 'y_train.npy', arr = y_train)
np.save(save_path + 'y_test.npy', arr = y_test)