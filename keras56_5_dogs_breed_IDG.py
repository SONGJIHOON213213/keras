from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
path_save ='d:/study_data/_save/breed/'
datagen = ImageDataGenerator(rescale=1./255)

breed = datagen.flow_from_directory('d:/study_data/_data/dog_breed/dog_v1/', target_size=(1000, 1000), batch_size=500, class_mode='categorical', shuffle=True, color_mode='rgb')

np.save(path_save + 'breed_x.npy', arr=breed[0][0])
np.save(path_save + 'breed_y.npy', arr=breed[0][1])