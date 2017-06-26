import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import glob
import pickle
from numpy import zeros, newaxis


data_file = 'ProcessedData.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)

x_train_data=data['X_train']
x_val_data=data['X_val']
x_test_data=data['X_test']
y_train_data=data['y_train']
y_val_data=data['y_val']
y_test_data=data['y_test']


def generator(samples,lables, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples,lables)
        for offset in range(0, num_samples, batch_size):
            images = samples[offset:offset+batch_size]
            lable = lables[offset:offset+batch_size]
            final_image=[]
            final_lable=[]
            for i in range(len(images)):
                final_image.append(images[i][:, :, newaxis])

                #print("index:",[i], "total:", len(images))
                #print("---------------------------here---------------------",lable[i])
                if lable[i]==1:
                    final_lable.append(np.ones((64, 64, 1)))
                else:
                    final_lable.append(np.zeros((64, 64, 1)))

            # trim image to only see section with road
            X_train = np.array(final_image)
            y_train = np.array(final_lable)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(x_train_data, y_train_data, batch_size=32)
validation_generator = generator(x_val_data, y_val_data, batch_size=32)

ch, row, col = 1, 64, 64  # Trimmed image format

'''
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
'''