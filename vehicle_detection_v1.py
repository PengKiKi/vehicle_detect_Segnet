from helper import *

import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D, Dense
from keras.layers import Deconvolution2D, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from helper_normal import *


# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 32
epochs = 20
pool_size = (2, 2)
input_shape = np.array(x_train_data).shape[1:]
input_shape=(64, 64, 1)
len(x_val_data)
print(input_shape)

### Here is the actual neural network ###
model = Sequential()

# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv Layer 1
model.add(Convolution2D(60, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv1'))

# Conv Layer 2
model.add(Convolution2D(50, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv2'))

# Pooling 1
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 3
model.add(Convolution2D(40, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv3'))
model.add(Dropout(0.2))

# Conv Layer 4
model.add(Convolution2D(30, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))

# Conv Layer 5
model.add(Convolution2D(20, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2))

# Pooling 2
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 6
model.add(Convolution2D(10, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv6'))
model.add(Dropout(0.2))

# Conv Layer 7
model.add(Convolution2D(5, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv7'))
model.add(Dropout(0.2))

# Pooling 3
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(1, 4, 4, border_mode='valid', subsample=(1,1), activation = 'relu', name = 'Conv8'))
model.add(Dropout(0.2))


'''
# Upsample 1
model.add(UpSampling2D(size=pool_size))

# Deconv 1
model.add(Deconvolution2D(10, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu',
                          output_shape = model.layers[8].output_shape, name = 'Deconv1'))
model.add(Dropout(0.2))

# Deconv 2
model.add(Deconvolution2D(20, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu',
                          output_shape = model.layers[7].output_shape, name = 'Deconv2'))
model.add(Dropout(0.2))

# Upsample 2
model.add(UpSampling2D(size=pool_size))

# Deconv 3
model.add(Deconvolution2D(30, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu',
                          output_shape = model.layers[5].output_shape, name = 'Deconv3'))
model.add(Dropout(0.2))

# Deconv 4
model.add(Deconvolution2D(40, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu',
                          output_shape = model.layers[4].output_shape, name = 'Deconv4'))
model.add(Dropout(0.2))

# Deconv 5
model.add(Deconvolution2D(50, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu',
                          output_shape = model.layers[3].output_shape, name = 'Deconv5'))
model.add(Dropout(0.2))

# Upsample 3
model.add(UpSampling2D(size=pool_size))

# Deconv 6
model.add(Deconvolution2D(60, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu',
                          output_shape = model.layers[1].output_shape, name = 'Deconv6'))

# Final layer - only including one channel so 1 filter
model.add(Deconvolution2D(1, 3, 3, border_mode='valid', subsample=(1,1), activation = 'relu',
                          output_shape = model.layers[0].output_shape, name = 'Final'))
'''

model.summary()
### End of network ###

val_step=int(len(x_val_data)/32)
samples_per_epoch_train = int(len(x_train_data)/1)

# Compiling and training the model
model.compile(optimizer='Adam', loss='mean_squared_error')

model.fit_generator(generator(x_train_data, y_train_data, batch_size=batch_size),
                    samples_per_epoch = samples_per_epoch_train,
                    nb_epoch=epochs,
                    verbose=1,
                    validation_data=generator(x_val_data, y_val_data, batch_size=batch_size),
                    validation_steps=val_step)

# Save model architecture and weights
model_json = model.to_json()
with open("full_CNN_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('full_CNN_model.h5')

# Show summary of model
model.summary()

