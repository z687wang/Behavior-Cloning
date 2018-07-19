import os

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

data = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)

train_data, validation_data = train_test_split(data, test_size=0.2)

def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    '''
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    # crop to 105x320x3
    #new_img = img[35:140,:,:]
    # crop to 40x320x3
    new_img = img[50:140,:,:]
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    # scale to ?x?x3
    #new_img = cv2.resize(new_img,(80, 10), interpolation = cv2.INTER_AREA)
    # convert to YUV color space (as nVidia paper suggests)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

def generator(data, batch_size=32):
    num_data = len(data)
    images_location = ''
    correction = 0.25
    while True:
        for offset in range(0, num_data, batch_size):
            batch_samples = data[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image_file = images_location + batch_sample[0]
                center_image = cv2.imread(center_image_file)
                center_angle = float(batch_sample[3])
                left_image_file = images_location + batch_sample[1]
                left_image = cv2.imread(left_image_file)
                left_angle = center_angle - correction
                right_image_file = images_location + batch_sample[2]
                right_image = cv2.imread(right_image_file)
                right_angle = center_angle + correction
                center_image = preprocess_image(center_image)
                left_image = preprocess_image(left_image)
                right_image = preprocess_image(right_image)
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_data, batch_size=32)
validation_generator = generator(validation_data, batch_size=32)


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3)))
# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# model.add(Dropout(0.50))

# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Add a flatten layer
model.add(Flatten())

# Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
# model.add(Dropout(0.50))

# Add a fully connected output layer
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

model.fit_generator(train_generator, samples_per_epoch=len(train_data),
                    validation_data=validation_generator, nb_val_samples=len(validation_data), nb_epoch=5, verbose=2, callbacks=[checkpoint])

print(model.summary())

model.save_weights('./model.h5')
json_string = model.to_json()
with open('./model.json', 'w') as f:
    f.write(json_string)





