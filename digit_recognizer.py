import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')

def get_data(im_shape):    
    
    train_set = pd.read_csv("../input/train.csv")
    test_set = pd.read_csv("../input/test.csv")

    rand_per = np.random.permutation(train_set.shape[0])

    train = train_set.iloc[rand_per[0:int(train_set.shape[0]*0.95)], :]
    dev = train_set.iloc[rand_per[int(train_set.shape[0]*0.95):], :]

    train_x = train.iloc[:, 1:]
    train_y = train.iloc[:, 0]
    dev_x = dev.iloc[:, 1:]
    dev_y = dev.iloc[:, 0]
    
    train_x = train_x.values
    train_y = train_y.values
    dev_x = dev_x.values
    dev_y = dev_y.values
    test_set = test_set.values

    train_x = train_x/255.
    dev_x  = dev_x/255.
    test_set = test_set/255.
    
    train_y = to_categorical(train_y, num_classes = 10)
    dev_y = to_categorical(dev_y, num_classes = 10)
    
    train_x = np.reshape(train_x, (train_x.shape[0], im_shape[0], im_shape[1], 1))
    dev_x = np.reshape(dev_x, (dev_x.shape[0], im_shape[0], im_shape[1], 1))
    test_set = np.reshape(test_set, (test_set.shape[0], im_shape[0], im_shape[1], 1))

    return train_x, train_y, dev_x, dev_y, test_set
    
def digit_model(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(10, (5, 5), strides = (1, 1), name = 'conv1', padding='same')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
               
    X = Conv2D(28, (5, 5), strides = (1, 1), name = 'conv2')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)
    
    X = Dropout(0.25)(X)
    
    X = Conv2D(36, (5, 5), strides = (1, 1), name = 'conv3', padding='same')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(52, (3, 3), strides = (1, 1), name = 'conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    
    X = Dropout(0.25)(X)
    
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv5')(X)
    X = BatchNormalization(axis = 3, name = 'bn5')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)
    
    X = Dropout(0.6)(X)
    
    X = Flatten()(X)
    X = Dense(100, activation='relu', name='fc1')(X)
    X = Dense(10, activation='softmax', name='fc2')(X)
    
    model = Model(inputs = X_input, outputs = X, name='MNIST_model')
    
    return model

def digit_classifier():
    
    im_shape = (28, 28, 1)

    train_x, train_y, dev_x, dev_y, test_set = get_data(im_shape)

    model = digit_model(im_shape)

    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(x = train_x, y = train_y, epochs = 100, batch_size = 32, validation_data = (dev_x, dev_y))

    imgIDs = np.arange(test_set.shape[0]) + 1
    pred = model.predict(test_set)
    results = np.argmax(pred, axis=1)

    submission = pd.DataFrame({
        "ImageId": imgIDs,
        "Label": results})
    submission.to_csv('submission.csv', index = False)
    
digit_classifier()
