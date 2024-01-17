import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import accuracy_score,classification_report
from keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Activation, Dropout, Flatten, Dense, BatchNormalization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm

# Train and Test Directory
train_dir = 'data\Train'
test_dir = 'data\Test'

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(train_dir, test_dir, classes, num_epochs, model_type='model_1'):

    # Creating Batches
    training_dataset = image_dataset_from_directory(directory=train_dir,
                                                    image_size=(32, 32),
                                                    batch_size=32,
                                                    label_mode='categorical')

    testing_dataset = image_dataset_from_directory(directory=test_dir,
                                                   image_size=(32, 32),
                                                   batch_size=32,
                                                   label_mode='categorical',
                                                   shuffle=False)

    # Choose model architecture
    if model_type == 'model_1':
        model = create_model_1(classes)
    elif model_type == 'model_2':
        model = create_model_2(classes)
    else:
        raise ValueError("Invalid model type. Choose 'model_1' or 'model_2'.")

    # Compile Model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Model fitting on the loaded dataset
    model_history = model.fit(training_dataset,
                              validation_data=testing_dataset,
                              epochs=num_epochs,
                              callbacks=[ModelCheckpoint('best_model.h5', save_best_only=True)])
    
    return model, model_history


def create_model_1(classes):
    model = tf.keras.Sequential([
        tf.keras.Sequential([layers.Rescaling(1. / 255), ]),
        layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)),
        layers.AveragePooling2D(),
        layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(len(classes), activation='softmax')
    ])
    return model


def create_model_2(classes):
    model = Sequential([layers.Rescaling(1. / 255), ])
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=1, activation="relu", input_shape=(32, 32, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=1, activation="relu", input_shape=(32, 32, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=1, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_initializer="uniform"))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu", kernel_initializer="uniform"))
    model.add(BatchNormalization())
    model.add(Dense(46, activation="softmax", kernel_initializer="uniform"))
    return model


# Example Usage:
train_dir = 'data\Train'
test_dir = 'data\Test'
classes = os.listdir(train_dir)
num_epochs = 10


# Training Model 1
model_1, model_1_history = train_model(train_dir, test_dir, classes, num_epochs, model_type='model_1')

# Training Model 2
model_2, model_2_history = train_model(train_dir, test_dir, classes, num_epochs, model_type='model_2')