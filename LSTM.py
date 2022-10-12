import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, SimpleRNN, LSTM
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import random
import datetime
from numpy import genfromtxt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

saved_model = 0
no_points = 50

model = Sequential()

# LSTM layers
model.add(LSTM(128, input_shape=(50,6),dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(64, return_sequences=True))  
model.add(LSTM(32))
model.add(Dense(16, activation='tanh'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


if saved_model == 0:
    # Train model
    # Note that validation data is NOT used to train model at all
    train_data = np.empty((0, no_points, 8), float)
    test_data = np.empty((0, no_points, 8), float)
    train_label = []
    test_label = []
    folder_dir = 'full'
    cols = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "class"]
    classes = []
    for files in os.listdir(folder_dir):
        classes.append(files)

    code = np.array(classes)
    label_encoder = LabelEncoder()
    l = label_encoder.fit_transform(code)
    labels = tf.keras.utils.to_categorical(l, num_classes=6)

    for file in classes:
        path = folder_dir + '/' + file
        i = 0
        chosen = []
        if  "overhang" not in file:
            for j in range(0,49):
                csv = random.choice(os.listdir(path))
                while csv in chosen:
                    csv = random.choice(os.listdir(path))
                chosen.append(csv)
                try:
                    print(folder_dir + '/' + file + '/' + csv)
                    my_data = genfromtxt(folder_dir + '/' + file + '/' + csv, delimiter=',')
                    for j in range(0, int(len(my_data) / no_points), no_points):
                        A = my_data[j:j + no_points, :]
                        A = A[:, 1:-1]
                        if i % 4 != 0:
                            train_data = np.append(train_data, A.reshape(1, A.shape[0], A.shape[1]), axis=0)
                            train_label.append(file)

                        else:
                            test_data = np.append(test_data, A.reshape(1, A.shape[0], A.shape[1]), axis=0)
                            test_label.append(file)
                        i = i + 1
                except :
                    print("Oops!  file Error...")


    code = np.array(train_label)
    label_encoder = LabelEncoder()
    l = label_encoder.fit_transform(code)
    train_label = tf.keras.utils.to_categorical(l, num_classes=6)
    code = np.array(test_label)
    label_encoder = LabelEncoder()
    l = label_encoder.fit_transform(code)
    test_label = tf.keras.utils.to_categorical(l, num_classes=6)

    history = model.fit(train_data, train_label, validation_data=(test_data, test_label),shuffle=True, epochs=500, batch_size=100)
    model.save('LSTMtanh.h5')

    