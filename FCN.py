import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import sklearn.metrics as metrics
import os
import random
import datetime
from numpy import genfromtxt
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True,build=True):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

        return

    def build_model(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60: 
            padding = 'same'

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model_FCN.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        mini_batch_size = 16
        nb_epochs = 250

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val))

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model_FCN.hdf5')

        keras.backend.clear_session()

    def predict(self, x_test,y_true,x_train,y_train,y_test,return_df_metrics = True):
        model_path = self.output_directory + 'best_model_FCN.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
 


fcn = Classifier_CNN("",(200,8),6)
no_points = 200
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
#         print(l)
labels = tf.keras.utils.to_categorical(l, num_classes=6)

for file in classes:
    path = folder_dir + '/' + file
    i = 0
    chosen = []
    k = int(len(os.listdir(path))*0.7)
    print(k)
    for j in range(0,k):
        csv = random.choice(os.listdir(path))
        while csv in chosen:
            csv = random.choice(os.listdir(path))
        chosen.append(csv)
        print(folder_dir + '/' + file + '/' + csv)
        try:
            my_data = genfromtxt(folder_dir + '/' + file + '/' + csv, delimiter=',')
            for j in range(0, int(len(my_data) / no_points), no_points):
                A = my_data[j:j + no_points, :]
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

cvscores=[]
fcn.fit(train_data,train_label,test_data,test_label,test_label)
fcn.model.save('fcn.h5')

# fcn.model.evaluate(test_data, test_label, verbose=0)
# scores = fcn.model.evaluate(test_data, test_label, verbose=0)
# print("%s: %.2f%%" % (fcn.model.metrics_names[1], scores[1]*100))
# cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# predictions = fcn.model.predict(test_data)
# matrix = metrics.confusion_matrix(test_label.argmax(axis=1), predictions.argmax(axis=1))
# print(matrix)
# fcn.model.save('fcn.h5')