import tensorflow.keras as keras
from tensorflow.keras import backend
import tensorflow as tf
import numpy as np
import time
import os
import random
import sklearn.metrics as metrics
import datetime
from numpy import genfromtxt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
# from keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras.optimizers import Adam 



class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, plot_test_acc=False):

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()


        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size,validation_data=(x_val, y_val), epochs=250)
        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        keras.backend.clear_session()

        return True

InceptionModel = Classifier_INCEPTION("",(50,8),6)
no_points = 50
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
labels = tf.keras.utils.to_categorical(l, num_classes=8)
for file in classes:
    path = folder_dir + '/' + file
    i = 0
    chosen = []
    k = int(len(os.listdir(path))/10)
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
start_time = time.time()
InceptionModel.fit(train_data,train_label,test_data,test_label,test_label)
test_duration = time.time() - start_time
print(test_duration)
InceptionModel.model.save('testinception.h5f5')
# # predict(self, x_test,y_true,x_train,y_train,y_test,return_df_metrics = True):
# # RSNET.predict(test_data,test_label,train_data,train_label,test_label)
# model.evaluate(test_data, test_label, verbose=0)
# cvscores=[]

# scores = InceptionModel.model.evaluate(test_data, test_label, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# predictions = InceptionModel.predict(test_data)
# matrix = metrics.confusion_matrix(test_label.argmax(axis=1), predictions.argmax(axis=1))
# print(matrix)
# # model.save('inception50.h5')