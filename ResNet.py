import tensorflow.keras as keras
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
# from utils.utils import save_logs
# from utils.utils import calculate_metrics
tf.keras.backend.clear_session
class Classifier_RESNET:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
        return

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64
        if not tf.test.is_gpu_available:
            print('error')
        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_models100.hdf5'



        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        batch_size = 512
        nb_epochs = 250

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()
        print("here")
        hist = self.model.fit(x_train, y_train, batch_size=64, epochs=nb_epochs, validation_data=(x_val, y_val))

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_models100.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        return True

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'last_models100.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            return y_pred
 


RSNET = Classifier_RESNET("",(100,8),6)
no_points = 100
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

RSNET.fit(train_data,train_label,test_data,test_label,test_label)
RSNET.model.save('Resnet100.h5')

# scores = RSNET.model.evaluate(test_data, test_label, verbose=0)
# print("%s: %.2f%%" % (RSNET.model.metrics_names[1], scores[1]*100))
# cvscores.append(scores[1] * 100)

# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
# predictions = RSNET.model.predict(test_data)
# matrix = metrics.confusion_matrix(test_label.argmax(axis=1), predictions.argmax(axis=1))
# print(matrix)
# RSNET.model.save('Resnet100.h5')