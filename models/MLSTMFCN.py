from __future__ import absolute_import
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
# from keras import layers
from tensorflow.keras.layers import InputSpec
# from keras.legacy import interfaces
from keras.layers import Recurrent

from keras.layers import Layer
from keras.layers import RepeatVector
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape, Dropout, Permute,Conv1D
from keras.layers import BatchNormalization,GlobalAveragePooling1D
from keras import Model





def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):

    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.int_shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class AttentionLSTM(Recurrent):

    # @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 attention_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 attention_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 attention_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_attention=False,
                 implementation=1,
                 **kwargs):
        super(AttentionLSTM, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.attention_activation = activations.get(attention_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.attention_initializer = initializers.get(attention_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.attention_regularizer = regularizers.get(attention_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attention_constraint = constraints.get(attention_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.return_attention = return_attention
        self.state_spec = [InputSpec(shape=(None, self.units)),
                           InputSpec(shape=(None, self.units))]
        self.implementation = implementation

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.timestep_dim = input_shape[1]
        self.input_dim = input_shape[2]
        # print("asdasdasd")
        # type(self.input_spec)
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        # add attention kernel
        self.attention_kernel = self.add_weight(
            shape=(self.input_dim, self.units * 4),
            name='attention_kernel',
            initializer=self.attention_initializer,
            regularizer=self.attention_regularizer,
            constraint=self.attention_constraint)

        # add attention weights
        # weights for attention model
        self.attention_weights = self.add_weight(shape=(self.input_dim, self.units),
                                                 name='attention_W',
                                                 initializer=self.attention_initializer,
                                                 regularizer=self.attention_regularizer,
                                                 constraint=self.attention_constraint)

        self.attention_recurrent_weights = self.add_weight(shape=(self.units, self.units),
                                                           name='attention_U',
                                                           initializer=self.recurrent_initializer,
                                                           regularizer=self.recurrent_regularizer,
                                                           constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

            self.attention_bias = self.add_weight(shape=(self.units,),
                                                  name='attention_b',
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint)

            self.attention_recurrent_bias = self.add_weight(shape=(self.units, 1),
                                                            name='attention_v',
                                                            initializer=self.bias_initializer,
                                                            regularizer=self.bias_regularizer,
                                                            constraint=self.bias_constraint)
        else:
            self.bias = None
            self.attention_bias = None
            self.attention_recurrent_bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        self.attention_i = self.attention_kernel[:, :self.units]
        self.attention_f = self.attention_kernel[:, self.units: self.units * 2]
        self.attention_c = self.attention_kernel[:, self.units * 2: self.units * 3]
        self.attention_o = self.attention_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None

        self.built = True

    def preprocess_input(self, inputs, training=None):
        return inputs

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation != 0 and 0 < self.dropout < 1:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # append the input as well for use later
        constants.append(inputs)
        return constants

    def step(self, inputs, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]
        x_input = states[4]

        # alignment model
        h_att = K.repeat(h_tm1, self.timestep_dim)
        att = _time_distributed_dense(x_input, self.attention_weights, self.attention_bias,
                                      output_dim=K.int_shape(self.attention_weights)[1])
        attention_ = self.attention_activation(K.dot(h_att, self.attention_recurrent_weights) + att)
        attention_ = K.squeeze(K.dot(attention_, self.attention_recurrent_bias), 2)

        alpha = K.exp(attention_)

        if dp_mask is not None:
            alpha *= dp_mask[0]

        alpha /= K.sum(alpha, axis=1, keepdims=True)
        alpha_r = K.repeat(alpha, self.input_dim)
        alpha_r = K.permute_dimensions(alpha_r, (0, 2, 1))

        # make context vector (soft attention after Bahdanau et al.)
        z_hat = x_input * alpha_r
        context_sequence = z_hat
        z_hat = K.sum(z_hat, axis=1)

        if self.implementation == 2:
            z = K.dot(inputs * dp_mask[0], self.kernel)
            z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
            z += K.dot(z_hat, self.attention_kernel)

            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)
        else:
            if self.implementation == 0:
                x_i = inputs[:, :self.units]
                x_f = inputs[:, self.units: 2 * self.units]
                x_c = inputs[:, 2 * self.units: 3 * self.units]
                x_o = inputs[:, 3 * self.units:]
            elif self.implementation == 1:
                x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
                x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
                x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
                x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o
            else:
                raise ValueError('Unknown `implementation` mode.')

            i = self.recurrent_activation(x_i + K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_i)
                                              + K.dot(z_hat, self.attention_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1 * rec_dp_mask[1], self.recurrent_kernel_f)
                                          + K.dot(z_hat, self.attention_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * rec_dp_mask[2], self.recurrent_kernel_c)
                                                + K.dot(z_hat, self.attention_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1 * rec_dp_mask[3], self.recurrent_kernel_o)
                                          + K.dot(z_hat, self.attention_o))
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True

        if self.return_attention:
            return context_sequence, [h, c]
        else:
            return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'attention_activation': activations.serialize(self.attention_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'attention_initializer': initializers.serialize(self.attention_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'attention_regularizer': regularizers.serialize(self.attention_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'attention_constraint': constraints.serialize(self.attention_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'return_attention': self.return_attention}
        base_config = super(AttentionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

DATASET_INDEX = 11

# MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
# MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
# NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True
def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se
from keras.layers import RepeatVector
def generate_model():
    ip = Input(shape=(400, 8))

    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y=squeeze_excite_block(y)
    # y = GlobalAveragePooling1D()(y)
    # y = Reshape((1, 128))(y)
    # y = Dense(128)(y)
    # y = multiply([input, y])
    #y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(6, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()
    return model


def generate_model_2():
    ip = Input(shape=(400, 8))

    x = Masking()(ip)
    x = LSTM(100)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    #y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(6, activation='softmax')(x)

    model = Model(ip, out)
    model.summary()
    return model


if __name__ == "__main__":
    model = generate_model()
    
    
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
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape, normalization
    from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

    no_points = 400
    train_data = np.empty((0, no_points, 8), float)
    test_data = np.empty((0, no_points, 8), float)
    train_label = []
    test_label = []
    folder_dir = '../full'
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
        k = int(len(os.listdir(path))/5)
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=1500, batch_size=1000)
    test_duration = time.time() - start_time
    print(test_duration)
    model.save('MLSTM.h5')


