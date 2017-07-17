#!/bin/env python
#-*- coding:utf8 -*-

__author__ = 'Akuchi <liangbin05@baidu.com>'
__date__ = '2017-06-14'


from keras.layers import Embedding, LSTM, TimeDistributed, Input, \
        Bidirectional, Conv1D, MaxPooling1D, merge, concatenate, Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Model
from keras import regularizers
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

from dataset import generate_train_data # tag_window, embed_dim, nb_class


def build_cnn(kernel_size, nb_filters, embed_x):
    """

    """
    name = 'cnn_' + str(kernel_size)
    cnn_x = Conv1D(filters=nb_filters,
            kernel_size=kernel_size,
            padding='valid',
            activation='tanh')(embed_x)
            # kernel_regularizer=regularizers.l2(0.01),
            # kernel_constraint=maxnorm(3.0),
            # activity_regularizer=regularizers.l2(0.01))(embed_x)
    maxPooling_x = MaxPooling1D(kernel_size)(cnn_x)
    return maxPooling_x

def build_model(data, word_weights, tag_window=5, embed_dim=100):
    batch_size = 50
    nb_epoch = 8
    nb_class = 4
    hidden_dim = 128
    nb_filters = 100

    train_x = np.array(list(data['x']))
    train_y = np.array(list(data['y']))
    train_y = np_utils.to_categorical(train_y, nb_class)

    print(train_x.shape)
    print(train_y.shape)
    input_x = Input(shape=(tag_window, ), dtype='float32', name='input_x')
    embed_x = Embedding(output_dim=embed_dim, 
            input_dim=word_weights.shape[0],
            input_length=tag_window,
            weights=[word_weights],
            name='embed_x')(input_x)
    # bi_lstm = Bidirectional(LSTM(hidden_dim, return_sequences=False), merge_mode='sum')(embed_x)

    maxPooling_2 = build_cnn(2, nb_filters, embed_x)
    print('finish 2')
    maxPooling_3 = build_cnn(3, nb_filters, embed_x)
    print('finish 3')
    maxPooling_4 = build_cnn(4, nb_filters, embed_x)
    print('finish 4')
    maxPooling_5 = build_cnn(5, nb_filters, embed_x)
    maxPooling = concatenate([maxPooling_2, maxPooling_3, maxPooling_4, maxPooling_5], axis=1)

    x_dropout = Dropout(0.5)(maxPooling_2)
    x_flatten = Flatten()(x_dropout)
    x_output = Dense(nb_class,
        # kernel_regularizer=regularizers.l2(0.01),
        # kernel_constraint=maxnorm(3.0),
        # activity_regularizer=regularizers.l2(0.01),
        activation='softmax')(x_flatten)
    model = Model(input=[input_x], output=[x_output])
    model.compile(optimizer='adamax', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit([train_x], [train_y], validation_split=0.2, 
            batch_size=batch_size, epochs=nb_epoch, shuffle=True)


if __name__ == '__main__':
    tag_window = 5
    embed_dim = 100
    nb_class = 4
    data, word_weights = generate_train_data(tag_window=tag_window, embed_dim=embed_dim, nb_class=nb_class)
    build_model(data, word_weights, tag_window=tag_window, embed_dim=embed_dim)
