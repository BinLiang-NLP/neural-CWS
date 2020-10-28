#!/bin/env python
#-*- coding:utf8 -*-

__author__ = 'BinLiang <bin.liang@stu.hit.edu.cn>'
__date__ = '2017-06-14'


from keras.layers import Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Model
from keras import regularizers
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np

from dataset import generate_train_data # tag_window, embed_dim, nb_class


def build_model(data, word_weights, tag_window=5, embed_dim=100):
    batch_size = 32
    nb_epoch = 16
    nb_class = 4
    hidden_dim = 128

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
    bi_lstm = Bidirectional(LSTM(hidden_dim, return_sequences=False), merge_mode='sum')(embed_x)
    x_dropout = Dropout(0.5)(bi_lstm)
    x_output = Dense(nb_class,
        # kernel_regularizer=regularizers.l2(0.01),
        # kernel_constraint=maxnorm(3.0),
        # activity_regularizer=regularizers.l2(0.01),
        activation='softmax')(x_dropout)
    model = Model(input=[input_x], output=[x_output])
    model.compile(optimizer='adamax', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit([train_x], [train_y], validation_split=0.2, 
            batch_size=batch_size, epochs=nb_epoch, shuffle=True)


if __name__ == '__main__':
    tag_window = 7
    embed_dim = 100
    nb_class = 4
    data, word_weights = generate_train_data(tag_window=tag_window, embed_dim=embed_dim, nb_class=nb_class)
    build_model(data, word_weights, tag_window=tag_window, embed_dim=embed_dim)
