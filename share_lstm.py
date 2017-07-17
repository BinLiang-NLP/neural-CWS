#!/bin/env python
#-*- coding:utf8 -*-

__author__ = 'Akuchi <liangbin05@baidu.com>'
__date__ = '2017-06-14'


from keras.layers import Embedding, LSTM, TimeDistributed, \
        Input, Bidirectional, merge, concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import random
import time

from share_dataset import generate_train_data # tag_window, embed_dim, nb_class

def build_model(data1, data2, word_weights, tag_window=5, embed_dim=100):
    batch_size = 2048
    nb_epoch = 16
    nb_class = 4
    hidden_dim = 128

    train_x1 = np.array(list(data1['x'])[:2871133])
    train_y1 = np.array(list(data1['y'])[:2871133])
    train_y1 = np_utils.to_categorical(train_y1, nb_class)
    train_x2 = np.array(list(data2['x'])[:2871133])
    train_y2 = np.array(list(data2['y'])[:2871133])
    train_y2 = np_utils.to_categorical(train_y2, nb_class)

    print(train_x1.shape)
    print(train_y1.shape)
    print(train_x2.shape)
    print(train_y2.shape)
 
    input_x1 = Input(shape=(tag_window, ), dtype='float32', name='input_x1')
    input_x2 = Input(shape=(tag_window, ), dtype='float32', name='input_x2')
    # share_input_x = Input(shape=(tag_window, ), dtype='float32', name='input_x')

    embed_x1 = Embedding(output_dim=embed_dim, 
            input_dim=word_weights.shape[0],
            input_length=tag_window,
            weights=[word_weights],
            name='embed_x1')(input_x1)
    embed_x2 = Embedding(output_dim=embed_dim, 
            input_dim=word_weights.shape[0],
            input_length=tag_window,
            weights=[word_weights],
            name='embed_x2')(input_x2)
    share_embed_x = merge([embed_x1, embed_x2],
            mode='concat', concat_axis=2)

    share_bi_lstm = Bidirectional(LSTM(hidden_dim, return_sequences=False), merge_mode='sum')(share_embed_x)
    bi_lstm1 = Bidirectional(LSTM(hidden_dim, return_sequences=False), merge_mode='sum')(embed_x1)
    bi_lstm2 = Bidirectional(LSTM(hidden_dim, return_sequences=False), merge_mode='sum')(embed_x2)

    share_x_dropout = Dropout(0.5)(share_bi_lstm)
    x_dropout1 = Dropout(0.5)(bi_lstm1)
    x_dropout2 = Dropout(0.5)(bi_lstm2)

    combine_dropout1 = concatenate([x_dropout1, share_x_dropout])
    x_output1 = Dense(nb_class,
        # kernel_regularizer=regularizers.l2(0.01),
        # kernel_constraint=maxnorm(3.0),
        # activity_regularizer=regularizers.l2(0.01),
        activation='softmax',
        name='x_output1')(combine_dropout1)
    
    combine_dropout2 = concatenate([x_dropout2, share_x_dropout])
    x_output2 = Dense(nb_class,
        # kernel_regularizer=regularizers.l2(0.01),
        # kernel_constraint=maxnorm(3.0),
        # activity_regularizer=regularizers.l2(0.01),
        activation='softmax',
        name='x_output2')(combine_dropout2)

  
    model = Model(inputs=[input_x1, input_x2], outputs=[x_output1, x_output2])
    model.compile(optimizer='adamax', loss='categorical_crossentropy',metrics=['accuracy'])
    print('Train...')
    model_path = './model/share_model_128hidden_2048batch.hdf5'
    modelcheckpoint = ModelCheckpoint(model_path, monitor='val_cc', verbose=1, save_best_only=True)
    model.fit([train_x1, train_x2], [train_y1, train_y2], validation_split=0.2, 
            batch_size=batch_size, epochs=nb_epoch, shuffle=True, callbacks=[modelcheckpoint])


if __name__ == '__main__':
    t0 = time.time()
    tag_window = 5
    embed_dim = 300
    nb_class = 4
    data1, data2, word_weights = generate_train_data(tag_window=tag_window, embed_dim=embed_dim, nb_class=nb_class)
    t1 = time.time()
    print('get data costs:', t1-t0)
    build_model(data1, data2, word_weights, tag_window=tag_window, embed_dim=embed_dim)
