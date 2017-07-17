'''
加入location特征的LSTM网络
'''
#!/bin/env python
#-*- coding:utf8 -*-

__author__ = 'Akuchi <liangbin05@baidu.com>'
__date__ = '2017-06-14'


from keras.layers import Embedding, LSTM, TimeDistributed, \
        Input, Bidirectional, merge
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Model
from keras import regularizers
from keras.constraints import maxnorm
from keras.utils import np_utils
import numpy as np
import time

from with_location_dataset import generate_train_data # tag_window, embed_dim, nb_class


def build_model(data, word_weights, max_len, tag_window=5, embed_dim=100, location_dim=10):
    batch_size = 2048
    nb_epoch = 16
    nb_class = 4
    hidden_dim = 128

    train_x = np.array(list(data['x']))
    train_l = np.array(list(data['l']))
    train_y = np.array(list(data['y']))
    train_y = np_utils.to_categorical(train_y, nb_class)

    print(train_x.shape)
    print(train_l.shape)
    print(train_y.shape)
    input_x = Input(shape=(tag_window, ), dtype='float32', name='input_x')
    input_l = Input(shape=(tag_window, ), dtype='float32', name='input_l')

    embed_x = Embedding(output_dim=embed_dim, 
            input_dim=word_weights.shape[0],
            input_length=tag_window,
            weights=[word_weights],
            name='embed_x')(input_x)
    embed_l = Embedding(output_dim=location_dim, 
            input_dim=max_len,
            input_length=tag_window,
            name='embed_l')(input_l)

    merge_embed = merge([embed_x, embed_l],
            mode='concat', concat_axis=2)
    bi_lstm = Bidirectional(LSTM(hidden_dim, return_sequences=False), merge_mode='sum')(merge_embed)
    x_dropout = Dropout(0.5)(bi_lstm)
    x_output = Dense(nb_class,
        # kernel_regularizer=regularizers.l2(0.01),
        # kernel_constraint=maxnorm(3.0),
        # activity_regularizer=regularizers.l2(0.01),
        activation='softmax')(x_dropout)
    model = Model(inputs=[input_x, input_l], outputs=[x_output])
    model.compile(optimizer='adamax', loss='categorical_crossentropy',metrics=['accuracy'])
    print('Train...')
    model_path = './model/location_128hidden_2048batch'
    modelcheckpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    model.fit([train_x, train_l], [train_y], validation_split=0.2, 
            batch_size=batch_size, epochs=nb_epoch, shuffle=True)

if __name__ == '__main__':
    tag_window = 5
    embed_dim = 300
    nb_class = 4
    t0 = time.time()
    data, word_weights, max_len = generate_train_data(tag_window=tag_window, embed_dim=embed_dim, nb_class=nb_class)
    t1 = time.time()
    print('load data costs:', t1-t0)
    build_model(data, word_weights, max_len, tag_window=tag_window, embed_dim=embed_dim)
