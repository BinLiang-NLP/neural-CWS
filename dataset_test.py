#!/bin/env python
#-*- coding:utf8 -*-

__author__ = 'Akuchi <liangbin05@baidu.com>'
__date__ = '2017-06-14'

# from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
# from keras.models import Model
from keras.utils import np_utils
import re
import sys
import numpy as np
import pandas as pd
from gensim.models import word2vec

corpus_tags = {'S':0, 'B':1, 'M':2, 'E':3, 'X':4}

maxlen = 32
word_size = 128
handle_data_path = './handle_data/web10w.utf-8'

def get_train_data():
    sentences = [] # 二维数组，元素为字列表
    labels = [] # 二维数组，元素为tag列表
    fp = open(handle_data_path, 'r')
    for line in fp:
        line = line.strip()
        if ' ' in line:
            words = []
            tags = []
            words_tags = line.split(' ')
            for word_tag in words_tags:
                if len(word_tag.split('/')) != 2:
                    # print(word_tag)
                    continue
                words.append(word_tag.split('/')[0])
                tags.append(corpus_tags[word_tag.split('/')[1]])
            sentences.append(words)
            labels.append(tags)
        else:
            sentences.append([line.split('/')[0]])
            labels.append([corpus_tags[line.split('/')[1]]])
    fp.close()
    return sentences, labels

def generate_train_data(tag_window = 5, embed_dim = 100, nb_class = 4, test = False):
    '''

    '''
    retain_padding = 'retain-padding'
    sentences, labels = get_train_data()
    data = pd.DataFrame(index=range(len(sentences)))
    data['sentences'] = sentences
    data['labels'] = labels
    data = data[data['sentences'].apply(len)<=maxlen] # 只保留长度小于maxlen的句子
    # tags = pd.Series({'S':0, 'B':1, 'M':2, 'E':3, 'X':4})

    all_chars = []
    for words in sentences:
        all_chars.extend(words)
    all_chars = pd.Series(all_chars).value_counts() # 统计每个字的出现次数（即去重）
    all_chars[:] = range(2, len(all_chars) + 2) # 为每个字打编号，从2开始，第1位给padding值
    all_chars[retain_padding] = 1

    # data['x'] = data['sentences'].apply(lambda a: np.array(list(all_chars[a])+[0]*(maxlen-len(a))))
    # data['y'] = data['labels'].apply(lambda x: np.array(list(map(lambda y:np_utils.to_categorical(y,5), tags[x].reshape((-1,1))))+[np.array([[0,0,0,0,1]])]*(maxlen-len(x))))
    padding = int((tag_window - 1) / 2)
    data['sentencesVec'] = data['sentences'].apply(lambda a: np.array([1]*padding + list(all_chars[a]) + [1]*padding)) # padding

    x = []
    y = []
    for index in data.index:
        sentence = data['sentencesVec'][index]
        cxt_labels = data['labels'][index]
        for i in range(len(sentence) - tag_window + 1):
            x.append(sentence[i:i+tag_window])
            # y.append(np.array(cxt_labels[i]))
            # y.append(np_utils.to_categorical(cxt_labels[i], nb_class))
            y.append(cxt_labels[i])
    data = pd.DataFrame(index=range(len(x)))

    data['x'] = x
    data['y'] = y

    vocabSize = len(all_chars) + 2
    word_weights = np.zeros((vocabSize, embed_dim), dtype='float32')
    random_weight = np.random.uniform(-0.1, 0.1, size=(embed_dim, ))
    w2vModel = word2vec.Word2Vec(sentences, size = embed_dim)
    for word in all_chars.index:
        word_id = all_chars[word]
        if word in w2vModel:
            word_weights[word_id, :] = w2vModel[word]
        else:
            random_weight = np.random.uniform(-0.1, 0.1, size=(embed_dim, ))
            word_weights[word_id, :] = random_weight
    
    # for i in data['x']:
        # print(i)
    # print(len(data['x']))
    # print(len(data['y']))
    # print(data)
    if test:
        return all_chars, word_weights
    return data, word_weights
            
def generate_test_data(embed_dim = 100, tag_window = 5, test_sentences=None):
    '''
    生成测试数据
    '''

    all_chars, word_weights = generate_train_data(test=True)
    retain_padding = 'retain-padding'
    data = pd.DataFrame(index=range(len(test_sentences)))
    data['sentences'] = test_sentences
    # print(data)
    # print(all_chars)
    padding = int((tag_window - 1) / 2)
    data['sentencesVec'] = data['sentences'].apply(lambda a: np.array([1]*padding + list(all_chars[a]) + [1]*padding)) # padding
    # print(data)

    x = []
    for index in data.index:
        sentence = data['sentencesVec'][index]
        for i in range(len(sentence) - tag_window + 1):
            x.append(sentence[i:i+tag_window])

    data = pd.DataFrame(index=range(len(x)))

    data['x'] = x
    # print(data)

    return data['x']




if __name__ == '__main__':
    generate_train_data()


