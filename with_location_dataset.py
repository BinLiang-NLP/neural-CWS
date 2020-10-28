#!/bin/env python
#-*- coding:utf8 -*-

__author__ = 'BinLiang <bin.liang@stu.hit.edu.cn>'
__date__ = '2017-06-14'

import numpy as np
import pandas as pd
from gensim.models import word2vec

corpus_tags = {'S':0, 'B':1, 'M':2, 'E':3, 'X':4}

max_len = 512
word_size = 128
handle_data_path = './handle_data/combineRule_10w.utf-8'

def get_train_data():
    '''
    加载训练数据集
    '''
    sentences = [] # 二维数组，元素为字列表
    labels = [] # 二维数组，元素为tag列表
    locations = [] # 二维数组，元素为字在句子中的位置列表
    fp = open(handle_data_path, 'r', encoding='utf-8')
    for line in fp:
        index = 0
        line = line.strip()
        if ' ' in line:
            words = []
            tags = []
            pos = []
            # index = 0
            words_tags = line.split(' ')
            for word_tag in words_tags:
                index += 1 # 位置编码从1开始，0号给padding
                if len(word_tag.split('/')) != 2:
                    continue
                words.append(word_tag.split('/')[0])
                tags.append(corpus_tags[word_tag.split('/')[1]])
                pos.append(index)
            sentences.append(words)
            labels.append(tags)
            locations.append(pos)
        else:
            index += 1
            sentences.append([line.split('/')[0]])
            labels.append([corpus_tags[line.split('/')[1]]])
            locations.append([index])
    fp.close()
    return sentences, labels, locations

def generate_train_data(tag_window=5, embed_dim=100, nb_class=4, test=False):
    '''
    生成训练集数据
    返回字的id列表，embedding权重矩阵
    '''
    retain_padding = 'retain-padding'
    sentences, labels, locations = get_train_data()
    data = pd.DataFrame(index=range(len(sentences)))
    data['sentences'] = sentences
    data['labels'] = labels
    data['locations'] = locations
    data = data[data['sentences'].apply(len) <= max_len] # 只保留长度小于maxlen的句子
    # tags = pd.Series({'S':0, 'B':1, 'M':2, 'E':3, 'X':4})

    all_chars = []
    for words in sentences:
        all_chars.extend(words)
    all_chars = pd.Series(all_chars).value_counts() # 统计每个字的出现次数（即去重）
    all_chars[:] = range(2, len(all_chars) + 2) # 为每个字打编号，从2开始，第1位给padding值
    all_chars[retain_padding] = 1

    all_pos = []
    for pos in locations:
        all_pos.extend(pos)
    all_pos = pd.Series(all_pos).value_counts() # 统计每位置值的出现次数（即去重）
    location_input_dim = len(all_pos) + 1
    all_pos[:] = range(1, len(all_pos) + 1) # 为每个字打编号，从1开始，第0位给padding值

    padding = int((tag_window - 1) / 2)
    data['sentencesVec'] = data['sentences'].apply(lambda a: np.array(
        [1]*padding + list(all_chars[a]) + [1]*padding)) # padding
    data['locationsVec'] = data['locations'].apply(lambda a: np.array(
        [0]*padding + list(a) + [0]*padding)) # padding

    data_x = []
    data_y = []
    data_l = []
    for index in data.index:
        sentence = data['sentencesVec'][index]
        cxt_labels = data['labels'][index]
        cxt_locations = data['locationsVec'][index]
        for i in range(len(sentence) - tag_window + 1):
            data_x.append(sentence[i:i+tag_window])
            data_y.append(cxt_labels[i])
            data_l.append(cxt_locations[i:i+tag_window])
            # print(cxt_locations[i:i+tag_window])
    data = pd.DataFrame(index=range(len(data_x)))

    data['x'] = data_x
    data['l'] = data_l
    data['y'] = data_y
    # print(data['l'])

    vocab_size = len(all_chars) + 2
    word_weights = np.zeros((vocab_size, embed_dim), dtype='float32')
    random_weight = np.random.uniform(-0.1, 0.1, size=(embed_dim, ))
    w2vModel = word2vec.Word2Vec(sentences, size = embed_dim)
    for word in all_chars.index:
        word_id = all_chars[word]
        if word in w2vModel:
            word_weights[word_id, :] = w2vModel[word]
        else:
            random_weight = np.random.uniform(-0.1, 0.1, size=(embed_dim, ))
            word_weights[word_id, :] = random_weight
    if test:
        return all_chars, word_weights
    return data, word_weights, max_len

def generate_test_data(tag_window=5, test_sentences=None):
    '''
    生成测试数据
    '''

    all_chars = generate_train_data(test=True)
    retain_padding = 'retain-padding'
    data = pd.DataFrame(index=range(len(test_sentences)))
    data['sentences'] = test_sentences
    # print(data)
    # print(all_chars)
    padding = int((tag_window - 1) / 2)
    data['sentencesVec'] = data['sentences'].apply(lambda a: np.array(
        [1]*padding + list(all_chars[a]) + [1]*padding)) # padding

    x = []
    for index in data.index:
        sentence = data['sentencesVec'][index]
        for i in range(len(sentence) - tag_window + 1):
            x.append(sentence[i:i+tag_window])

    data = pd.DataFrame(index=range(len(x)))

    data['x'] = x
    print(data)

    return data['x']

if __name__ == '__main__':
    generate_train_data()
