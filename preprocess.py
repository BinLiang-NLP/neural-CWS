#!/bin/env python
#-*- coding:utf-8 -*-

__author__ = 'Akuchi <liangbin05@baidu.com>'
__date__ = '2017-06-14'

"""
处理icwb2原始语料
将句子中的字按词来序列标注
"""

import sys

def load_data():
    '''

    '''
    sentence_list = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if ' ' in line:
            words = line.split(' ')
        else:
            words = []
            words.append(line)
        sentence = []
        for word in words:
            if not word:
                continue
            if len(word) == 1:
                sentence.append(word + '/' + 'S')
            elif len(word) == 2:
                sentence.append(word[0] + '/' + 'B' + ' ' + word[1] + '/' + 'E')
            else:
                string = word[0] + '/' + 'B' + ' '
                for i in range(1,len(word) - 1):
                    string += word[i] + '/' + 'M' + ' '
                string += word[-1] + '/' + 'E'
                sentence.append(string)
        sentence = ' '.join(sentence)
        sentence_list[sentence] = 1
    return sentence_list

def pre_data():
    '''

    '''
    sentence_list = load_data()
    for sentence in sentence_list:
        print(sentence)

if __name__ == '__main__':
    pre_data()
