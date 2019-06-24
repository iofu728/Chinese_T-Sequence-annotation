# -*- coding: utf-8 -*-
# @Author: v-huji
# @Date:   2019-06-21 10:22:48
# @Last Modified by:   v-huji
# @Last Modified time: 2019-06-24 14:15:22

import os

from sklearn.model_selection import train_test_split

import param
from util import dump_bigger, echo, load_bigger


def load_data(sa_type: param.SA_TYPE) -> list:
    ''' load data '''
    types = 'CWS' if param.SA_TYPE.CWS == sa_type else 'NER'

    if not os.path.exists(param.PKL_SET_PATH('Train')[types]):
        prepare_data()

    train = load_bigger(param.PKL_SET_PATH('Train')[types])
    dev = load_bigger(param.PKL_SET_PATH('Dev')[types])
    test = load_bigger(param.PKL_SET_PATH('Test')[types])
    return train, dev, test


def prepare_data(mode=None):
    ''' prepare data '''
    if not os.path.exists(param.PKL_DIR):
        echo(3, 'mkdiring data/pkl')
        os.makedirs(param.PKL_DIR)

    basic_prepare_data(param.SA_TYPE.CWS, mode)
    basic_prepare_data(param.SA_TYPE.NER, mode)


def basic_prepare_data(sa_type: param.SA_TYPE, mode=None):
    ''' basic prepare data function '''
    read_function = read_cws_data if param.SA_TYPE.CWS == sa_type else read_ner_data
    types = 'CWS' if param.SA_TYPE.CWS == sa_type else 'NER'
    echo(3, 'parapre', types, 'data')
    train_set = read_function(param.ORIGIN_SET_PATH('Train')[types])
    test_set = read_function(param.ORIGIN_SET_PATH('Test')[types])
    train_set, dev_set = train_test_split(train_set, test_size=0.3)
    if mode == None:
        dump_bigger(train_set, param.PKL_SET_PATH('Train')[types])
        dump_bigger(dev_set, param.PKL_SET_PATH('Dev')[types])
        dump_bigger(test_set, param.PKL_SET_PATH('Test')[types])
    else:
        write_data(train_set, 'Train')
        write_data(dev_set, 'Dev')
        write_data(test_set, 'Test')


def write_data(data, types: str):
    with open(f'{param.PKL_DIR}{types}_data.txt', 'w') as f:
        for ii in data:
            f.write('\n'.join([f'{kk} {mm}' for kk, mm in ii]) + '\n\n')


def read_ner_data(filePath: str) -> list:
    ''' read ner data '''
    # version = begin_time()
    fileLists = read_data(filePath)
    dataLists, temp_data = [], []
    for line in fileLists:
        if not len(line):
            dataLists.append(temp_data)
            temp_data = []
        else:
            if len(line.split()) == 1:
                word, label = line, 'N'
            else:
                word, label = line.split(' ', 1)
            temp_data.append((word, label))
    # end_time(version)
    return dataLists


def read_cws_data(filePath: str) -> list:
    ''' read cws data '''
    # version = begin_time()
    fileLists = read_data(filePath)
    dataLists, temp_data = [], []
    for line in fileLists:
        if not len(line):
            continue
        for word in line.split():
            if len(word) == 1:
                temp_data.append((word, 'S'))
            else:
                temp_data.append((word[0], 'B'))
                for char in word[1:-1]:
                    temp_data.append((char, 'M'))
                temp_data.append((word[-1], 'E'))
        dataLists.append(temp_data)
        temp_data = []
    # end_time(version)
    return dataLists


def read_data(filePath: str) -> list:
    ''' read data'''
    with open(filePath, 'r', encoding='utf-16') as f:
        origin_data = [ii.strip() for ii in f.readlines()]
    return origin_data


if __name__ == "__main__":
    prepare_data('TXT')
