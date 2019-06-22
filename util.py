# -*- coding: utf-8 -*-
# @Author: v-huji
# @Date:   2019-06-21 10:24:27
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-22 16:37:58


import param
import os
import pickle
import platform
import time
import numpy as np

from sklearn.model_selection import train_test_split

start, spend_list = [], []


def load_data(sa_type: param.SA_TYPE) -> list:
    ''' load data '''
    types = 'CWS' if param.SA_TYPE.CWS == sa_type else 'NER'

    if not os.path.exists(param.PKL_SET_PATH('Train')[types]):
        prepare_data()

    train = load_bigger(param.PKL_SET_PATH('Train')[types])
    dev = load_bigger(param.PKL_SET_PATH('Dev')[types])
    test = load_bigger(param.PKL_SET_PATH('Test')[types])
    return train, dev, test


def prepare_data():
    ''' prepare data '''
    if not os.path.exists(param.PKL_DIR):
        echo(3, 'mkdiring data/pkl')
        os.makedirs(param.PKL_DIR)

    basic_prepare_data(param.SA_TYPE.CWS)
    basic_prepare_data(param.SA_TYPE.NER)


def basic_prepare_data(sa_type: param.SA_TYPE):
    ''' basic prepare data function '''
    read_function = read_cws_data if param.SA_TYPE.CWS == sa_type else read_ner_data
    types = 'CWS' if param.SA_TYPE.CWS == sa_type else 'NER'
    echo(3, 'parapre', types, 'data')
    train_set = read_function(param.ORIGIN_SET_PATH('Train')[types])
    test_set = read_function(param.ORIGIN_SET_PATH('Test')[types])
    train_set, dev_set = train_test_split(train_set, test_size=0.3)
    dump_bigger(train_set, param.PKL_SET_PATH('Train')[types])
    dump_bigger(dev_set, param.PKL_SET_PATH('Dev')[types])
    dump_bigger(test_set, param.PKL_SET_PATH('Test')[types])


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
    # end_time(version)
    return dataLists


def read_data(filePath: str) -> list:
    ''' read data'''
    with open(filePath, 'r', encoding='utf-16') as f:
        origin_data = [ii.strip() for ii in f.readlines()]
    return origin_data


def echo(color: int, *args):
    ''' echo log @param: color: 0 -> red, 1 -> green, 2 -> yellow, 3 -> blue '''
    args = ' '.join([str(ii) for ii in args])
    if param.is_service:
        with open(param.log_path, 'a') as f:
            f.write('{}\n'.format(args))
        return
    colors = {'red': '\033[91m', 'green': '\033[92m',
              'yellow': '\033[93m', 'blue': '\033[94m'}
    if type(color) != int or not color in list(range(len(colors.keys()))) or platform.system() == 'Windows':
        print(args)
    else:
        print(list(colors.values())[color], args, '\033[0m')


def dump_bigger(data, output_file: str):
    ''' pickle.dump big file which size more than 4GB '''
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file: str):
    ''' pickle.load big file which size more than 4GB '''
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


def time_str(timestamp: int = -1, format: str = '%Y-%m-%d %H:%M:%S') -> str:
    ''' time str '''
    if timestamp > 0:
        return time.strftime(format, time.localtime(timestamp))
    return time.strftime(format, time.localtime(time.time()))


def log(log_str: str):
    ''' log record '''
    with open(param.log_path, 'a') as f:
        f.write(f'{log_str}\n')


def begin_time() -> int:
    ''' multi-version time manage '''
    global start
    start.append(time.time())
    return len(start) - 1


def end_time(version: int, mode: int = 1):
    time_spend = time.time() - start[version]
    if mode:
        echo(2, f'{time_spend:.3f}s')
    else:
        return time_spend


if __name__ == "__main__":
    prepare_data()
