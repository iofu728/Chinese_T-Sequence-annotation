# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-06-22 19:05:22
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-27 13:47:15

import pickle
import re
from enum import Enum
from typing import Dict, List

import numpy as np
from numba import jit

import param
from modelLayer.bilstm_crf import BiLSTM_CRF_Model, BiLSTMTrain
from util import echo


class MODEL_TYPE(Enum):
    BILSTM_CRF = 0


class Model:
    ''' model of sequence annotation '''

    def __init__(self, train_set: List, dev_set: List, test_set: List,
                 sa_type: param.SA_TYPE = param.SA_TYPE.CWS, model_type: MODEL_TYPE = MODEL_TYPE.BILSTM_CRF):
        self.MAX_LEN = 0
        self.origin_train_set = train_set
        self.origin_dev_set = dev_set
        self.origin_test_set = test_set
        self.model_type = model_type
        self.sa_type = sa_type
        self.load_word(train_set, dev_set, test_set)

    def load_word(self, train_set: List, dev_set: List, test_set: List):
        ''' load word '''
        echo(0, 'begin load word')
        total_set = [*train_set, *dev_set, *test_set]
        echo(0, 'begin load word I ', len(total_set))

        word_list = set(' '.join([' '.join([ii[0] for ii in jj])
                                  for jj in total_set]).split())
        echo(0, 'begin load word II')
        word_set = ['[PAD]', *list(set(word_list))]
        word_set = self.load_word_list(total_set)
        echo(1, len(word_set))
        word2id = {jj: ii for ii, jj in enumerate(word_set)}
        self.word2id = word2id
        MAX_LEN = max([len(ii) for ii in total_set])
        self.train_set = self.load_word_once(train_set, MAX_LEN)
        self.dev_set = self.load_word_once(dev_set, MAX_LEN)
        self.test_set = self.load_word_once(test_set, MAX_LEN)
        self.MAX_LEN = MAX_LEN

    # @jit
    def load_word_list(self, origin_set: list):
        result = {}
        for ii in origin_set:
            for jj in ii:
                result[jj[0]] = 0
        return list(result.keys())

    def load_word_once(self, origin_set: List, MAX_LEN: int) -> List:
        ''' load word once '''
        data_set = [[self.word2id[jj[0]] if jj[0] in self.word2id else 0 for jj in ii] +
                    [0] * (MAX_LEN - len(ii)) for ii in origin_set]
        lab2id = param.CWS_LAB2ID if self.sa_type == param.SA_TYPE.CWS else param.NER_LAB2ID
        label = [[lab2id[jj[1]] for jj in ii] + [0] *
                 (MAX_LEN - len(ii)) for ii in origin_set]
        seq = [len(ii) for ii in origin_set]
        echo(1, np.array(data_set).shape, np.array(seq).shape)
        return [np.array(data_set), np.array(label), np.array(seq), origin_set]

    def run_model(self):
        ''' run model schedule '''
        self.run_bilstm_crf()

    def run_bilstm_crf(self):
        ''' run bilstm crf '''
        label2id = param.CWS_LAB2ID if self.sa_type == param.SA_TYPE.CWS else param.NER_LAB2ID
        model = BiLSTM_CRF_Model(max_len=self.MAX_LEN,
                                 vocab_size=len(self.word2id),
                                 num_tag=len(label2id),
                                 model_save_path=param.CHECK_PATH,
                                 embed_size=256,
                                 hs=512)
        train = BiLSTMTrain(self.train_set, self.dev_set,
                            self.test_set, model, self.sa_type)
        predict = train.train(100, 200, 64)
