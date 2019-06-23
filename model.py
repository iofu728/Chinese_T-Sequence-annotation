# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-06-22 19:05:22
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-23 14:50:03

import param
import numpy as np
import pickle
import re
from random import random

from modelLayer.bilstm_crf import BiLSTMTrain, BiLSTM_CRF_Model
from collections import Counter
from enum import Enum
from numba import jit
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from typing import List, Dict
from util import echo
import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer


class EMBED_TYPE(Enum):
    ONE_HOT = 0
    TF_IDF = 1
    FAST_TEXT = 2
    BERT = 3


class MODEL_TYPE(Enum):
    BILSTM_CRF = 0


embed_type = EMBED_TYPE.BERT


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

    def statistical_data(self, train_set: List, dev_set: List, test_set: List, do_reshape: bool = True):
        ''' statistical data '''
        if embed_type == EMBED_TYPE.FAST_TEXT or embed_type == EMBED_TYPE.BERT:
            pre_set = [*train_set, *test_set, *dev_set]
        else:
            pre_set = train_set
        word_list = sum([[jj[0] for jj in ii] for ii in pre_set], [])
        word_set = ['[OOV]', *list(set(word_list))]
        echo(1, len(word_list))
        word2id = {jj: ii for ii, jj in enumerate(word_set)}

        if not do_reshape:
            train_set = [[(word2id[jj] if jj in word2id else 0, con.CWS_LAB2ID[kk])
                          for jj, kk in ii] for ii in train_set]
            dev_set = [[(word2id[jj] if jj in word2id else 0, con.CWS_LAB2ID[kk])
                        for jj, kk in ii] for ii in dev_set]
            test_set = [[(word2id[jj] if jj in word2id else 0, con.CWS_LAB2ID[kk])
                         for jj, kk in ii] for ii in test_set]
            self.train_set = train_set
            self.dev_set = dev_set
            self.test_set = test_set
        else:
            ''' a way to reduce memory using '''
            self.word2id = word2id
            self.train_set = self.reshape_data(train_set)
            self.dev_set = self.reshape_data(dev_set)
            self.test_set = self.reshape_data(test_set)

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

    def reshape_data(self, origin_set: List, MAX_LEN: int = 200) -> List:
        ''' reshape data '''
        data_set = sum([[(self.word2id[jj] if jj in self.word2id else 0, param.CWS_LAB2ID[kk])
                         for jj, kk in ii] for ii in origin_set], [])
        data_len = len(data_set)
        pad_len = MAX_LEN - data_len % MAX_LEN
        echo(2, data_len, pad_len)
        data_set = np.array([*data_set, *[(0, 0)] * pad_len])
        reshape_data = data_set.reshape([MAX_LEN, len(data_set) // MAX_LEN, 2])
        if not pad_len:
            last_id = reshape_data.shape[0] - 1
            reshape_data = [jj if ii != last_id else jj[:MAX_LEN - pad_len]
                            for ii, jj in enumerate(reshape_data)]
        return reshape_data

    def prepare_data(self, now_set: List, origin_set: List) -> (List, List, List):
        ''' prepare_data '''
        MAX_LEN = max([len(ii) for ii in now_set])
        print(f'MAX_LEN: {MAX_LEN}')
        seq_len = [len(ii) for ii in now_set]
        seq = [len(ii) for ii in origin_set]
        x = self.pad_pattern(now_set, 0, MAX_LEN)
        y = self.pad_pattern(now_set, 1, MAX_LEN)
        if embed_type == EMBED_TYPE.ONE_HOT:
            x = self.one_hot(x)
        elif embed_type == EMBED_TYPE.TF_IDF:
            x = self.tf_idf(x, seq)
        elif embed_type == EMBED_TYPE.FAST_TEXT:
            x = self.char_embed(x)
        elif embed_type == EMBED_TYPE.BERT:
            x = self.bert(x)

        return x, y, seq_len, seq

    def pad_pattern(self, origin_set: List, idx: int, MAX_LEN: int) -> List:
        ''' pad pattern '''
        return [[jj[idx] for jj in ii] + [0] * (MAX_LEN - len(ii)) for ii in origin_set]

    def one_hot(self, word_set: List):
        ''' one hot embed '''
        word_set = np.array(word_set)
        num_fea = len(self.word2id)
        num_seq, num_word = word_set.shape
        echo(0, num_seq, num_word, num_fea)

        return np.squeeze(np.eye(num_fea)[word_set.reshape(-1)]).reshape([num_seq, num_word, num_fea]).astype(np.int16)

    def tf_idf(self, word_set: List, seq: List, n_gram: int = 4):
        ''' tf-idf embed'''

        word_set = np.array(word_set)
        num_fea = len(self.word2id)
        num_seq, num_word = word_set.shape
        echo(0, num_seq, num_word, num_fea)
        origin_set_one = word_set.reshape(-1)

        n_gram_dict = self.prepare_n_gram(origin_set_one, seq, n_gram)
        n_gram_dict += [{}] * (num_seq * num_word - len(n_gram_dict))

        to_pipeline = [DictVectorizer(), TfidfTransformer()]
        data_transformer = make_pipeline(*to_pipeline)
        transformed = np.array(data_transformer.fit_transform(
            n_gram_dict).todense(), dtype=np.float16)
        echo(1, 'Tf idf Over')
        for ii in n_gram_dict[0].keys():
            echo(0, transformed[0][ii])
        return transformed.reshape([num_seq, num_word, num_fea])

    def bert(self, word_set: List, seq: List, n_gram: int = 4):
        ''' bert '''
        bert_dir = '../bert/chinese_L-12_H-768_A-12'
        bert = BertModel.from_pretrained(bert_dir)
        tokenizer = BertTokenizer.from_pretrained(
            f'{bert_dir}/chinese_L-12_H-768_A-12/vocab.txt')
        word_set = np.array(word_set)
        num_fea = len(self.word2id)
        num_seq, num_word = word_set.shape
        echo(0, num_seq, num_word, num_fea)
        origin_set_one = word_set.reshape(-1)
        id2word = {jj: ii for ii, jj in self.word2id.items()}

        n_gram_dict = self.prepare_n_gram(origin_set_one, seq, n_gram)
        n_gram_word = [
            ' '.join([id2word[jj] for jj, kk in ii.items() if kk]) for ii in n_gram_dict]
        transformed = []
        for ii in n_gram_word:
            ids = torch.tensor(
                [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ii))])
            transformed.append(
                bert(ids, output_all_encoded_layers=False)[-1][0].detach().numpy())
        transformed += [np.zeros(len(transformed[0]))] * \
            (num_seq * num_word - len(n_gram_dict))

        echo(1, 'Bert Over')

        return transformed.reshape([num_seq, num_word, num_fea])

    @jit
    def prepare_n_gram(self, origin_set_one: List, seq: List, n_gram: int = 4):
        ''' prepare n gram'''
        idx, origin_set, n_gram_dict, exist_word, no_exist_word = 0, [], [], {}, []
        for ii in seq:
            origin_set.append(list(origin_set_one[idx:idx+ii]))
            idx += ii
        echo(1, 'Seq Length Over')
        for ii in origin_set:
            for jj, _ in enumerate(ii):
                t_seq_len = len(ii)
                begin_idx = max(0, jj - n_gram)
                end_idx = min(t_seq_len, jj + n_gram)
                n_gram_word = ii[begin_idx:end_idx]
                n_gram_count = dict(Counter(n_gram_word))
                n_gram_dict.append(n_gram_count)
                for kk, mm in n_gram_count.items():
                    exist_word[kk] = mm
        echo(1, 'n_gram Over')
        for ii in self.word2id.values():
            if ii not in exist_word:
                no_exist_word.append(ii)
        for ii in no_exist_word:
            n_gram_dict[-1][ii] = 0

        echo(1, len(no_exist_word), 'no exist Over')
        return n_gram_dict

    def char_embed(self, word_set: List):
        ''' char embed '''
        if embed_type == EMBED_TYPE.FAST_TEXT:
            embed_path = 'embedding/gigaword_chn.all.a2b.uni.ite50.vec'
        embed = self.load_embedding(embed_path)
        echo(1, 'len of embed', len(embed))
        word_set = np.array(word_set)
        num_fea = len(list(embed.values())[0])
        num_seq, num_word = word_set.shape
        echo(0, num_seq, num_word, num_fea)
        word_set = word_set.reshape(-1)
        result_set = np.array([embed[ii] if ii in embed else np.zeros(
            num_fea) for ii in word_set])
        return result_set.reshape([num_seq, num_word, num_fea])

    def load_embedding(self, data_path: str) -> Dict[str, List[float]]:
        ''' load embedding '''
        with open(data_path) as f:
            origin_embed = [ii.strip() for ii in f.readlines()]
        origin_embed = [ii for ii in origin_embed if ii.split(' ')[
            0] in self.word2id]
        embed = {self.word2id[ii.split(' ')[0]]: np.array(
            ii.split(' ')[1:]).astype(np.float16) for ii in origin_embed}
        return embed

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
