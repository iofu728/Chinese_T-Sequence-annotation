# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-06-22 16:55:08
# @Last Modified by:   v-huji
# @Last Modified time: 2019-06-25 11:27:49

import param
import numpy as np
import pickle
import tensorflow as tf
import time

from numba import jit
from typing import List
from util import echo, time_str, log
import sys, os
from evaluation import evaluate_ner
sys.path.append(os.getcwd())


def evaluation(y: List, y_predict: List, seq: List, types: str):
    ''' evaluation '''
    y_t, y_p_t = [], []
    for ii, jj in enumerate(y):
        y_t.extend(jj[:seq[ii]])
    for ii, jj in enumerate(y_predict):
        y_p_t.extend(jj[:seq[ii]])
    y = [int(ii > 1) for ii in y_t]
    y_predict = [int(ii > 1) for ii in y_p_t]
    change_idx, idx = [], -1
    for ii in seq:
        change_idx.append(ii + idx)
        idx += ii
    for ii in change_idx:
        try:
            y_predict[ii] = 1
        except:
            echo(0, ii, len(y_predict))

    p, r, macro_f1 = fastF1(y, y_predict)
    print(f"{types} P: {p:.2f}%, R: {r:.2f}%, Macro_f1: {macro_f1:.2f}%")

    return p, r, macro_f1

def evaluation_ner(y: List, y_predict: List, seq: List, types: str):
    ''' evaluation ner '''
    y_t, y_p_t = [], []
    for ii, jj in enumerate(y):
        y_t.extend(jj[:seq[ii]])
        y_t.append(-1)
    for ii, jj in enumerate(y_predict):
        y_p_t.extend(jj[:seq[ii]])
        y_t.append(-1)
    p, r, macro_f1, log_text = evaluate_ner(y_p_t, y_t)
    print(f"{types} P: {p:.2f}%, R: {r:.2f}%, Macro_f1: {macro_f1:.2f}%, {log_text}")
    return p, r, macro_f1, log_text
    

@jit
def fastF1(result, predict):
    ''' cws f1 score calculate '''
    recallNum = sum(result)
    precisionNum = sum(predict)
    last_result, last_predict, trueNum = -1, -1, 0
    for ii in range(len(result)):
        if result[ii] and result[ii] == predict[ii] and last_predict == last_result:
            trueNum += 1
        if result[ii]:
            last_result = ii
        if predict[ii]:
            last_predict = ii
    r = trueNum / recallNum if recallNum else 0
    p = trueNum / precisionNum if precisionNum else 0
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0

    return p * 100, r * 100, macro_f1 * 100


class BiLSTM_CRF_Model():
    ''' bi lstm crf model '''

    def __init__(self, max_len=200, vocab_size=None, num_tag=None, model_save_path=None, embed_size=256, hs=512):
        self.timestep_size = self.max_len = max_len
        self.vocab_size = vocab_size
        self.input_size = self.embedding_size = embed_size
        self.num_tag = num_tag
        self.hidden_size = hs
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.model_save_path = model_save_path

        # Embedding vector
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable("embedding", [vocab_size, self.embedding_size], dtype=tf.float32)
        self.train()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def lstm_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_lstm(self, X_inputs):
        # The actual input parameters and the converted output as follows:
        # X_inputs.shape = [batchsize, timestep_size]  ->  inputs.shape = [batchsize, timestep_size, embedding_size]
        self.inputs = tf.nn.embedding_lookup(self.embedding, X_inputs)
        # The input sentence is still padding filled data.
        # Calculate the actual length of each sentence, that is, the actual length of the non-zero non-padding portion.
        self.length = tf.reduce_sum(tf.sign(X_inputs), 1)
        self.length = tf.cast(self.length, tf.int32)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell(), self.lstm_cell(), self.inputs,
                                                                    sequence_length=self.length, dtype=tf.float32)

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, [-1, self.hidden_size * 2])
        return output

    def train(self):
        with tf.variable_scope('Inputs'):
            self.X_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='X_input')
            self.y_inputs = tf.placeholder(tf.int32, [None, self.timestep_size], name='y_input')

        bilstm_output = self.bi_lstm(self.X_inputs)

        echo(1, 'The shape of BiLSTM Layer output:', bilstm_output.shape)

        with tf.variable_scope('outputs'):
            softmax_w = self.weight_variable([self.hidden_size * 2, self.num_tag])
            softmax_b = self.bias_variable([self.num_tag])
            self.y_pred = tf.matmul(bilstm_output, softmax_w) + softmax_b

            self.scores = tf.reshape(self.y_pred, [-1, self.timestep_size, self.num_tag])
            print('The shape of Output Layer:', self.scores.shape)
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.y_inputs, self.length)
            self.loss = tf.reduce_mean(-log_likelihood)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss)

class BiLSTMTrain(object):
    ''' bi lstm train '''
     
    def __init__(self, data_train:List, data_dev:List, data_test:List, model:BiLSTM_CRF_Model, sa_type:param.SA_TYPE):
        self.data_train = data_train
        self.data_dev = data_dev
        self.data_test = data_test
        self.sa_type = sa_type
        self.model = model
        
    def train(self, max_epoch:int, max_max_epoch:int, tr_batch_size:int, display_num:int=5, do_finetune:bool=False):
        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        if do_finetune:
            ckpt = tf.train.latest_checkpoint(param.CHECK_DIR)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)
            print(f'Finetune ckpt: {ckpt} ...')

        echo(0, 'Train shape', self.data_train[0].shape, self.data_train[1].shape)
        echo(0, 'Dev shape', self.data_dev[0].shape, self.data_dev[1].shape)
        echo(0, 'Test shape', self.data_test[0].shape, self.data_test[1].shape)

        tr_batch_num = int(self.data_train[1].shape[0] / tr_batch_size)
        echo(3, tr_batch_num) 
        display_batch = int(tr_batch_num / display_num)  

        saver = tf.train.Saver(max_to_keep=10)  

        log(f'------- {time_str()} -------')

        for epoch in range(max_max_epoch):
            print(f'------  \033[92m{epoch} epochs \033[0m -------') 
            _lr = 0.01 if epoch < max_epoch else 0.005
            # !!! very import. The learning rate of CWS model & NER model not same
            if self.sa_type == param.SA_TYPE.NER:
                _lr /= 10 
            start_time = time.time()
            _losstotal, show_loss, best_dev_acc = 0.0, 0.0, -1

            for batch in range(tr_batch_num):  
                fetches = [self.model.loss, self.model.train_op]
                begin_index = batch * tr_batch_size
                end_index = (batch + 1) * tr_batch_size
                X_batch = self.data_train[0][begin_index:end_index]
                Y_batch = self.data_train[1][begin_index:end_index]
                # echo(0, X_batch[57,0], Y_batch[57,0])

                feed_dict = {self.model.X_inputs: X_batch,
                            self.model.y_inputs: Y_batch,
                            self.model.lr: _lr,
                            self.model.batch_size: tr_batch_size,
                            self.model.keep_prob: 0.5}
                _loss, _ = sess.run(fetches, feed_dict)  
                _losstotal += _loss
                show_loss += _loss
                if not (batch + 1) % display_batch:
                    train_p, train_r, train_macro_f1, train_log = self.test_epoch(self.data_train, sess, 'Train')
                    dev_p, dev_r, dev_macro_f1, dev_log = self.test_epoch(self.data_dev, sess, 'Dev')
                    if dev_macro_f1 > best_dev_acc:
                        test_p, test_r, test_macro_f1, predict = self.test_epoch(self.data_test, sess, 'Test')
                        best_dev_acc = dev_macro_f1
                        if self.sa_type == param.SA_TYPE.CWS:
                            log(f'{time_str()}|{epoch}-{batch}|{train_p:.2f}|{train_r:.2f}|{train_macro_f1:.2f}|{dev_p:.2f}|{dev_r:.2f}|{dev_macro_f1:.2f}|')
                        else:
                            log(f'{time_str()}|{epoch}-{batch}|{train_p:.2f}|{train_r:.2f}|{train_macro_f1:.2f}|{dev_p:.2f}|{dev_r:.2f}|{dev_macro_f1:.2f}| {train_log} | {dev_log}')
                    else:
                        if self.sa_type == param.SA_TYPE.CWS:
                            log(f'{time_str()}|{epoch}-{batch}|{train_p:.2f}|{train_r:.2f}|{train_macro_f1:.2f}|{dev_p:.2f}|{dev_r:.2f}|{dev_macro_f1:.2f}|')
                        else:
                            log(f'{time_str()}|{epoch}-{batch}|{train_p:.2f}|{train_r:.2f}|{train_macro_f1:.2f}|{dev_p:.2f}|{dev_r:.2f}|{dev_macro_f1:.2f}| {train_log} | {dev_log}')

                    echo(f'training loss={show_loss / display_batch}')
                    show_loss = 0.0
            mean_loss = _losstotal / tr_batch_num
            
            save_path = saver.save(sess, self.model.model_save_path, global_step=(epoch + 1))
            print('the save path is ', save_path)

            echo(1, f'Training {self.data_train[1].shape[0]}, loss={mean_loss:g} ')
            echo(2, f'Epoch training {self.data_train[1].shape[0]}, loss={mean_loss:g}, speed={time.time() - start_time:g} s/epoch')

        log(f"Best Dev Macro_f1: {best_dev_acc:.2f}%")
        log(f"Best Test P: {test_p:.2f}%, R: {test_r:.2f}%, Macro_f1: {test_macro_f1:.2f}%")

        sess.close()
        return predict

    def test_epoch(self, dataset, sess, types:str):
        ''' Test one epoch '''
        _batch_size = 128
        _y = dataset[1]
        data_size = _y.shape[0]
        batch_num = int(data_size / _batch_size) + 1
        predict = []
        fetches = [self.model.scores, self.model.length, self.model.transition_params]
        echo(1, 'Test Batch Num:', batch_num)

        for i in range(batch_num):
            begin_index = i * _batch_size
            end_index = min((i + 1) * _batch_size, data_size)
            X_batch = dataset[0][begin_index:end_index]
            Y_batch = dataset[1][begin_index:end_index]
            feed_dict = {self.model.X_inputs: X_batch,
                         self.model.y_inputs: Y_batch,
                         self.model.lr: 1e-5,
                         self.model.batch_size: _batch_size,
                         self.model.keep_prob: 1.0}
            test_score, test_length, transition_params = sess.run(fetches=fetches, feed_dict=feed_dict)

            for tf_unary_scores_, y_, sequence_length_ in zip(test_score, Y_batch, test_length):
                tf_unary_scores_ = tf_unary_scores_
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores_, transition_params)
                predict.append(viterbi_sequence)
            if not (i + 1) % 100:
                echo(0, i)
            
        
        if types == 'Test':
            pickle.dump(predict, open(f"{param.RESULT_PATH(self.sa_type)}.pkl", 'wb'))
            self.output_result(dataset, predict, types)
        echo(1, 'Predict Result shape:', np.array(predict).shape)
        if self.sa_type == param.SA_TYPE.CWS:
            p, r, macro_f1 = evaluation(_y, predict, dataset[2], types)
            return p, r, macro_f1, predict
        else:
            p, r, macro_f1, log_text = evaluation_ner(_y, predict, dataset[2], types)
            return p, r, macro_f1, log_text

    def output_result(self, data_set:List, predict:List, types:str):
        ''' output result '''

        predict = sum([ii[:data_set[2][jj]] for jj, ii in enumerate(predict)], [])

        idx, test_predict_text = 0, []
        if self.sa_type == param.SA_TYPE.CWS:
            for ii in data_set[-1]:
                temp_len = len(ii)
                temp_tag = predict[idx: idx + temp_len]
                temp_text = ''.join([f'{kk[0]}{"" if temp_tag[jj] < 2 else " "}' for jj, kk in enumerate(ii)]).strip()
                test_predict_text.append(temp_text)
                idx += temp_len
        else:
            for ii in data_set[-1]:
                temp_len = len(ii)
                temp_tag = predict[idx: idx + temp_len]
                temp_text = [f'{kk[0]} {param.NER_ID2LAB[temp_tag[jj]]}' for jj, kk in enumerate(ii)]
                test_predict_text.extend([*temp_text, ''])

        output_path = f'{param.RESULT_PATH(self.sa_type)}_{time_str()}.txt'
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(test_predict_text))

    def predict(self):
        config = tf.ConfigProto()
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.latest_checkpoint('./checkpoint')
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        _ = self.test_epoch(self.data_test, sess, 'Test') 
        echo('Over')

def test_params(num_seq: int, num_word: int, word_size: int, num_tag: int):
    x = np.random.randint(word_size, size=[num_seq, num_word]).astype(np.int32)
    y = np.random.randint(num_tag, size=[num_seq, num_word]).astype(np.int32)
    seq = np.random.randint(0, 5, size=[num_seq]).astype(np.int32)
    origin_set = []
    for ii, jj in enumerate(seq):
        origin_set.append([(x[ii][kk], y[ii][kk]) for kk in range(jj)])
    return x, y, seq, origin_set


if __name__ == "__main__":
    sa_type = param.SA_TYPE.NER
    num_seq = 10
    num_word = 20
    word_size = 3333
    num_tag = 7 if sa_type == param.SA_TYPE.NER else 4
    data_train = test_params(num_seq, num_word, word_size, num_tag)
    data_dev = test_params(10, num_word, word_size, num_tag)
    data_test = test_params(10, num_word, word_size, num_tag)
    param.change_run_id('Unit_Test')

    model = BiLSTM_CRF_Model(max_len=num_word, 
                            vocab_size=word_size, 
                            num_tag=num_tag, 
                            model_save_path='./checkpoint/checkpoint', 
                            embed_size=256,  
                            hs=512)
    train = BiLSTMTrain(data_train, data_dev, data_test, model, sa_type)
    train.train(100, 200, 5, 1)
    
    

