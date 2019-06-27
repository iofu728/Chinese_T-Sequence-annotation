# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-06-24 22:33:33
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-27 14:10:44

from __future__ import absolute_import, division, print_function

import collections
import os
import pickle
import re

import metrics
import numpy as np
import param
import tensorflow as tf
from absl import flags, logging
from bert import modeling, optimization, tokenization
from numba import jit
from util import log, time_str, dump_bigger
from dataLoad import prepare_data


## Required parameters
flags.DEFINE_string("middle_output", "result", 'middle output dir')
flags.DEFINE_string("crf", "True", "use crf!")
flags.DEFINE_string("data_dir", None, 'data dir')
flags.DEFINE_string("bert_config_file", None, 'configure')
flags.DEFINE_string("task_name", None, 'task name')
flags.DEFINE_string("vocab_file", None, 'vocab')
flags.DEFINE_string("output_dir", None, 'output')
flags.DEFINE_string("init_checkpoint", None, 'init ckpt')
flags.DEFINE_bool("do_lower_case", True, 'do lower')
flags.DEFINE_integer("max_seq_length", 128, 'max seq len')
flags.DEFINE_bool("do_train", False, 'train?')
flags.DEFINE_bool("do_eval", False, 'eval?')
flags.DEFINE_bool("do_predict", False, 'predict?')

flags.DEFINE_integer("train_batch_size", 32, 'train batch')
flags.DEFINE_integer("eval_batch_size", 8, 'eval batch')
flags.DEFINE_integer("predict_batch_size", 8, 'predict batch')
flags.DEFINE_float("learning_rate", 2e-5, 'lr')
flags.DEFINE_float("num_train_epochs", 3.0, 'train epoch')
flags.DEFINE_float("warmup_proportion", 0.1, 'warmup precent')
flags.DEFINE_integer("save_checkpoints_steps", 1000, 'ckpt save steps')
flags.DEFINE_integer("iterations_per_loop", 1000, 'loop')

# for tpu
flags.DEFINE_bool("use_tpu", False, 'tpu?')
flags.DEFINE_string("tpu_name", None, '!tpu')
flags.DEFINE_string("tpu_zone", None, '!tpu')
flags.DEFINE_string("gcp_project", None, '!tpu')
flags.DEFINE_string("master", None, 'tpu cluster')
flags.DEFINE_integer("num_tpu_cores", 8, 'tpu cores')

FLAGS = flags.FLAGS

class InputExample(object):
  """A single training/test example for simple sequence classification."""
  def __init__(self, guid, text, label=None):
    self.guid = guid
    self.text = text
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids,
               mask,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.mask = mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file: str):
        """Read a BIO data!"""
        with open(input_file, 'r') as f:
            rf = [ii.strip() for ii in f.readlines()]

        lines, words, labels = [], [], []
        for line in rf:
            if not len(line):
                if not len(words):
                    continue
                lines.append((' '.join(labels), ' '.join(words)))
                words, labels = [], []
            else:
                word, label = line.split()
                words.append(word)
                labels.append(label)
        return lines

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir: str):
        return self._create_example(
            self._read_data(os.path.join(data_dir, f"Train_data_{FLAGS.task_name}.txt")), "train"
        )

    def get_dev_examples(self, data_dir: str):
        return self._create_example(
            self._read_data(os.path.join(data_dir, f"Dev_data_{FLAGS.task_name}.txt")), "dev"
        )

    def get_test_examples(self,data_dir: str, types: str):
        return self._create_example(
            self._read_data(os.path.join(data_dir, f"{types}_data_{FLAGS.task_name}.txt")), "test"
        )


    def get_labels(self):
        if 'CWS' == FLAGS.task_name:
            return ["[PAD]", "B", "M", "E", "S", "[CLS]","[SEP]"]
        else:
            return ["[PAD]", "N", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]

    def _create_example(self, lines: list, set_type: str):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list: list, max_seq_length: int, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map, tokens, labels = {}, [], []
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i
    with open(f"{FLAGS.middle_output}/{FLAGS.task_name}_label2id.pkl",'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split()
    labellist = example.label.split()
    for i,(word, label) in enumerate(zip(textlist, labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens, segment_ids, label_ids = [], [], []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1]*len(input_ids)
    #use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature,ntokens,label_ids

def filed_based_convert_examples_to_features(examples, label_list: list, max_seq_length: int, tokenizer, output_file: str, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature,ntokens,label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)
        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens,batch_labels

def file_based_input_fn_builder(input_file, seq_length: int, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

# all above are related to data preprocess
# Following i about the model

def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass,activation=None)
    return linear(hiddenlayer)

def crf_loss(logits, labels, mask, num_labels, mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
                "transition",
                shape=[num_labels, num_labels],
                initializer=tf.contrib.layers.xavier_initializer()
        )
    
    log_likelihood,transition = tf.contrib.crf.crf_log_likelihood(logits, labels,transition_params=trans , sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)
   
    return loss, transition

def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12 # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config = bert_config,
        is_training = is_training,
        input_ids = input_ids,
        input_mask = mask,
        token_type_ids = segment_ids,
        use_one_hot_embeddings = use_one_hot_embeddings
        )

    output_layer = model.get_sequence_output()
    #output_layer shape is
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
    logits = hidden2tag(output_layer, num_labels)
    logits = tf.reshape(logits,[-1,FLAGS.max_seq_length,num_labels])
    if FLAGS.crf:
        mask2len = tf.reduce_sum(mask,axis=1)
        loss, trans = crf_loss(logits,labels,mask,num_labels,mask2len)
        predict,viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        return (loss, logits,predict)

    else:
        loss,predict  = softmax_layer(logits, labels, num_labels, mask)

        return (loss, logits, predict)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if FLAGS.crf:
            (total_loss, logits,predicts) = create_model(bert_config, is_training, input_ids,
                                                            mask, segment_ids, label_ids,num_labels, 
                                                            use_one_hot_embeddings)

        else:
            (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                            mask, segment_ids, label_ids,num_labels, 
                                                            use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits,num_labels,mask):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels-1, weights=mask)
                return {
                    "confusion_matrix":cm
                }
                #
            eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i, types:str):
    token = batch_tokens[i]
    predict = prediction
    true_l = id2label[batch_labels[i]]
    if FLAGS.task_name == 'NER':
        if token!="[PAD]" and token!="[CLS]" and true_l!="X":
            if predict=="X" and not predict.startswith("##"):
                predict="N"
            if types == 'Test':
                line = f"{token} {predict}\n"
            else:
                line = f"{token} {true_l} {predict}\n"
            wf.write(line)
    else:
        if token in ['PAD]', '[SEP]']:
            return
        if token == '[CLS]':
            wf.write('\n')
        elif predict < 3:
            wf.write(token)
        elif predict < 5:
            wf.write(f'{token} ')
        else:
            wf.write(f'{token}\n')

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


def conlleval(label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "./conlleval_rev.pl"
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics

def Writer(output_predict_file:str, result, batch_tokens, batch_labels, id2label, types:str):
    with open(output_predict_file,'w') as wf:
        if  FLAGS.crf:
            predictions  = []
            for m, pred in enumerate(result):
                predictions.extend(pred)
            for i,prediction in enumerate(predictions):
                _write_base(batch_tokens,id2label,prediction,batch_labels,wf,i, types)
                
        else:
            for i,prediction in enumerate(result):
                _write_base(batch_tokens,id2label,prediction,batch_labels,wf,i, types)

def evaluation(processor, label_list, tokenizer, estimator, types:str):
    
    with open(f'{FLAGS.middle_output}/{FLAGS.task_name}_label2id.pkl', 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}

    predict_examples = processor.get_test_examples(FLAGS.data_dir, types)

    predict_file = os.path.join(FLAGS.output_dir, f"{types}_predict.tf_record")
    batch_tokens,batch_labels = filed_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

    log("***** Running prediction*****")
    log(f"  Num examples = {len(predict_examples)}")
    log(f"  Batch size = {FLAGS.predict_batch_size}")

    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)
    output_predict_file = f'{param.RESULT_PATH(param.SA_TYPE.NER)}_{types}_{time_str()}.txt'
    metric_path = f'{param.RESULT_PATH(param.SA_TYPE.NER)}_{types}_metric_{time_str()}.txt'
    #here if the tag is "X" means it belong to its before token, here for convenient evaluate use
    # conlleval.pl we  discarding it directly
    Writer(output_predict_file, result.copy(), batch_tokens, batch_labels, id2label, types)
    if types == 'Test':
        return
    if FLAGS.task_name == 'CWS':
        result = [int(ii < 3)  for ii in result]
        batch_labels = [int(ii < 3)  for ii in batch_labels]
        dump_bigger([batch_labels, result], f'{param.RESULT_PATH(param.SA_TYPE.NER)}test.pkl')
        p, r, f1 = fastF1(batch_labels, result)
        log(f'{time_str()}|{types}|{p}|{r}|{f1}')
        return p, r, f1, 0

    result = conlleval(output_predict_file, metric_path)
    acc, p, r, f1, result_text = 0, 0, 0, 0, ''

    for ii, kk in enumerate(result):
        print(kk)
        if ii > 1:
            tp, tr, tf1 = re.findall('(\d{1,2}\.\d{2})%', kk)
            result_text += f'{tp}|{tr}|{tf1}|'
        elif ii == 1:
            acc, p, r, f1 = re.findall('(\d{1,2}\.\d{2})%', kk)
            result_text = f'{acc}|'
    log(f'{time_str()}|{types}|{p}|{r}|{f1}|{result_text}')
    return float(p), float(r), float(f1), result_text[:-1]

def main(_):
    if not os.path.exists(os.path.join(FLAGS.data_dir, f"Train_data_{FLAGS.task_name}.txt")):
        prepare_data('TXT')
    param.change_run_id(f'Bert_{FLAGS.task_name}')
    logging.set_verbosity(logging.INFO)
    processors = {"ner": NerProcessor}
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    task_name = FLAGS.task_name.lower()
    processor = processors['ner']()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)


    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        _,_ = filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        log("***** Running training *****")
        log(f"  Num examples = {len(train_examples)}")
        log(f"  Batch size = {FLAGS.train_batch_size}")
        log(f"  Num steps = {num_train_steps}")
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # evaluation(processor, label_list, tokenizer, estimator, 'Test')
    dev_p, dev_r, dev_macro_f1, dev_log = evaluation(processor, label_list, tokenizer, estimator, 'Dev')
    train_p, train_r, train_macro_f1, train_log = evaluation(processor, label_list, tokenizer, estimator, 'Train')

if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
