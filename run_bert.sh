#!/bin/bash
# @Author: gunjianpan
# @Date:   2019-06-24 22:47:56
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-25 00:57:16

ipython3 BERT_SA.py -- \
    --task_name="CWS" \
    --do_lower_case=False \
    --crf=True \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --data_dir=data/pkl \
    --vocab_file=/Users/gunjianpan/Desktop/git/bert/chinese_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=/Users/gunjianpan/Desktop/git/bert/chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/Users/gunjianpan/Desktop/git/bert/chinese_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=2.0 \
    --output_dir=./result/BERT
