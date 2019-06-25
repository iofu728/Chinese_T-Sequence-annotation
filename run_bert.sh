#!/bin/bash
# @Author: gunjianpan
# @Date:   2019-06-24 22:47:56
# @Last Modified by:   v-huji
# @Last Modified time: 2019-06-25 12:05:38

ipython3 bert_sa.py -- \
    --task_name="CWS" \
    --do_lower_case=False \
    --crf=True \
    --do_train=False \
    --do_eval=False \
    --do_predict=True \
    --data_dir=data/pkl \
    --vocab_file=../bert/chinese_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=../bert/chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=../bert/chinese_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=2.0 \
    --output_dir=./result/CWSBert1
