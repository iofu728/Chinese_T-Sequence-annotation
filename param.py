# -*- coding: utf-8 -*-
# @Author: v-huji
# @Date:   2019-06-21 10:24:51
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-23 14:49:40

import os
from enum import Enum
from util import time_str


class SA_TYPE(Enum):
    CWS = 0
    NER = 1


def getFileName(sa_type: SA_TYPE) -> str:
    return 'seg' if SA_TYPE.CWS == sa_type else 'ner'


DATA_DIR = 'data/'
CHECK_BASIC_DIR = 'checkpoint/'
CHECK_DIR = CHECK_BASIC_DIR
CHECK_PATH = f'{CHECK_DIR}checkpoint'
RESULT_ROOT_DIR = 'result/'
LOG_DIR = f'{RESULT_ROOT_DIR}log/'
LOG_PATH = f'{LOG_DIR}service.log'
PKL_DIR = f'{DATA_DIR}pkl/'
RESULT_DIR = RESULT_ROOT_DIR
is_service = False
run_id = ''

CWS_LAB2ID = {
    'B': 0,
    'M': 1,
    'E': 2,
    'S': 3
}

NER_LAB2ID = {
    'B-PER': 0,
    'N': 1,
    'I-LOC': 2,
    'B-ORG': 3,
    'B-LOC': 4,
    'I-PER': 5,
    'I-ORG': 6
}


def ORIGIN_SET_PATH(train_type) -> dict:
    return {
        'CWS': f'{DATA_DIR}{train_type}_utf16.seg',
        'NER': f'{DATA_DIR}{train_type}_utf16.ner'
    }


def PKL_SET_PATH(train_type: str) -> dict:
    return {
        'CWS': f'{PKL_DIR}{train_type}_seg.pkl',
        'NER': f'{PKL_DIR}{train_type}_ner.pkl'
    }


def RESULT_PATH(sa_type: SA_TYPE) -> str:
    return f'{RESULT_DIR}cws_' if sa_type == SA_TYPE.CWS else f'{RESULT_DIR}ner_'


if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


def change_run_id(new_run_id: str):
    global run_id, RESULT_DIR, LOG_PATH
    run_id = new_run_id
    RESULT_DIR = f'{RESULT_ROOT_DIR}{run_id}_{time_str()}/'
    os.mkdir(RESULT_DIR)
    LOG_PATH = f'{LOG_DIR}{run_id}_service_{time_str()}.log'
    CHECK_DIR = f'{CHECK_BASIC_DIR}{run_id}_{time_str()}/'
    os.mkdir(CHECK_DIR)
    CHECK_PATH = f'{CHECK_DIR}checkpoint'
