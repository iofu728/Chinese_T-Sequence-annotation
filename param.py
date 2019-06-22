# -*- coding: utf-8 -*-
# @Author: v-huji
# @Date:   2019-06-21 10:24:51
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-22 16:19:17

from enum import Enum


class SA_TYPE(Enum):
    CWS = 0
    NER = 1


def getFileName(sa_type: SA_TYPE) -> str:
    return 'seg' if SA_TYPE.CWS == sa_type else 'ner'


DATA_DIR = 'data/'
PKL_DIR = 'data/pkl/'
log_path = 'service.log'
is_service = False

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
