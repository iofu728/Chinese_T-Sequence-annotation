# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-06-22 23:39:03
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-06-24 02:37:38

import param
from util import echo
import dataLoad
from model import Model


def main(sa_type: param.SA_TYPE, run_id: str):
    ''' run model '''
    param.change_run_id(run_id)
    echo(1, '....... Load data ......')
    train, dev, test = dataLoad.load_data(sa_type)
    echo(1, '....... Data load Over ......')
    sa_model = Model(train, dev, test, sa_type)
    echo(1, '------ Begin Train -------')
    sa_model.run_model()


if __name__ == "__main__":
    main(param.SA_TYPE.NER, 'NER_BILSTM_CRF')
