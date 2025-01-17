# Chinese Traditional Sequence Annotation

- [x] [final paper](https://github.com/iofu728/Chinese_T-Sequence-annotation/blob/master/final_paper/main.pdf)
- [x] [final result](https://github.com/iofu728/Chinese_T-Sequence-annotation/tree/master/public_result)
- [x] bert + crf trained model - [Google Drive 1](https://drive.google.com/file/d/1YcYAdRhqIoGXD8hl70Tl8MsfnXrkCjfS/view), [Google Drive 2](https://drive.google.com/file/d/16h-vdLkq1Hmihgr4bejTSo6dhqhQAl4L/view), [BaiduYun(rjoa)](https://pan.baidu.com/s/1XgRUQI29l9-MpWbtbpxxXw)

## Design Idea

- [x] BiLSTM + CRF(baseline)
- [x] Bert + CRF

## Model Structure

![ModelStructure](https://cdn.nlark.com/yuque/0/2019/png/104214/1561613165004-639f2a76-5816-4e66-891a-f1d264642c37.png)

## SetUp

```bash
git clone https://github.com/iofu728/Chinese_T-Sequence-annotation
cd Chinese_T-Sequence-annotation && git clone https://github.com/google-research/bert
pip install -r requirement.txt --user

## for BiLSTM + CRF
python run.py

## for Bert + CRF
bash run_bert.sh
```

## Final result

### For CWS

| Model        | Train P | Train R | Train F1 | Dev P | Dev R | Dev F1    | Best Epoch   |
| ------------ | ------- | ------- | -------- | ----- | ----- | --------- | ------------ |
| BiLSTM + CRF | 98.36   | 98.42   | 98.39    | 93.70 | 93.82 | **93.76** | 12           |
| Bert + CRF   | -       | -       | -        | 97.97 | 97.60 | **97.78** | 1(only test) |

### For NER

> Table 1 for P, R, F1 for common way

| Model        | Train P | Train R | Train F1 | Dev P | Dev R | Dev F1    | Best Epoch |
| ------------ | ------- | ------- | -------- | ----- | ----- | --------- | ---------- |
| BiLSTM + CRF | 99.75   | 99.76   | 99.76    | 86.95 | 81.30 | **84.03** | 30         |
| Bert + CRF   | -       | -       | -        | 96.62 | 97.23 | **96.92** | 3          |

> Table 2 for domain performance

| Model        | Train Acc | Loc P | Loc R | Loc F1 | Org P | Org R | Org F1 | Per P | Per R | Per F1 | Test Acc | Loc P | Loc R | Loc F1 | Org P | Org R | Org F1 | Per P | Per R | Per F1 |
| ------------ | --------- | ----- | ----- | ------ | ----- | ----- | ------ | ----- | ----- | ------ | -------- | ----- | ----- | ------ | ----- | ----- | ------ | ----- | ----- | ------ |
| BiLSTM + CRF | 99.98     | 99.81 | 99.84 | 99.83  | 99.68 | 99.73 | 99.70  | 99.67 | 99.64 | 99.65  | 97.84    | 87.62 | 85.33 | 86.46  | 83.46 | 70.94 | 76.69  | 87.99 | 80.68 | 84.17  |
| Bert + CRF   | -         | -     | -     | -      | -     | -     | -      | -     | -     | -      | 99.70    | 98.18 | 98.44 | 98.31  | 93.42 | 95.56 | 94.48  | 98.27 | 98.84 | 98.55  |

## Data Distribution

### NER Data

| Type  | Train Set Num | Train Set Percent % | Dev Set Num | Dev Set Percent % |
| ----- | ------------- | ------------------- | ----------- | ----------------- |
| N     | 1125991       | 90.63               | 479394      | 90.46             |
| B-LOC | 25211         | 2.03                | 11002       | 2.08              |
| I-LOC | 32022         | 2.58                | 13973       | 2.64              |
| B-ORG | 9428          | 0.76                | 4062        | 0.77              |
| I_ORG | 15220         | 1.23                | 6605        | 1.23              |
| B-PER | 11562         | 0.93                | 4990        | 0.94              |
| I-PER | 22858         | 1.84                | 9884        | 1.87              |

## Experiment

### CWS

```json
other param = {
    'Model' = 'BiLSTM CRF',
    'Hidden Size' = 512,
    'Embed Size' = 256,
    'learning rate' = 0.01
}
```

| Batch Size | Train P | Train R | Train F1 | Dev P | Dev R | Dev F1    | Best Epoch |
| ---------- | ------- | ------- | -------- | ----- | ----- | --------- | ---------- |
| 32         | 91.15   | 91.30   | 91.23    | 89.77 | 89.97 | 89.87     | 7.8        |
| 64         | 93.53   | 93.54   | 93.53    | 91.65 | 91.78 | 91.71     | 8.8        |
| 128        | 95.34   | 95.15   | 95.24    | 92.83 | 92.59 | 92.71     | 9.2        |
| 256        | 97.21   | 97.04   | 97.13    | 93.58 | 93.37 | 93.48     | 8.2        |
| 512        | 98.36   | 98.42   | 98.39    | 93.70 | 93.82 | **93.76** | 12.0       |
| 768        | 99.43   | 99.30   | 99.37    | 93.99 | 93.53 | **93.76** | 19.6       |

![CWSBatchChart](https://cdn.nlark.com/yuque/0/2019/png/104214/1561498478868-0e0937d8-97f8-49bb-9b20-2f89de61bb8f.png)

#### Hyperparameter of BiLSTM + CRF in CWS

```json
other param = {
    'Model' = 'BiLSTM CRF',
    'Batch Size' = 64,
    'learning rate' = 0.01
}
```

| Hidden Size | Embed Size | Train P | Train R | Train F1 | Dev P | Dev R | Dev F1    | Best Epoch |
| ----------- | ---------- | ------- | ------- | -------- | ----- | ----- | --------- | ---------- |
| 512         | 256        | 93.53   | 93.54   | 93.53    | 91.65 | 91.78 | 91.71     | 8.8        |
| 512         | 512        | 91.81   | 91.92   | 91.86    | 90.38 | 90.56 | 90.47     | 3.8        |
| 768         | 256        | 93.51   | 93.94   | 93.72    | 91.50 | 92.03 | **91.77** | 10.8       |

#### Bert + CRF in CWS

```json
other param = {
    'Model' = 'Bert CRF',
    'Batch Size' = 32,
    'learning rate' = 2e-5
}
```

| Dev P | Dev R | Dev F1    |
| ----- | ----- | --------- |
| 97.97 | 97.60 | **97.78** |

### NER

#### Batch Size of BiLSTM + CRF in NER

```json
other param = {
    'Model' = 'BiLSTM CRF',
    'Hidden Size' = 512,
    'Embed Size' = 256,
    'learning rate' = 0.001
}
```

> Table 1 for P, R, F1 for common way

| Batch Size | Train P | Train R | Train F1 | Dev P | Dev R | Dev F1    | Best Epoch |
| ---------- | ------- | ------- | -------- | ----- | ----- | --------- | ---------- |
| 64         | 99.75   | 99.76   | 99.76    | 86.95 | 81.30 | **84.03** | 30         |
| 256        | 97.75   | 96.59   | 97.17    | 84.56 | 79.27 | 81.83     | 33         |
| 512        | 94.75   | 92.12   | 93.42    | 83.54 | 78.14 | 80.75     | 36         |
| 768        | 88.41   | 83.74   | 86.02    | 81.36 | 75.80 | 78.48     | 38         |

> Table 2 for domain performance

| Batch Size | Train Acc | Loc P | Loc R | Loc F1 | Org P | Org R | Org F1 | Per P | Per R | Per F1 | Test Acc | Loc P | Loc R | Loc F1 | Org P | Org R | Org F1 | Per P | Per R | Per F1 |
| ---------- | --------- | ----- | ----- | ------ | ----- | ----- | ------ | ----- | ----- | ------ | -------- | ----- | ----- | ------ | ----- | ----- | ------ | ----- | ----- | ------ |
| 64         | 99.98     | 99.81 | 99.84 | 99.83  | 99.68 | 99.73 | 99.70  | 99.67 | 99.64 | 99.65  | 97.84    | 87.62 | 85.33 | 86.46  | 83.46 | 70.94 | 76.69  | 87.99 | 80.68 | 84.17  |
| 256        | 99.72     | 97.57 | 97.08 | 97.33  | 97.21 | 93.82 | 95.48  | 98.57 | 97.78 | 98.17  | 97.61    | 85.75 | 83.40 | 84.56  | 81.62 | 68.27 | 74.35  | 83.92 | 78.94 | 81.35  |
| 512        | 99.31     | 94.25 | 93.26 | 93.75  | 94.67 | 86.68 | 90.50  | 95.89 | 94.09 | 94.98  | 97.54    | 84.38 | 82.73 | 83.55  | 81.56 | 66.35 | 73.17  | 82.94 | 77.44 | 80.10  |
| 768        | 98.44     | 88.20 | 86.48 | 87.33  | 86.64 | 72.98 | 79.22  | 90.16 | 86.59 | 97.34  | 97.34    | 82.01 | 79.92 | 80.95  | 79.42 | 64.26 | 71.04  | 81.22 | 75.95 | 78.49  |

![NERBatchChart](https://cdn.nlark.com/yuque/0/2019/png/104214/1561513101185-4a8c6ff1-0eb7-4ae5-93c8-c67c7542f08e.png)

#### Hyperparameter of BiLSTM + CRF in NER

```json
other param = {
    'Model' = 'BiLSTM CRF',
    'Batch Size' = 64,
    'learning rate' = 0.001
}
```

> Table 1 for P, R, F1 for common way

| Hidden Size | Embed Size | Train P | Train R | Train F1 | Dev P | Dev R | Dev F1    | Best Epoch |
| ----------- | ---------- | ------- | ------- | -------- | ----- | ----- | --------- | ---------- |
| 512         | 256        | 99.75   | 99.76   | 99.76    | 86.95 | 81.30 | 84.03     | 30         |
| 300         | 300        | 99.00   | 97.76   | 98.38    | 87.38 | 81.10 | **84.13** | 25         |

> Table 2 for domain performance

| Hidden Size | Embed Size | Train Acc | Loc P | Loc R | Loc F1 | Org P | Org R | Org F1 | Per P | Per R | Per F1 | Test Acc | Loc P | Loc R | Loc F1 | Org P | Org R | Org F1 | Per P | Per R | Per F1 |
| ----------- | ---------- | --------- | ----- | ----- | ------ | ----- | ----- | ------ | ----- | ----- | ------ | -------- | ----- | ----- | ------ | ----- | ----- | ------ | ----- | ----- | ------ |
| 512         | 256        | 99.98     | 99.81 | 99.84 | 99.83  | 99.68 | 99.73 | 99.70  | 99.67 | 99.64 | 99.65  | 97.84    | 87.62 | 85.33 | 86.46  | 83.46 | 70.94 | 76.69  | 87.99 | 80.68 | 84.17  |
| 300         | 300        | 99.81     | 99.02 | 98.31 | 98.67  | 98.46 | 96.17 | 97.30  | 99.41 | 97.87 | 98.64  | 97.87    | 87.68 | 85.06 | 86.35  | 84.10 | 71.19 | 77.11  | 89.20 | 80.29 | 84.51  |

#### Bert + CRF in NER

```json
other param = {
    'Model' = 'Bert CRF',
    'Batch Size' = 32,
    'learning rate' = 2e-5
}
```

| Epoch | Dev Acc | Dev P | Dev R | Dev F1    | Loc P | Loc R | Loc F1 | Org P | Org R | Org F1 | Per P | Per R | Per F1 |
| ----- | ------- | ----- | ----- | --------- | ----- | ----- | ------ | ----- | ----- | ------ | ----- | ----- | ------ |
| 1     | 99.63   | 95.28 | 96.42 | 95.85     | 97.53 | 97.83 | 97.68  | 89.99 | 93.85 | 91.88  | 97.49 | 98.62 | 98.05  |
| 2     | 99.67   | 96.30 | 96.97 | 96.64     | 97.97 | 98.08 | 98.02  | 92.96 | 95.15 | 94.04  | 97.59 | 98.82 | 98.20  |
| 3     | 99.70   | 96.62 | 97.23 | **96.92** | 98.18 | 98.44 | 98.31  | 93.42 | 95.56 | 94.48  | 98.27 | 98.84 | 98.55  |

## License

[MIT](https://github.com/iofu728/Chinese_T-Sequence-annotation/blob/master/LICENSE)
