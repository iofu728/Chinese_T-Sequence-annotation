# Chinese Traditional Sequence Annotation

## Design Idea

- BiLSTM + CRF(baseline)
- Bert
- Bert + BiLSTM + CRF

## Data Distribution

### NER

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

#### Test for Batch Size

```json
other param = {
    'Model' = 'BiLSTM CRF',
    'Hidden Size' = 512,
    'Embed Size' = 256
}
```

| Batch Size | Train P | Train R | Train F1 | Dev P | Dev R | Dev Test | Best Epoch |
| ---------- | ------- | ------- | -------- | ----- | ----- | -------- | ---------- |
| 32         | 91.15   | 91.30   | 91.23    | 89.77 | 89.97 | 89.87    | 7.8        |
| 64         | 93.53   | 93.54   | 93.53    | 91.65 | 91.78 | 91.71    | 8.8        |
| 128        | 95.34   | 95.15   | 95.24    | 92.83 | 92.59 | 92.71    | 9.2        |
| 256        | 97.21   | 97.04   | 97.13    | 93.58 | 93.37 | 93.48    | 8.2        |
| 512        | 97.21   | 97.04   | 97.13    | 93.58 | 93.37 | 93.48    | 8.2        |
| 768        | 97.21   | 97.04   | 97.13    | 93.58 | 93.37 | 93.48    | 8.2        |
