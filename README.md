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
