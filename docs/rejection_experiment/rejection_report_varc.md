# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_varc_best.keras`

## 4-bin classification (supported test)
accuracy=0.9611 (baseline 0.9822), macro-F1=0.9460 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9950 | 0.9535 | 0.9738 |
| organic | 0.8605 | 1.0000 | 0.9250 |
| hazardous | 0.9195 | 0.9648 | 0.9416 |
| general trash | 0.9259 | 0.9615 | 0.9434 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9996 AUPRC=0.9997

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.3125 | 0.9943 | 0.0146 |
| 0.05 | 0.0396 | 0.9985 | 0.0600 |
| 0.10 | 0.0109 | 1.0000 | 0.1184 |
| 0.15 | 0.0051 | 1.0000 | 0.1671 |
| 0.20 | 0.0025 | 1.0000 | 0.2198 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9966 |
| collage | 0.9889 |
| nonwaste | 0.9987 |
