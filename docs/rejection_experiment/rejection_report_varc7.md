# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_varc7_03.keras`

## 4-bin classification (supported test)
accuracy=0.9667 (baseline 0.9822), macro-F1=0.9536 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9914 | 0.9607 | 0.9758 |
| organic | 0.9608 | 0.9932 | 0.9767 |
| hazardous | 0.9189 | 0.9577 | 0.9379 |
| general trash | 0.8655 | 0.9904 | 0.9238 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9995 AUPRC=0.9997

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.5287 | 0.9943 | 0.0154 |
| 0.05 | 0.0923 | 0.9979 | 0.0535 |
| 0.10 | 0.0264 | 1.0000 | 0.1160 |
| 0.15 | 0.0134 | 1.0000 | 0.1598 |
| 0.20 | 0.0067 | 1.0000 | 0.2084 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9966 |
| collage | 0.9778 |
| nonwaste | 0.9987 |
