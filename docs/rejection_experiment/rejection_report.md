# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_best.keras`

## 4-bin classification (supported test)
accuracy=0.9716 (baseline 0.9822), macro-F1=0.9599 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9975 | 0.9642 | 0.9806 |
| organic | 0.9130 | 0.9932 | 0.9515 |
| hazardous | 0.9276 | 0.9930 | 0.9592 |
| general trash | 0.9266 | 0.9712 | 0.9484 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9997 AUPRC=0.9998

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.3101 | 0.9969 | 0.0130 |
| 0.05 | 0.0137 | 0.9990 | 0.0616 |
| 0.10 | 0.0025 | 1.0000 | 0.1192 |
| 0.15 | 0.0009 | 1.0000 | 0.1719 |
| 0.20 | 0.0004 | 1.0000 | 0.2287 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9966 |
| collage | 1.0000 |
| nonwaste | 0.9987 |
