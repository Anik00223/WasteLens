# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_varc7_10.keras`

## 4-bin classification (supported test)
accuracy=0.9789 (baseline 0.9822), macro-F1=0.9698 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9788 | 0.9917 | 0.9852 |
| organic | 0.9862 | 0.9662 | 0.9761 |
| hazardous | 0.9710 | 0.9437 | 0.9571 |
| general trash | 0.9800 | 0.9423 | 0.9608 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9995 AUPRC=0.9997

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.2261 | 0.9943 | 0.0130 |
| 0.05 | 0.0081 | 0.9985 | 0.0560 |
| 0.10 | 0.0010 | 0.9990 | 0.1160 |
| 0.15 | 0.0003 | 1.0000 | 0.1687 |
| 0.20 | 0.0001 | 1.0000 | 0.2247 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9966 |
| collage | 0.9889 |
| nonwaste | 0.9987 |
