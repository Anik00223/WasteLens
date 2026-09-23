# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_varc7_05.keras`

## 4-bin classification (supported test)
accuracy=0.9732 (baseline 0.9822), macro-F1=0.9595 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9798 | 0.9845 | 0.9822 |
| organic | 0.9667 | 0.9797 | 0.9732 |
| hazardous | 0.9441 | 0.9507 | 0.9474 |
| general trash | 0.9691 | 0.9038 | 0.9353 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9995 AUPRC=0.9997

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.0491 | 0.9959 | 0.0178 |
| 0.05 | 0.0042 | 0.9985 | 0.0535 |
| 0.10 | 0.0008 | 0.9995 | 0.1192 |
| 0.15 | 0.0003 | 0.9995 | 0.1768 |
| 0.20 | 0.0002 | 1.0000 | 0.2466 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9966 |
| collage | 0.9889 |
| nonwaste | 0.9987 |
