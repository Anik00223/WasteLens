# WasteLens Iteration 4 - rejection experiment report

Model: `models\checkpoints\wastelens_rej_varc7_06.keras`

## 4-bin classification (supported test)
accuracy=0.9773 (baseline 0.9822), macro-F1=0.9658 (baseline 0.9742)

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9892 | 0.9797 | 0.9844 |
| organic | 0.9545 | 0.9932 | 0.9735 |
| hazardous | 0.9524 | 0.9859 | 0.9689 |
| general trash | 0.9505 | 0.9231 | 0.9366 |

## Rejection (thresholds frozen on VAL)
AUROC=0.9996 AUPRC=0.9998

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.1811 | 0.9948 | 0.0146 |
| 0.05 | 0.0113 | 0.9985 | 0.0535 |
| 0.10 | 0.0016 | 1.0000 | 0.1184 |
| 0.15 | 0.0006 | 1.0000 | 0.1833 |
| 0.20 | 0.0003 | 1.0000 | 0.2384 |

### Per-source OOD detection @5% FRR target
| source | detect |
|---|---|
| clothes | 1.0000 |
| shoes | 0.9966 |
| collage | 0.9889 |
| nonwaste | 0.9987 |
